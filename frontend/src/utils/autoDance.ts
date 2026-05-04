import type { AgentStateResponse } from "../types/agent";

export const AUTO_DANCE_LOOKAHEAD_SEC = 45;
export const AUTO_DANCE_TICK_MS = 2500;
export const AUTO_DANCE_STABLE_INTERVAL_MS = 15_000;
export const AUTO_DANCE_MIN_GAP_MS = 12_000;
export const AUTO_DANCE_SECTION_NEAR_SEC = 8;
export const AUTO_DANCE_RECENT_HUMAN_STEER_MS = 60_000;

export interface AutoDanceLastCheckpoint {
  wallTimeMs: number;
  audioTime: number;
  sectionIndex: number | null;
}

export interface AutoDanceTriggerContext {
  enabled: boolean;
  bridgeConnected: boolean;
  frontendConnected: boolean;
  submitting: boolean;
  state: AgentStateResponse | null;
  nowWallTimeMs: number;
  last: AutoDanceLastCheckpoint;
}

export interface AutoDanceDecision {
  shouldTrigger: boolean;
  reason: "section_change" | "section_near" | "stable_interval" | "ineligible";
  audioTime: number;
  sectionIndex: number | null;
  secondsToNextSection: number | null;
}

export interface AutoDanceDirectiveContext {
  recentHumanDirective?: string | null;
  recentHumanDirectiveAgeMs?: number | null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function compactDirectiveText(text: string, maxLength = 180) {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length > maxLength
    ? `${normalized.slice(0, maxLength - 1)}...`
    : normalized;
}

export function sectionsFromState(state: AgentStateResponse | null): number[] {
  const raw = state?.song_profile?.sections;
  return Array.isArray(raw)
    ? raw.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    : [];
}

export function sectionInfo(
  sections: number[],
  duration: number | null,
  audioTime: number,
) {
  if (sections.length === 0) {
    return { sectionIndex: null, secondsToNextSection: null };
  }

  let sectionIndex = 0;
  for (let i = 0; i < sections.length; i += 1) {
    if (sections[i] <= audioTime) sectionIndex = i;
    else break;
  }

  const nextBoundary = sections[sectionIndex + 1] ?? duration;
  return {
    sectionIndex,
    secondsToNextSection: nextBoundary == null ? null : Math.max(0, nextBoundary - audioTime),
  };
}

export function shouldTriggerAutoDance(context: AutoDanceTriggerContext): AutoDanceDecision {
  const audioTime = numberOrNull(context.state?.entry_context?.audio.current_time) ?? 0;
  const duration = numberOrNull(context.state?.entry_context?.audio.duration);
  const sections = sectionsFromState(context.state);
  const { sectionIndex, secondsToNextSection } = sectionInfo(sections, duration, audioTime);
  const ineligible: AutoDanceDecision = {
    shouldTrigger: false,
    reason: "ineligible",
    audioTime,
    sectionIndex,
    secondsToNextSection,
  };

  if (
    !context.enabled ||
    !context.bridgeConnected ||
    !context.frontendConnected ||
    context.submitting ||
    !context.state?.armed ||
    !context.state.active_session ||
    !context.state.entry_context?.audio.is_playing ||
    !context.state.entry_context.audio.audio_id_present ||
    !context.state.song_analysis_available
  ) {
    return ineligible;
  }

  const enoughGap = context.nowWallTimeMs - context.last.wallTimeMs >= AUTO_DANCE_MIN_GAP_MS;
  if (!enoughGap) return ineligible;

  if (
    sectionIndex != null &&
    context.last.sectionIndex != null &&
    sectionIndex !== context.last.sectionIndex
  ) {
    return {
      shouldTrigger: true,
      reason: "section_change",
      audioTime,
      sectionIndex,
      secondsToNextSection,
    };
  }

  if (
    sectionIndex != null &&
    context.last.sectionIndex !== sectionIndex &&
    secondsToNextSection != null &&
    secondsToNextSection <= AUTO_DANCE_SECTION_NEAR_SEC
  ) {
    return {
      shouldTrigger: true,
      reason: "section_near",
      audioTime,
      sectionIndex,
      secondsToNextSection,
    };
  }

  if (context.nowWallTimeMs - context.last.wallTimeMs >= AUTO_DANCE_STABLE_INTERVAL_MS) {
    return {
      shouldTrigger: true,
      reason: "stable_interval",
      audioTime,
      sectionIndex,
      secondsToNextSection,
    };
  }

  return ineligible;
}

export function buildAutoDanceDirective(
  decision: AutoDanceDecision,
  context: AutoDanceDirectiveContext = {},
) {
  const sectionText = decision.sectionIndex == null
    ? "unknown section"
    : `section ${decision.sectionIndex}`;
  const nextText = decision.secondsToNextSection == null
    ? "unknown seconds to next section"
    : `${decision.secondsToNextSection.toFixed(1)}s to next section`;
  const steerText = context.recentHumanDirective?.trim()
    ? `Recent human steer still active (${Math.max(0, Math.round((context.recentHumanDirectiveAgeMs ?? 0) / 1000))}s ago): "${compactDirectiveText(context.recentHumanDirective)}". Preserve this as a constraint; Auto Dance may evolve around it but must not undo it.`
    : null;

  return [
    `Auto Dance checkpoint: ${decision.reason} at ${decision.audioTime.toFixed(2)}s (${sectionText}, ${nextText}).`,
    "You are not responding to a human scene request. You are dancing Hamba forward on your own.",
    steerText,
    "Required tool order: call hamba_get_state, then hamba_get_song_analysis, then hamba_get_music_window with lookahead=45, then hamba_get_feature_palette before any apply.",
    "If hamba_get_feature_palette says available=false, call hamba_prepare_feature_palette with the current visual theme and this directive divergence, then call hamba_get_feature_palette again.",
    "Do not do open-ended live feature hunting during Auto Dance. Use prepared palette candidates first; at most one targeted hamba_search_features call is allowed only if the palette is empty or a required user anchor is missing.",
    "Use the music window's ranked_window_targets, target_windows, auto_dance_hints, and the whole-song section_target_summary as DSP truth.",
    "Make at least one visible evolution every checkpoint, usually 1-3 coordinated visible mutations. Do not hold the rig just because the music is stable; stable groove means invent a new visual chapter inside the same pulse. Demo cadence is about every 15 seconds, so move decisively.",
    "At high divergence, dance harder: change latent/composition motion plus at least one feature, prompt endpoint, link target, intensity source/curve, or strength range.",
    "Schema reminder: change prompt endpoints with separate `set_destination` actions for prompt slot a/b; never send `set_prompt`, `prompt_a`, or `prompt_b`. Do not put `target` inside `set_reactive_config`.",
    "Prefer unused palette candidates. If the user recently made a major vibe change, prepare/reset a new palette epoch; if it was a small steer, keep the current palette.",
    "Use `sae_rank: 1` for every enabled SAE block; create subtlety with strength ranges, masks, curves, and link targets, not lower ranks.",
    "Do not reset the whole scene unless the current setup is blank or broken. Transform the existing visual character rather than replacing the world, but the change must be visible.",
    "Do not send `based_on_audio_time`, `based_on_wall_time_ms`, or `max_staleness_sec`; these are durable steering changes applied when ready, not beat-perfect cues.",
  ].filter(Boolean).join("\n");
}
