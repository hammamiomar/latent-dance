import { useCallback, useEffect, useRef, useState } from "react";
import { AGENT_BRIDGE_WS_URL, IS_DESKTOP_MODE } from "../constants";
import { useAgentStore } from "../stores/useAgentStore";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import { useAudioStore } from "../stores/useAudioStore";
import { useBlockStore } from "../stores/useBlockStore";
import { useCompositionStore } from "../stores/useCompositionStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import { useSongIntelligenceStore } from "../stores/useSongIntelligenceStore";
import { buildControlSurface } from "../data/controlSurface";
import { applyAgentVisualAction, type AgentPlanApplySenders } from "../utils/agentPlanApply";
import { buildControlState } from "../utils/controlState";
import { validateAgentVisualPlan } from "../utils/agentPlanValidation";
import {
  findSectionIndex,
  sampleCurve,
  sampleLockIndexCurves,
  sampleTrend,
  sampleWindowStats,
} from "../utils/curveSampling";
import type {
  AgentBridgeController,
  AgentBridgeRequest,
  AgentBridgeStatus,
  AgentVisualPlan,
} from "../types/agentBridge";
import type { AgentEntryContext, AgentEvent, AgentPhase } from "../types/agent";
import type { ConnectionStatus } from "../types";
import type { LinkTarget, StemChannelData } from "../types/sae";

const REQUEST_TIMEOUT_MS = 120_000;
const RECONNECT_DELAY_MS = 1_000;
const TARGET_WINDOW_CHANNELS = [
  "energy_smooth",
  "transient",
  "flux",
  "brightness",
  "sustain",
  "pitch_confidence",
  "pitch_normalized",
  "chroma_centroid",
  "tension",
  "tonal_distance",
  "novelty_long",
] as const;
const WINDOW_RANKINGS = {
  primary_energy: "energy_smooth",
  rhythmic_hits: "transient",
  texture_motion: "flux",
  bright_air: "brightness",
  sustain_body: "sustain",
} as const;

interface UseAgentBridgeOptions extends AgentPlanApplySenders {
  generationStatus: ConnectionStatus;
  isGenerating: boolean;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: number;
}

function requestId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function bridgeError(message: string) {
  return new Error(message);
}

function validationErrorMessage(errors: Array<{ path: string; message: string }>) {
  return errors
    .map((item) => item.path ? `${item.path}: ${item.message}` : item.message)
    .join("; ");
}

function assertObject(payload: unknown, label: string): asserts payload is Record<string, unknown> {
  if (!payload || typeof payload !== "object") {
    throw bridgeError(`${label} must be an object`);
  }
}

function sendNotification(ws: WebSocket | null, type: string, payload: unknown) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type, payload }));
}

function compactMusicSnapshot() {
  const activity = useAudioActivityStore.getState();
  return {
    time: activity.audioTime,
    stems: activity.stems,
    prominence: activity.prominence ?? {},
    blocks: activity.blocks ?? {},
    last_update_time: activity.lastUpdateTime,
    is_receiving: activity.isReceiving,
  };
}

function roundMetric(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
    ? Number(value.toFixed(4))
    : 0;
}

function compactCurrentStem(channels: {
  energy_smooth?: number;
  transient?: number;
  flux?: number;
  brightness?: number;
  flash?: number;
  sustain?: number;
}) {
  return {
    energy_smooth: roundMetric(channels.energy_smooth),
    transient: roundMetric(channels.transient),
    flux: roundMetric(channels.flux),
    brightness: roundMetric(channels.brightness),
    flash: roundMetric(channels.flash),
    sustain: roundMetric(channels.sustain),
  };
}

function compactWindowStats(stats: ReturnType<typeof sampleWindowStats>) {
  return {
    start: roundMetric(stats.start),
    end: roundMetric(stats.end),
    mean: roundMetric(stats.mean),
    min: roundMetric(stats.min),
    max: roundMetric(stats.max),
    trend: stats.trend,
  };
}

function buildTargetWindows(
  currentTime: number,
  lookback: number,
  lookahead: number,
) {
  const activity = useAudioActivityStore.getState();
  const activityStems = activity.stems as Partial<Record<LinkTarget, StemChannelData>>;
  const intelligence = useSongIntelligenceStore.getState();
  const profiles = intelligence.analysis?.link_targets ?? {};
  const targetCurves = intelligence.curves.targetCurves;
  const start = Math.max(0, currentTime - lookback);
  const end = Math.min(
    intelligence.profile?.duration ?? Number.POSITIVE_INFINITY,
    currentTime + lookahead,
  );
  const targets = Array.from(new Set([
    ...Object.keys(profiles),
    ...Object.keys(activity.stems),
    ...Object.keys(targetCurves),
  ])).sort() as LinkTarget[];

  return Object.fromEntries(
    targets.map((target) => {
      const channels = activityStems[target] ?? {};
      const curveChannels = targetCurves[target] ?? {};
      const windows = Object.fromEntries(
        TARGET_WINDOW_CHANNELS
          .filter((channel) => curveChannels[channel])
          .map((channel) => [
            channel,
            {
              recent: compactWindowStats(sampleWindowStats(
                curveChannels[channel],
                intelligence.timestamps,
                currentTime,
                start,
                currentTime,
              )),
              upcoming: compactWindowStats(sampleWindowStats(
                curveChannels[channel],
                intelligence.timestamps,
                currentTime,
                currentTime,
                end,
              )),
            },
          ]),
      );
      return [
        target,
        {
          current: compactCurrentStem(channels),
          prominence: activity.prominence?.[target]?.prominence ?? null,
          windows,
          global_profile: profiles[target]
            ? {
                movement_words: profiles[target].movement_words,
                good_for: profiles[target].good_for,
                preferred_intensity_source: profiles[target].preferred_intensity_source,
                position_source_affordances: profiles[target].position_source_affordances,
                stats: profiles[target].stats,
              }
            : null,
        },
      ];
    }),
  );
}

function buildRankedWindowTargets(
  targetWindows: Record<string, any>,
) {
  const rankChannel = (channel: string) => Object.entries(targetWindows)
    .map(([target, window]) => {
      const stats = window?.windows?.[channel];
      if (!stats) return null;
      const score = 0.65 * Number(stats.upcoming?.max ?? 0) + 0.35 * Number(stats.upcoming?.mean ?? 0);
      return {
        target,
        channel,
        score: roundMetric(score),
        recent_mean: roundMetric(stats.recent?.mean),
        upcoming_mean: roundMetric(stats.upcoming?.mean),
        upcoming_max: roundMetric(stats.upcoming?.max),
        delta: roundMetric(Number(stats.upcoming?.mean ?? 0) - Number(stats.recent?.mean ?? 0)),
      };
    })
    .filter((item): item is NonNullable<typeof item> => item != null && item.score > 0.03)
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

  const ranked = Object.fromEntries(
    Object.entries(WINDOW_RANKINGS).map(([name, channel]) => [name, rankChannel(channel)]),
  );
  const rising_targets = Object.values(WINDOW_RANKINGS)
    .flatMap((channel) => rankChannel(channel))
    .sort((a, b) => b.delta - a.delta)
    .slice(0, 6);
  return { ...ranked, rising_targets };
}

function buildAggregateWindows(
  currentTime: number,
  lookback: number,
  lookahead: number,
) {
  const intelligence = useSongIntelligenceStore.getState();
  const start = Math.max(0, currentTime - lookback);
  const end = Math.min(
    intelligence.profile?.duration ?? Number.POSITIVE_INFINITY,
    currentTime + lookahead,
  );
  const build = (curve: Float32Array | null) => ({
    recent: sampleWindowStats(curve, intelligence.timestamps, currentTime, start, currentTime),
    upcoming: sampleWindowStats(curve, intelligence.timestamps, currentTime, currentTime, end),
  });

  return {
    tension: build(intelligence.curves.tension),
    tonal_distance: build(intelligence.curves.tonal_distance),
    novelty_long: build(intelligence.curves.novelty_long),
  };
}

function buildDominantTargets() {
  const activity = useAudioActivityStore.getState();
  const intelligence = useSongIntelligenceStore.getState();
  const current = Object.entries(activity.stems)
    .map(([target, channels]) => ({
      target,
      energy_smooth: roundMetric(channels.energy_smooth),
      transient: roundMetric(channels.transient),
      flux: roundMetric(channels.flux),
    }))
    .sort((a, b) => b.energy_smooth - a.energy_smooth)
    .slice(0, 6);

  return {
    current,
    global_primary: intelligence.analysis?.ranked_drivers?.primary_driver?.slice(0, 6) ?? [],
    rhythmic_hits: intelligence.analysis?.ranked_drivers?.rhythmic_hits?.slice(0, 6) ?? [],
    prompt_position: intelligence.analysis?.ranked_drivers?.prompt_position?.slice(0, 6) ?? [],
  };
}

function buildAutoDanceHints(
  secondsRemaining: number,
  aggregateWindows: ReturnType<typeof buildAggregateWindows>,
  rankedWindowTargets: Record<string, unknown>,
) {
  const tensionRecent = aggregateWindows.tension.recent.mean;
  const tensionUpcoming = aggregateWindows.tension.upcoming.mean;
  const noveltyUpcoming = aggregateWindows.novelty_long.upcoming.max;
  const reasons: string[] = [];

  if (secondsRemaining <= 8) reasons.push("section_near");
  if (tensionUpcoming - tensionRecent > 0.08) reasons.push("tension_rising");
  if (noveltyUpcoming > 0.65) reasons.push("novelty_peak");
  if (reasons.length === 0) reasons.push("stable_window");

  return {
    should_consider_revision: true,
    visible_evolution_required: true,
    minimum_visible_mutations: 1,
    stable_window_policy:
      reasons[0] === "stable_window"
        ? "invent a new visual chapter; do not merely hold the rig"
        : "revise the rig for this window",
    revision_reasons: reasons,
    suggested_focus_targets: rankedWindowTargets,
  };
}

function buildMusicWindow(lookback: number, lookahead: number) {
  const audio = useAudioStore.getState();
  const blocks = useBlockStore.getState();
  const activity = useAudioActivityStore.getState();
  const intelligence = useSongIntelligenceStore.getState();
  const sampledAtAudioTime = audio.currentTime;
  const sampledAtWallTimeMs = Date.now();
  const hasProfile = Boolean(intelligence.profile);

  const base = {
    active_session: false,
    current_time: sampledAtAudioTime,
    sampled_at_audio_time: sampledAtAudioTime,
    sampled_at_wall_time_ms: sampledAtWallTimeMs,
    duration: audio.duration || intelligence.profile?.duration || null,
    bpm: intelligence.profile?.bpm ?? blocks.trackInfo?.bpm ?? null,
    is_playing: audio.isPlaying,
    lookback,
    lookahead,
    song_intelligence_available: hasProfile,
    song_analysis_available: Boolean(intelligence.analysis),
    song_metadata: intelligence.analysis?.metadata ?? {},
    snapshots: [compactMusicSnapshot()],
  };

  if (!intelligence.profile || !intelligence.timestamps) {
    return base;
  }

  const sectionIndex = findSectionIndex(intelligence.sections, sampledAtAudioTime);
  const sectionStart = intelligence.sections[sectionIndex] ?? 0;
  const sectionEnd = intelligence.sections[sectionIndex + 1] ?? intelligence.profile.duration;
  const secondsRemaining = Math.max(0, sectionEnd - sampledAtAudioTime);
  const aggregateWindows = buildAggregateWindows(sampledAtAudioTime, lookback, lookahead);
  const targetWindows = buildTargetWindows(sampledAtAudioTime, lookback, lookahead);
  const rankedWindowTargets = buildRankedWindowTargets(targetWindows);

  return {
    ...base,
    active_session: true,
    song_profile: intelligence.profile,
    section: {
      index: sectionIndex,
      start: sectionStart,
      end: sectionEnd,
      seconds_remaining: secondsRemaining,
    },
    at_current_time: {
      tension: sampleCurve(
        intelligence.curves.tension,
        intelligence.timestamps,
        sampledAtAudioTime,
      ),
      tension_trend: sampleTrend(
        intelligence.curves.tension,
        intelligence.timestamps,
        sampledAtAudioTime,
      ),
      tonal_distance: sampleCurve(
        intelligence.curves.tonal_distance,
        intelligence.timestamps,
        sampledAtAudioTime,
      ),
      novelty_long: sampleCurve(
        intelligence.curves.novelty_long,
        intelligence.timestamps,
        sampledAtAudioTime,
      ),
      coupling: sampleLockIndexCurves(
        intelligence.curves.lock_index,
        intelligence.timestamps,
        sampledAtAudioTime,
      ),
    },
    lookahead_context: {
      next_section_in: secondsRemaining,
      tension_at_next_section: sampleCurve(
        intelligence.curves.tension,
        intelligence.timestamps,
        sectionEnd,
      ),
      tonal_distance_at_next_section: sampleCurve(
        intelligence.curves.tonal_distance,
        intelligence.timestamps,
        sectionEnd,
      ),
    },
    window_summary: {
      section_index: sectionIndex,
      seconds_to_next_section: secondsRemaining,
      song_metadata: intelligence.analysis?.metadata ?? {},
      aggregate_windows: aggregateWindows,
      ranked_window_targets: rankedWindowTargets,
    },
    aggregate_windows: aggregateWindows,
    target_windows: targetWindows,
    ranked_window_targets: rankedWindowTargets,
    auto_dance_hints: buildAutoDanceHints(
      secondsRemaining,
      aggregateWindows,
      rankedWindowTargets,
    ),
    dominant_targets: buildDominantTargets(),
    stems: activity.stems,
    prominence: activity.prominence ?? {},
    block_configs: blocks.blockMappings,
  };
}

function buildSongAnalysis() {
  const intelligence = useSongIntelligenceStore.getState();
  return {
    available: Boolean(intelligence.analysis),
    audio_id: intelligence.audioId,
    sampled_at_wall_time_ms: Date.now(),
    analysis: intelligence.analysis,
  };
}

type ControlStateSnapshot = ReturnType<typeof buildControlState>;

function buildCurrentControlState(): ControlStateSnapshot {
  const blocks = useBlockStore.getState();
  const destinations = useDestinationStore.getState();
  const composition = useCompositionStore.getState();
  return buildControlState({
    blockMappings: blocks.blockMappings,
    latent: destinations.latent,
    prompt: destinations.prompt,
    composition,
  });
}

function isProcessingUploadPhase(phase: string) {
  return phase === "uploading" || phase === "processing" || phase === "loading_stems";
}

function buildAgentEntryContext(
  controlState: ControlStateSnapshot,
  generationStatus: ConnectionStatus,
  isGenerating: boolean,
): AgentEntryContext {
  const audio = useAudioStore.getState();
  const intelligence = useSongIntelligenceStore.getState();
  const summary = controlState.summary;
  const promptEmpty = summary.prompt.destination_a == null && summary.prompt.destination_b == null;
  const latentEmpty = summary.composition.seed_a == null && summary.composition.seed_b == null;
  const enabledBlockCount = summary.enabled_block_count;
  const freshBlankSetup = enabledBlockCount === 0 && promptEmpty && latentEmpty;
  const hasSong = Boolean(audio.audioId || audio.duration || intelligence.profile);

  let situation: AgentEntryContext["situation"];
  let situationSummary: string;
  let recommendedNextStep: string;

  if (isProcessingUploadPhase(audio.uploadPhase)) {
    situation = "song_processing";
    situationSummary = "Agent armed while song processing is still running";
    recommendedNextStep = "Wait for processing to finish before judging musical drivers; use state only for blank-rig setup.";
  } else if (isGenerating && audio.isPlaying) {
    situation = "visualizer_playing";
    situationSummary = "Agent armed during active playback";
    recommendedNextStep = "Read the music window before window-dependent changes; apply durable steering without timing metadata.";
  } else if (isGenerating) {
    situation = "visualizer_paused";
    situationSummary = "Agent armed while the visualizer is live but playback is paused";
    recommendedNextStep = "Use durable controls or read the current music window before changing timed behavior.";
  } else if (hasSong) {
    situation = "song_loaded_idle";
    situationSummary = freshBlankSetup
      ? "Agent armed with a loaded idle song and a blank visual setup"
      : "Agent armed with a loaded idle song and existing visual controls";
    recommendedNextStep = intelligence.analysis
      ? freshBlankSetup
        ? "Call hamba_get_song_analysis next before any visual opinions, feature search, or first-plan apply."
        : "Use whole-song analysis to build or refine the visual before playback."
      : audio.uploadPhase === "ready"
        ? "Song is ready, but frontend song analysis is missing; load song intelligence before a global first plan."
        : "Wait for song analysis or ask for visual direction before applying a first setup.";
  } else {
    situation = "no_song_loaded";
    situationSummary = "Agent armed with no song loaded";
    recommendedNextStep = "Ask the user to load a song or give a visual direction; avoid pretending to hear DSP.";
  }

  return {
    situation,
    summary: situationSummary,
    recommended_next_step: recommendedNextStep,
    audio: {
      audio_id_present: Boolean(audio.audioId),
      upload_phase: audio.uploadPhase,
      duration: audio.duration || null,
      current_time: audio.currentTime,
      is_playing: audio.isPlaying,
    },
    generation: {
      status: generationStatus,
      is_generating: isGenerating,
    },
    song_intelligence: {
      profile_available: Boolean(intelligence.profile),
      analysis_available: Boolean(intelligence.analysis),
    },
    control: {
      enabled_block_count: enabledBlockCount,
      prompt_empty: promptEmpty,
      latent_empty: latentEmpty,
      fresh_blank_setup: freshBlankSetup,
      composition: summary.composition,
    },
  };
}

function entryContextSignature(context: AgentEntryContext) {
  return JSON.stringify({
    situation: context.situation,
    audio_id_present: context.audio.audio_id_present,
    upload_phase: context.audio.upload_phase,
    duration: context.audio.duration,
    is_playing: context.audio.is_playing,
    generation_status: context.generation.status,
    is_generating: context.generation.is_generating,
    profile_available: context.song_intelligence.profile_available,
    analysis_available: context.song_intelligence.analysis_available,
    prompt_empty: context.control.prompt_empty,
    latent_empty: context.control.latent_empty,
    fresh_blank_setup: context.control.fresh_blank_setup,
    enabled_block_count: context.control.enabled_block_count,
  });
}

export function useAgentBridge(options: UseAgentBridgeOptions): AgentBridgeController {
  const [status, setStatus] = useState<AgentBridgeStatus>("idle");
  const wsRef = useRef<WebSocket | null>(null);
  const pendingRef = useRef(new Map<string, PendingRequest>());
  const reconnectRef = useRef<number | null>(null);
  const optionsRef = useRef(options);

  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const setBridgeStatus = useAgentStore((state) => state.setBridgeStatus);
  useEffect(() => {
    setBridgeStatus(status);
  }, [setBridgeStatus, status]);

  const sendResult = useCallback((id: string, payload: unknown) => {
    wsRef.current?.send(JSON.stringify({ id, type: "result", payload }));
  }, []);

  const sendError = useCallback((id: string | undefined, error: unknown) => {
    wsRef.current?.send(JSON.stringify({
      id,
      type: "error",
      error: {
        message: error instanceof Error ? error.message : String(error),
      },
    }));
  }, []);

  const broadcastAgentEvent = useCallback((event: AgentEvent) => {
    sendNotification(wsRef.current, "brain.agent_event", event);
  }, []);

  const request = useCallback((type: string, payload?: unknown) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(bridgeError("Agent bridge is not connected"));
    }

    const id = requestId();
    ws.send(JSON.stringify({ id, type, payload }));
    return new Promise<unknown>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        pendingRef.current.delete(id);
        reject(bridgeError(`Agent bridge request timed out: ${type}`));
      }, REQUEST_TIMEOUT_MS);
      pendingRef.current.set(id, { resolve, reject, timeout });
    });
  }, []);

  const buildState = useCallback(() => {
    const agent = useAgentStore.getState();
    const audio = useAudioStore.getState();
    const blocks = useBlockStore.getState();
    const destinations = useDestinationStore.getState();
    const composition = useCompositionStore.getState();
    const activity = useAudioActivityStore.getState();
    const songIntelligence = useSongIntelligenceStore.getState();
    const controlState = buildCurrentControlState();
    const entryContext = buildAgentEntryContext(
      controlState,
      optionsRef.current.generationStatus,
      optionsRef.current.isGenerating,
    );
    return {
      armed: agent.armed,
      mode: agent.mode,
      active_session: optionsRef.current.isGenerating,
      entry_context: entryContext,
      latest_event: agent.latestEvent,
      event_log: agent.events,
      generation: {
        status: optionsRef.current.generationStatus,
        is_generating: optionsRef.current.isGenerating,
      },
      audio: {
        audio_id: audio.audioId,
        duration: audio.duration,
        current_time: audio.currentTime,
        is_playing: audio.isPlaying,
        stems: audio.stems,
      },
      track_info: blocks.trackInfo,
      block_configs: blocks.blockMappings,
      control_state: controlState,
      destinations: {
        latent: destinations.latent,
        prompt: destinations.prompt,
      },
      composition: {
        distance: composition.distance,
        mode: composition.mode,
      },
      song_profile: songIntelligence.profile,
      song_analysis_available: Boolean(songIntelligence.analysis),
      activity: {
        audio_time: activity.audioTime,
        stems: activity.stems,
        prominence: activity.prominence ?? {},
        blocks: activity.blocks ?? {},
        is_receiving: activity.isReceiving,
      },
    };
  }, []);

  const applyVisualPlan = useCallback((payload: unknown) => {
    assertObject(payload, "Visual plan");
    const agent = useAgentStore.getState();
    const audio = useAudioStore.getState();
    const blocks = useBlockStore.getState();
    const validation = validateAgentVisualPlan(payload, {
      armed: agent.armed,
      activeSession: optionsRef.current.isGenerating,
      bridgeConnected: wsRef.current?.readyState === WebSocket.OPEN,
      currentAudioTime: audio.currentTime,
      mode: agent.mode,
      currentFeatureIdsByBlock: Object.fromEntries(
        Object.entries(blocks.blockMappings).map(([block, mapping]) => [block, mapping.featureId]),
      ),
    });
    if (!validation.ok) {
      const message = `Visual plan rejected: ${validationErrorMessage(validation.errors)}`;
      agent.setPhase("error", {
        summary: "Visual plan rejected",
        error: message,
      });
      throw bridgeError(message);
    }
    const plan: AgentVisualPlan = validation.plan;

    agent.setPhase("applying", {
      provider: plan.provider ?? undefined,
      model: plan.model ?? undefined,
      summary: plan.reason ?? undefined,
      feature_candidates: plan.feature_candidates ?? [],
    });
    const changes = plan.actions.map((action) => applyAgentVisualAction(action, optionsRef.current));
    agent.setPhase("watching", {
      provider: plan.provider ?? undefined,
      model: plan.model ?? undefined,
      summary: plan.reason ?? "Visual plan applied",
      feature_candidates: plan.feature_candidates ?? [],
      changes,
    });
    return { accepted: true, changes };
  }, []);

  const handleRequest = useCallback((message: AgentBridgeRequest) => {
    switch (message.type) {
      case "agent.get_state":
        return buildState();
      case "agent.get_control_surface":
        return buildControlSurface();
      case "agent.get_music_window": {
        const payload = (message.payload ?? {}) as { lookback?: number; lookahead?: number };
        return {
          ...buildMusicWindow(payload.lookback ?? 8, payload.lookahead ?? 16),
          active_session: optionsRef.current.isGenerating,
        };
      }
      case "agent.get_song_analysis":
        return buildSongAnalysis();
      case "agent.report_phase": {
        assertObject(message.payload, "Agent event");
        const event = message.payload;
        const phase = typeof event.phase === "string" ? event.phase : "thinking";
        useAgentStore.getState().setPhase(phase as AgentPhase, event);
        return { accepted: true };
      }
      case "agent.apply_visual_plan":
        return applyVisualPlan(message.payload);
      case "agent.set_armed": {
        const payload = (message.payload ?? {}) as { armed?: boolean; mode?: string };
        const armed = Boolean(payload.armed);
        const mode = payload.mode === "dj" ? "dj" : "directive";
        const controlState = buildCurrentControlState();
        const entryContext = buildAgentEntryContext(
          controlState,
          optionsRef.current.generationStatus,
          optionsRef.current.isGenerating,
        );
        useAgentStore.getState().setArmed(
          armed,
          armed ? mode : "off",
          armed
            ? {
                summary: entryContext.summary,
                changes: [{
                  action: "agent_entry_context",
                  target: "arm",
                  after: entryContext,
                }],
              }
            : undefined,
        );
        return { accepted: true, armed, mode: armed ? mode : "off", entry_context: entryContext };
      }
      default:
        throw bridgeError(`Unknown bridge request: ${message.type}`);
    }
  }, [applyVisualPlan, buildState]);

  useEffect(() => {
    if (!IS_DESKTOP_MODE) return;
    let lastEventId: string | null = null;
    return useAgentStore.subscribe((state) => {
      const event = state.latestEvent;
      if (!event || event.event_id === lastEventId) return;
      lastEventId = event.event_id;
      broadcastAgentEvent(event);
    });
  }, [broadcastAgentEvent]);

  useEffect(() => {
    if (!IS_DESKTOP_MODE) return;
    let lastSignature = "";
    let timeout: number | null = null;

    const emitEntryContext = () => {
      timeout = null;
      const agent = useAgentStore.getState();
      if (!agent.armed) return;

      const entryContext = buildAgentEntryContext(
        buildCurrentControlState(),
        optionsRef.current.generationStatus,
        optionsRef.current.isGenerating,
      );
      const signature = entryContextSignature(entryContext);
      if (signature === lastSignature) return;
      lastSignature = signature;

      agent.setPhase("armed", {
        summary: entryContext.summary,
        changes: [{
          action: "agent_entry_context",
          target: "state_change",
          after: entryContext,
        }],
      });
    };

    const scheduleEntryContext = () => {
      if (timeout !== null) window.clearTimeout(timeout);
      timeout = window.setTimeout(emitEntryContext, 0);
    };

    const unsubscribeAudio = useAudioStore.subscribe((state, previous) => {
      if (
        state.audioId !== previous.audioId ||
        state.uploadPhase !== previous.uploadPhase ||
        state.duration !== previous.duration ||
        state.isPlaying !== previous.isPlaying
      ) {
        scheduleEntryContext();
      }
    });
    const unsubscribeSongIntelligence = useSongIntelligenceStore.subscribe((state, previous) => {
      if (
        state.audioId !== previous.audioId ||
        state.profile !== previous.profile ||
        state.analysis !== previous.analysis
      ) {
        scheduleEntryContext();
      }
    });
    return () => {
      if (timeout !== null) window.clearTimeout(timeout);
      unsubscribeAudio();
      unsubscribeSongIntelligence();
    };
  }, []);

  const handleMessage = useCallback((event: MessageEvent<string>) => {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(event.data) as Record<string, unknown>;
    } catch {
      return;
    }

    if (message.type === "result" || message.type === "error") {
      const id = typeof message.id === "string" ? message.id : "";
      const pending = pendingRef.current.get(id);
      if (!pending) return;
      window.clearTimeout(pending.timeout);
      pendingRef.current.delete(id);
      if (message.type === "error") {
        const error = message.error as { message?: string } | undefined;
        pending.reject(bridgeError(error?.message ?? "Bridge request failed"));
      } else {
        pending.resolve(message.payload);
      }
      return;
    }

    if (typeof message.id !== "string" || typeof message.type !== "string") return;
    const requestMessage = message as unknown as AgentBridgeRequest;
    void Promise.resolve()
      .then(() => handleRequest(requestMessage))
      .then((payload) => sendResult(requestMessage.id, payload))
      .catch((error) => sendError(requestMessage.id, error));
  }, [handleRequest, sendError, sendResult]);

  useEffect(() => {
    if (!IS_DESKTOP_MODE) return;
    let closed = false;
    const pending = pendingRef.current;

    const connect = () => {
      if (closed) return;
      setStatus("connecting");
      const ws = new WebSocket(AGENT_BRIDGE_WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        if (closed || wsRef.current !== ws) return;
        setStatus("connected");
      };
      ws.onmessage = (event) => {
        if (closed || wsRef.current !== ws) return;
        handleMessage(event);
      };
      ws.onerror = () => {
        if (closed || wsRef.current !== ws) return;
        setStatus("error");
      };
      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        if (wsRef.current === ws) wsRef.current = null;
        if (closed) return;
        setStatus("error");
        reconnectRef.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };
    };

    connect();
    return () => {
      closed = true;
      if (reconnectRef.current) window.clearTimeout(reconnectRef.current);
      for (const request of pending.values()) {
        window.clearTimeout(request.timeout);
        request.reject(bridgeError("Agent bridge disconnected"));
      }
      pending.clear();
      const activeSocket = wsRef.current;
      if (activeSocket) {
        activeSocket.onopen = null;
        activeSocket.onmessage = null;
        activeSocket.onerror = null;
        activeSocket.onclose = null;
        activeSocket.close(1000, "Frontend bridge closed");
      }
      if (wsRef.current === activeSocket) wsRef.current = null;
      setStatus("idle");
    };
  }, [handleMessage]);

  const submitDirective = useCallback(async (directive: string) => {
    const trimmed = directive.trim();
    if (!trimmed) throw bridgeError("Directive cannot be empty");
    const agent = useAgentStore.getState();
    if (!agent.armed) throw bridgeError("Agent control is disarmed");
    agent.setPhase("thinking", { transcript: trimmed, summary: "Directive sent to Hermes" });
    try {
      const result = await request("agent.submit_directive", {
        directive: trimmed,
        mode: agent.mode,
      });
      agent.setPhase("watching", { transcript: trimmed, summary: "Hermes directive finished" });
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      agent.setPhase("error", {
        transcript: trimmed,
        summary: "Hermes directive failed",
        error: message,
      });
      agent.setError(message);
      throw error;
    }
  }, [request]);

  return { status, submitDirective };
}
