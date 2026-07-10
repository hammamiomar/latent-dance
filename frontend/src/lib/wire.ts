/**
 * wire — typed send functions for every client→backend control message.
 *
 * This is the single outbound vocabulary of the streaming protocol
 * (mirrors app/schemas.py client messages). Stores, control functions, and
 * components import these directly instead of receiving send callbacks
 * through props. All of them funnel through transport.send(), which drops
 * silently when the socket is not open.
 */

import { send } from "./transport";
import { useConnectionStore } from "../stores/useConnectionStore";
import { useSongIntelligenceStore } from "../stores/useSongIntelligenceStore";
import type { SteeringMode } from "../stores/useSessionStore";
import type {
  LinkTarget,
  UpdateBlockConfigMessage,
  UpdateSlotConfigMessage,
} from "../types/sae";
import type {
  DestinationMode,
  DestinationSlot,
  DestinationSpace,
  DestinationType,
  ReactiveConfig,
} from "../types/destinations";

// === Generation lifecycle ===

export function sendStartSAESteering(audioId: string): void {
  useSongIntelligenceStore.getState().clear();
  send({ action: "start_sae_steering", audio_id: audioId });
  useConnectionStore.getState().setGenerating(true);
}

export function sendStopGeneration(): void {
  send({ action: "stop_generation" });
  useConnectionStore.getState().setGenerating(false);
}

// === Slot steering ===

export function sendUpdateSlotConfig(message: UpdateSlotConfigMessage): void {
  send(message as unknown as Record<string, unknown>);
}

/** Frozen Hermes agent dialect — forwarded verbatim by agentPlanApply only. */
export function sendUpdateBlockConfig(message: UpdateBlockConfigMessage): void {
  send(message as unknown as Record<string, unknown>);
}

export function sendSetSteeringMode(mode: SteeringMode): void {
  send({ action: "set_steering_mode", mode });
}

// === Audio sync ===

export function sendAudioTimeUpdate(time: number): void {
  send({ action: "audio_timeupdate", time });
}

export function sendAudioPlay(time: number): void {
  send({ action: "audio_play", time });
}

export function sendAudioPause(): void {
  send({ action: "audio_pause" });
}

export function sendAudioSeek(time: number): void {
  send({ action: "audio_seek", time });
}

// === Destination modulation ===

export function sendSetDestination(
  space: DestinationSpace,
  slot: DestinationSlot,
  destinationType: DestinationType,
  value: { seed?: number; prompt?: string },
  replaceMode: "direct" | "from_blend" = "direct",
): void {
  send({
    action: "set_destination",
    space,
    slot,
    destination_type: destinationType,
    seed: value.seed,
    prompt: value.prompt,
    replace_mode: replaceMode,
  });
}

export function sendClearDestination(space: DestinationSpace, slot: DestinationSlot): void {
  send({ action: "clear_destination", space, slot });
}

export function sendFreezeBlend(space: DestinationSpace, targetSlot: DestinationSlot): void {
  send({ action: "freeze_blend", space, target_slot: targetSlot });
}

export function sendSetBlendPosition(space: DestinationSpace, position: number): void {
  send({ action: "set_blend_position", space, position });
}

export function sendSetDestinationMode(space: DestinationSpace, mode: DestinationMode): void {
  if (mode === "linked") {
    console.warn("Use sendSetDestinationLink() for linked mode");
    return;
  }
  send({ action: "set_destination_mode", space, mode });
}

export function sendSetReactiveConfig(
  space: DestinationSpace,
  config: Partial<ReactiveConfig>,
): void {
  send({
    action: "set_reactive_config",
    space,
    stage_left: config.stageLeft,
    stage_home: config.stageHome,
    stage_right: config.stageRight,
    position_source: config.positionSource,
    intensity_source: config.intensitySource,
    position_smoothing_ms: config.positionSmoothingMs,
    silence_behavior: config.silenceBehavior,
    drift_ms: config.driftMs,
    intensity_curve: config.intensityCurve,
    intensity_gamma: config.intensityGamma,
    stem_rankings: config.stemRankings,
    rank_weights: config.rankWeights,
    blend_slew_rate: config.blendSlewRate,
  });
}

export function sendSetDestinationLink(space: DestinationSpace, linkTarget: LinkTarget): void {
  send({ action: "set_destination_link", space, link_target: linkTarget });
}

// === Composition engine ===

export function sendSetCompositionConfig(config: { distance?: number; mode?: string }): void {
  send({ action: "set_composition_config", ...config });
}
