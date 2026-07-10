import { getFeature } from "../data/featureLoader";
import { useSlotStore } from "../stores/useSlotStore";
import { useCompositionStore } from "../stores/useCompositionStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import {
  sendClearDestination,
  sendFreezeBlend,
  sendSetBlendPosition,
  sendSetCompositionConfig,
  sendSetDestination,
  sendSetDestinationLink,
  sendSetDestinationMode,
  sendSetReactiveConfig,
  sendUpdateBlockConfig,
} from "../lib/wire";
import type {
  AgentSetDestinationAction,
  AgentSetReactiveConfigAction,
  AgentUpdateBlockConfigAction,
  AgentVisualAction,
} from "../types/agentBridge";
import type { Destination, ReactiveConfig } from "../types/destinations";
import type { StrengthRange } from "../types/sae";

export interface AgentPlanChangeSummary extends Record<string, unknown> {
  action: string;
  target: string;
  after: Record<string, unknown>;
}

function agentApplyError(message: string) {
  return new Error(message);
}

function destinationLabel(destinationType: "seed" | "prompt", value: { seed?: number; prompt?: string }) {
  if (destinationType === "seed") return `Seed ${value.seed ?? 0}`;
  const prompt = value.prompt ?? "";
  return prompt.length > 20 ? `${prompt.slice(0, 20)}...` : prompt;
}

function reactiveConfigFromAction(action: AgentSetReactiveConfigAction): Partial<ReactiveConfig> {
  return {
    stageLeft: action.stage_left,
    stageHome: action.stage_home,
    stageRight: action.stage_right,
    positionSource: action.position_source,
    intensitySource: action.intensity_source,
    positionSmoothingMs: action.position_smoothing_ms,
    silenceBehavior: action.silence_behavior,
    driftMs: action.drift_ms,
    intensityCurve: action.intensity_curve,
    intensityGamma: action.intensity_gamma,
    stemRankings: action.stem_rankings,
    rankWeights: action.rank_weights,
    blendSlewRate: action.blend_slew_rate,
  };
}

function summary(action: AgentVisualAction, target: string): AgentPlanChangeSummary {
  return {
    action: action.action,
    target,
    after: action as unknown as Record<string, unknown>,
  };
}

function applyBlockConfig(action: AgentUpdateBlockConfigAction): AgentPlanChangeSummary {
  const store = useSlotStore.getState();
  const current = store.slots[action.block];
  if (!current) throw agentApplyError(`Unknown block: ${action.block}`);

  if (action.feature_id != null) {
    // Frozen Hermes dialect bound: the agent drives the SAE backend only.
    if (action.feature_id < 0 || action.feature_id > 5119) {
      throw agentApplyError(`Feature id out of range: ${action.feature_id}`);
    }
    const label = action.feature_label
      ?? getFeature(action.block, action.feature_id)?.label
      ?? `#${action.feature_id}`;
    store.setSlotFeature(action.block, action.feature_id, label);
  }
  if (action.link_target) store.setSlotLinkTarget(action.block, action.link_target);
  if (action.enabled != null) store.setSlotEnabled(action.block, action.enabled);
  if (action.auto_config != null) store.setSlotAutoConfig(action.block, action.auto_config);
  if ("sae_rank" in action) store.setSlotSaeRank(action.block, action.sae_rank ?? null);
  if (action.spatial_mode) store.setSlotSpatialMode(action.block, action.spatial_mode);
  if (action.spatial_mask) {
    if (action.spatial_mask.length !== 256) throw agentApplyError("spatial_mask must contain 256 values");
    store.setSlotSpatialMask(action.block, action.spatial_mask);
  }
  if (action.intensity_source) store.setSlotIntensitySource(action.block, action.intensity_source);
  if (action.intensity_curve) store.setSlotIntensityCurve(action.block, action.intensity_curve);
  if (action.intensity_gamma != null) store.setSlotIntensityGamma(action.block, action.intensity_gamma);

  if (
    action.strength_min != null ||
    action.strength_max != null ||
    action.stage_left != null ||
    action.stage_right != null ||
    action.stage_home != null
  ) {
    const range: StrengthRange = {
      strengthMin: action.stage_left ?? action.strength_min ?? current.strengthRange.strengthMin,
      strengthMax: action.stage_right ?? action.strength_max ?? current.strengthRange.strengthMax,
      stageHome: action.stage_home ?? current.strengthRange.stageHome,
    };
    store.setSlotStrengthRange(action.block, range);
  }

  sendUpdateBlockConfig(action);
  return summary(action, action.block);
}

function applyDestination(action: AgentSetDestinationAction): AgentPlanChangeSummary {
  const value = { seed: action.seed, prompt: action.prompt };
  const destination: Destination = {
    type: action.destination_type,
    label: destinationLabel(action.destination_type, value),
    ...value,
  };
  useDestinationStore.getState().setDestination(action.space, action.slot, destination);
  sendSetDestination(
    action.space,
    action.slot,
    action.destination_type,
    value,
    action.replace_mode ?? "direct",
  );
  return summary(action, `${action.space}:${action.slot}`);
}

export function applyAgentVisualAction(action: AgentVisualAction): AgentPlanChangeSummary {
  switch (action.action) {
    case "update_block_config":
      return applyBlockConfig(action);
    case "set_destination":
      return applyDestination(action);
    case "clear_destination":
      useDestinationStore.getState().clearDestination(action.space, action.slot);
      sendClearDestination(action.space, action.slot);
      return summary(action, `${action.space}:${action.slot}`);
    case "freeze_blend":
      sendFreezeBlend(action.space, action.target_slot);
      return summary(action, `${action.space}:${action.target_slot}`);
    case "set_destination_mode":
      if (action.mode === "linked") throw agentApplyError("Use set_destination_link for linked mode");
      useDestinationStore.getState().setMode(action.space, action.mode);
      sendSetDestinationMode(action.space, action.mode);
      return summary(action, action.space);
    case "set_destination_link":
      useDestinationStore.getState().setLinkTarget(action.space, action.link_target);
      sendSetDestinationLink(action.space, action.link_target);
      return summary(action, action.space);
    case "set_reactive_config": {
      const config = reactiveConfigFromAction(action);
      useDestinationStore.getState().setReactiveConfig(action.space, config);
      sendSetReactiveConfig(action.space, config);
      return summary(action, action.space);
    }
    case "set_blend_position":
      useDestinationStore.getState().setBlendPosition(action.space, action.position);
      sendSetBlendPosition(action.space, action.position);
      return summary(action, action.space);
    case "set_composition_config":
      useCompositionStore.getState().setConfig({
        distance: action.distance,
        mode: action.mode,
      });
      sendSetCompositionConfig({
        distance: action.distance,
        mode: action.mode,
      });
      return summary(action, "composition");
    default:
      throw agentApplyError(`Unsupported agent action: ${(action as { action?: string }).action}`);
  }
}
