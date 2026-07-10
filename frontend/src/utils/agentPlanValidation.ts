import type { AgentMode } from "../types/agent";
import type { AgentVisualPlan } from "../types/agentBridge";
import type {
  DestinationMode,
  DestinationSlot,
  DestinationSpace,
  DestinationType,
  ReplaceMode,
} from "../types/destinations";
import type {
  IntensityCurve,
  IntensitySource,
  PositionSource,
  Rank,
  SilenceBehavior,
  SpatialMode,
} from "../types/sae";
import { ALL_LINK_TARGETS, VALID_RANKS } from "../types/sae";

const DESTINATION_SPACES = ["latent", "prompt"] as const satisfies readonly DestinationSpace[];
const DESTINATION_SLOTS = ["a", "b"] as const satisfies readonly DestinationSlot[];
const DESTINATION_TYPES = ["seed", "prompt"] as const satisfies readonly DestinationType[];
const DESTINATION_MODES = ["slider", "reactive", "linked"] as const satisfies readonly DestinationMode[];
const REPLACE_MODES = ["direct", "from_blend"] as const satisfies readonly ReplaceMode[];
const SPATIAL_MODES = ["draw", "pitch_aligned"] as const satisfies readonly SpatialMode[];
const POSITION_SOURCES = ["auto", "pitch", "brightness", "chroma", "tension", "tension_global"] as const satisfies readonly PositionSource[];
const INTENSITY_SOURCES = ["energy_smooth", "transient", "flux", "envelope"] as const satisfies readonly IntensitySource[];
const SILENCE_BEHAVIORS = ["drift_center", "hold_last"] as const satisfies readonly SilenceBehavior[];
const INTENSITY_CURVES = ["linear", "gamma", "clip"] as const satisfies readonly IntensityCurve[];
const COMPOSITION_MODES = ["auto", "pulse", "continuous"] as const;
const ACTION_TYPES = [
  "update_block_config",
  "set_destination",
  "clear_destination",
  "freeze_blend",
  "set_destination_mode",
  "set_destination_link",
  "set_reactive_config",
  "set_blend_position",
  "set_composition_config",
] as const;

// Hermes drives the SAE backend only; these numeric bounds are part of the
// frozen agent dialect. Slot names, by contrast, come from the capability
// manifest via context.slotNames.
const FEATURE_ID_MIN = 0;
const FEATURE_ID_MAX = 5119;
const STAGE_MIN = -50;
const STAGE_MAX = 50;
const SPATIAL_MASK_SIZE = 256;

export type AgentPlanValidationCode =
  | "invalid_plan"
  | "invalid_action"
  | "invalid_block"
  | "invalid_link_target"
  | "invalid_feature_id"
  | "invented_feature_id"
  | "invalid_spatial_mask"
  | "invalid_stage_bounds"
  | "invalid_value"
  | "agent_disarmed"
  | "inactive_session"
  | "bridge_disconnected"
  | "stale_plan"
  | "missing_current_audio_time";

export interface AgentPlanRejection {
  code: AgentPlanValidationCode;
  path: string;
  message: string;
  value?: unknown;
  requires_refresh?: boolean;
}

export type AgentPlanTimingProfile =
  | "directive"
  | "dj_calm"
  | "dj_balanced"
  | "dj_active"
  | "section_timed";

export interface AgentPlanValidationContext {
  armed: boolean;
  activeSession: boolean;
  /** Slot names from the capability manifest — the only valid `block` values. */
  slotNames: readonly string[];
  bridgeConnected?: boolean;
  currentAudioTime?: number;
  mode?: AgentMode;
  djIntensity?: "calm" | "balanced" | "active";
  timingProfile?: AgentPlanTimingProfile;
  knownFeatureIdsByBlock?: Partial<Record<string, readonly number[] | ReadonlySet<number>>>;
  currentFeatureIdsByBlock?: Partial<Record<string, number | null | undefined>>;
}

export type AgentPlanValidationResult =
  | { ok: true; plan: AgentVisualPlan }
  | { ok: false; errors: AgentPlanRejection[] };

export function defaultMaxStalenessSec(context: AgentPlanValidationContext): number {
  const profile = context.timingProfile ?? (
    context.mode === "dj"
      ? `dj_${context.djIntensity ?? "balanced"}` as AgentPlanTimingProfile
      : "directive"
  );

  switch (profile) {
    case "dj_calm":
      return 12;
    case "dj_balanced":
      return 8;
    case "dj_active":
      return 4;
    case "section_timed":
      return 2;
    case "directive":
    default:
      return 20;
  }
}

export function validateAgentVisualPlan(
  value: unknown,
  context: AgentPlanValidationContext,
): AgentPlanValidationResult {
  const errors: AgentPlanRejection[] = [];

  if (!isRecord(value)) {
    return {
      ok: false,
      errors: [error("invalid_plan", "", "Visual plan must be an object", value)],
    };
  }

  if (!context.armed) {
    errors.push(error("agent_disarmed", "context.armed", "Agent control is disarmed", context.armed));
  }
  if (context.bridgeConnected === false) {
    errors.push(error("bridge_disconnected", "context.bridgeConnected", "Agent bridge is not connected", context.bridgeConnected));
  }

  validateTiming(value, context, errors);

  const actions = value.actions;
  if (!Array.isArray(actions) || actions.length === 0) {
    errors.push(error("invalid_plan", "actions", "Visual plan needs at least one action", actions));
  } else {
    const evidence = collectFeatureEvidence(value, context);
    actions.forEach((action, index) => {
      validateAction(action, `actions[${index}]`, evidence, context, errors);
    });
  }

  return errors.length === 0
    ? { ok: true, plan: value as unknown as AgentVisualPlan }
    : { ok: false, errors };
}

function validateTiming(
  _plan: Record<string, unknown>,
  _context: AgentPlanValidationContext,
  _errors: AgentPlanRejection[],
) {
  // Hermes plans are durable rig mutations. Timing metadata is accepted only as
  // legacy agent noise and must never block a visual edit after rewind/looping.
}

function validateAction(
  value: unknown,
  path: string,
  evidence: ReadonlyMap<string, ReadonlySet<number>>,
  context: AgentPlanValidationContext,
  errors: AgentPlanRejection[],
) {
  if (!isRecord(value)) {
    errors.push(error("invalid_action", path, "Action must be an object", value));
    return;
  }
  if (!isOneOf(value.action, ACTION_TYPES)) {
    errors.push(error("invalid_action", `${path}.action`, "Unknown action type", value.action));
    return;
  }

  switch (value.action) {
    case "update_block_config":
      validateUpdateBlockConfig(value, path, evidence, context, errors);
      break;
    case "set_destination":
      validateSetDestination(value, path, errors);
      break;
    case "clear_destination":
      validateSpace(value.space, `${path}.space`, errors);
      validateSlot(value.slot, `${path}.slot`, errors);
      break;
    case "freeze_blend":
      if (!context.activeSession) {
        errors.push(error(
          "inactive_session",
          "context.activeSession",
          "freeze_blend requires an active visualizer session",
          context.activeSession,
        ));
      }
      validatePromptSpace(value.space, `${path}.space`, "freeze_blend", errors);
      validateSlot(value.target_slot, `${path}.target_slot`, errors);
      break;
    case "set_destination_mode":
      validatePromptSpace(value.space, `${path}.space`, "set_destination_mode", errors);
      if (!isOneOf(value.mode, DESTINATION_MODES)) {
        errors.push(error("invalid_value", `${path}.mode`, "Invalid destination mode", value.mode));
      } else if (value.mode === "linked") {
        errors.push(error("invalid_value", `${path}.mode`, "Use set_destination_link for linked mode", value.mode));
      }
      break;
    case "set_destination_link":
      validatePromptSpace(value.space, `${path}.space`, "set_destination_link", errors);
      validateLinkTarget(value.link_target, `${path}.link_target`, errors);
      break;
    case "set_reactive_config":
      validateSetReactiveConfig(value, path, errors);
      break;
    case "set_blend_position":
      validatePromptSpace(value.space, `${path}.space`, "set_blend_position", errors);
      validateFiniteRange(value.position, `${path}.position`, 0, 1, errors);
      break;
    case "set_composition_config":
      validateOptionalFinite(value.distance, `${path}.distance`, 0, 4, errors);
      if (value.mode != null && !isOneOf(value.mode, COMPOSITION_MODES)) {
        errors.push(error("invalid_value", `${path}.mode`, "Invalid composition mode", value.mode));
      }
      break;
  }
}

function validateUpdateBlockConfig(
  action: Record<string, unknown>,
  path: string,
  evidence: ReadonlyMap<string, ReadonlySet<number>>,
  context: AgentPlanValidationContext,
  errors: AgentPlanRejection[],
) {
  if (typeof action.block !== "string" || !context.slotNames.includes(action.block)) {
    errors.push(error("invalid_block", `${path}.block`, "Unknown SAE block", action.block));
    return;
  }
  const block = action.block;
  if (action.link_target != null) validateLinkTarget(action.link_target, `${path}.link_target`, errors);
  if (action.spatial_mode != null && !isOneOf(action.spatial_mode, SPATIAL_MODES)) {
    errors.push(error("invalid_value", `${path}.spatial_mode`, "Invalid spatial mode", action.spatial_mode));
  }
  if (action.intensity_source != null && !isOneOf(action.intensity_source, INTENSITY_SOURCES)) {
    errors.push(error("invalid_value", `${path}.intensity_source`, "Invalid intensity source", action.intensity_source));
  }
  if (action.intensity_curve != null && !isOneOf(action.intensity_curve, INTENSITY_CURVES)) {
    errors.push(error("invalid_value", `${path}.intensity_curve`, "Invalid intensity curve", action.intensity_curve));
  }
  if ("sae_rank" in action && !isRank(action.sae_rank)) {
    errors.push(error("invalid_value", `${path}.sae_rank`, "Invalid SAE rank", action.sae_rank));
  }

  validateFeatureId(action.feature_id, block, `${path}.feature_id`, evidence, errors);
  validateSpatialMask(action.spatial_mask, `${path}.spatial_mask`, errors);
  validateOptionalFinite(action.intensity_gamma, `${path}.intensity_gamma`, Number.MIN_VALUE, Number.POSITIVE_INFINITY, errors);
  validateStageBounds(action, path, errors);
}

function validateSetDestination(
  action: Record<string, unknown>,
  path: string,
  errors: AgentPlanRejection[],
) {
  validateSpace(action.space, `${path}.space`, errors);
  validateSlot(action.slot, `${path}.slot`, errors);
  if (!isOneOf(action.destination_type, DESTINATION_TYPES)) {
    errors.push(error("invalid_value", `${path}.destination_type`, "Invalid destination type", action.destination_type));
    return;
  }
  if (action.replace_mode != null && !isOneOf(action.replace_mode, REPLACE_MODES)) {
    errors.push(error("invalid_value", `${path}.replace_mode`, "Invalid replace mode", action.replace_mode));
  }
  if (action.space === "latent" && action.destination_type !== "seed") {
    errors.push(error("invalid_value", `${path}.destination_type`, "Latent destinations require seed type", action.destination_type));
  }
  if (action.space === "prompt" && action.destination_type !== "prompt") {
    errors.push(error("invalid_value", `${path}.destination_type`, "Prompt destinations require prompt type", action.destination_type));
  }
  if (action.destination_type === "seed") {
    if (!Number.isInteger(action.seed)) {
      errors.push(error("invalid_value", `${path}.seed`, "Seed destination requires an integer seed", action.seed));
    }
    if (action.prompt != null) {
      errors.push(error("invalid_value", `${path}.prompt`, "Seed destination must not include prompt", action.prompt));
    }
  }
  if (action.destination_type === "prompt") {
    if (typeof action.prompt !== "string" || action.prompt.trim().length === 0) {
      errors.push(error("invalid_value", `${path}.prompt`, "Prompt destination requires prompt text", action.prompt));
    }
    if (action.seed != null) {
      errors.push(error("invalid_value", `${path}.seed`, "Prompt destination must not include seed", action.seed));
    }
  }
}

function validateSetReactiveConfig(
  action: Record<string, unknown>,
  path: string,
  errors: AgentPlanRejection[],
) {
  validatePromptSpace(action.space, `${path}.space`, "set_reactive_config", errors);
  validateStageBounds(action, path, errors);
  if (action.position_source != null && !isOneOf(action.position_source, POSITION_SOURCES)) {
    errors.push(error("invalid_value", `${path}.position_source`, "Invalid position source", action.position_source));
  }
  if (action.intensity_source != null && !isOneOf(action.intensity_source, INTENSITY_SOURCES)) {
    errors.push(error("invalid_value", `${path}.intensity_source`, "Invalid intensity source", action.intensity_source));
  }
  if (action.silence_behavior != null && !isOneOf(action.silence_behavior, SILENCE_BEHAVIORS)) {
    errors.push(error("invalid_value", `${path}.silence_behavior`, "Invalid silence behavior", action.silence_behavior));
  }
  if (action.intensity_curve != null && !isOneOf(action.intensity_curve, INTENSITY_CURVES)) {
    errors.push(error("invalid_value", `${path}.intensity_curve`, "Invalid intensity curve", action.intensity_curve));
  }
  validateOptionalFinite(action.position_smoothing_ms, `${path}.position_smoothing_ms`, 0, Number.POSITIVE_INFINITY, errors);
  validateOptionalFinite(action.drift_ms, `${path}.drift_ms`, 0, Number.POSITIVE_INFINITY, errors);
  validateOptionalFinite(action.intensity_gamma, `${path}.intensity_gamma`, Number.MIN_VALUE, Number.POSITIVE_INFINITY, errors);
  validateOptionalFinite(action.blend_slew_rate, `${path}.blend_slew_rate`, 0, Number.POSITIVE_INFINITY, errors);
}

function validateStageBounds(action: Record<string, unknown>, path: string, errors: AgentPlanRejection[]) {
  const left = coalesceNumber(action.stage_left, action.strength_min);
  const home = asOptionalNumber(action.stage_home);
  const right = coalesceNumber(action.stage_right, action.strength_max);

  validateOptionalFinite(action.stage_left, `${path}.stage_left`, STAGE_MIN, STAGE_MAX, errors);
  validateOptionalFinite(action.stage_home, `${path}.stage_home`, STAGE_MIN, STAGE_MAX, errors);
  validateOptionalFinite(action.stage_right, `${path}.stage_right`, STAGE_MIN, STAGE_MAX, errors);
  validateOptionalFinite(action.strength_min, `${path}.strength_min`, STAGE_MIN, STAGE_MAX, errors);
  validateOptionalFinite(action.strength_max, `${path}.strength_max`, STAGE_MIN, STAGE_MAX, errors);

  if (isFiniteNumber(left) && isFiniteNumber(home) && left > home) {
    errors.push(error("invalid_stage_bounds", path, "stage_left must be <= stage_home", { left, home }));
  }
  if (isFiniteNumber(home) && isFiniteNumber(right) && home > right) {
    errors.push(error("invalid_stage_bounds", path, "stage_home must be <= stage_right", { home, right }));
  }
  if (isFiniteNumber(left) && isFiniteNumber(right) && left > right) {
    errors.push(error("invalid_stage_bounds", path, "stage_left must be <= stage_right", { left, right }));
  }
}

function validateFeatureId(
  value: unknown,
  block: string,
  path: string,
  evidence: ReadonlyMap<string, ReadonlySet<number>>,
  errors: AgentPlanRejection[],
) {
  if (value == null) return;
  if (typeof value !== "number" || !Number.isInteger(value) || value < FEATURE_ID_MIN || value > FEATURE_ID_MAX) {
    errors.push(error("invalid_feature_id", path, "Feature id must be an integer in [0, 5119]", value));
    return;
  }
  if (!evidence.get(block)?.has(value)) {
    errors.push(error("invented_feature_id", path, "Feature id must come from feature search/browse results or current config", value));
  }
}

function validateSpatialMask(value: unknown, path: string, errors: AgentPlanRejection[]) {
  if (value == null) return;
  if (!Array.isArray(value) || value.length !== SPATIAL_MASK_SIZE) {
    errors.push(error("invalid_spatial_mask", path, "spatial_mask must contain 256 values", value));
    return;
  }
  const badIndex = value.findIndex((item) => !isFiniteNumber(item) || item < 0 || item > 1);
  if (badIndex >= 0) {
    errors.push(error("invalid_spatial_mask", `${path}[${badIndex}]`, "spatial_mask values must be finite numbers in [0, 1]", value[badIndex]));
  }
}

function validateLinkTarget(value: unknown, path: string, errors: AgentPlanRejection[]) {
  if (!isOneOf(value, ALL_LINK_TARGETS)) {
    errors.push(error("invalid_link_target", path, "Unknown link target", value));
  }
}

function validateSpace(value: unknown, path: string, errors: AgentPlanRejection[]) {
  if (!isOneOf(value, DESTINATION_SPACES)) {
    errors.push(error("invalid_value", path, "Invalid destination space", value));
  }
}

function validatePromptSpace(
  value: unknown,
  path: string,
  action: string,
  errors: AgentPlanRejection[],
) {
  validateSpace(value, path, errors);
  if (value === "latent") {
    errors.push(error(
      "invalid_value",
      path,
      `${action} only applies to prompt space; use latent seeds and set_composition_config for composition`,
      value,
    ));
  }
}

function validateSlot(value: unknown, path: string, errors: AgentPlanRejection[]) {
  if (!isOneOf(value, DESTINATION_SLOTS)) {
    errors.push(error("invalid_value", path, "Invalid destination slot", value));
  }
}

function validateFiniteRange(
  value: unknown,
  path: string,
  min: number,
  max: number,
  errors: AgentPlanRejection[],
) {
  if (!isFiniteNumber(value) || value < min || value > max) {
    errors.push(error("invalid_value", path, `Value must be a finite number in [${min}, ${max}]`, value));
  }
}

function validateOptionalFinite(
  value: unknown,
  path: string,
  min: number,
  max: number,
  errors: AgentPlanRejection[],
) {
  if (value == null) return;
  validateFiniteRange(value, path, min, max, errors);
}

function collectFeatureEvidence(
  plan: Record<string, unknown>,
  context: AgentPlanValidationContext,
): ReadonlyMap<string, ReadonlySet<number>> {
  const evidence = new Map<string, Set<number>>();
  const isKnownSlot = (block: unknown): block is string =>
    typeof block === "string" && context.slotNames.includes(block);
  const add = (block: string, featureId: number) => {
    if (!evidence.has(block)) evidence.set(block, new Set());
    evidence.get(block)?.add(featureId);
  };

  for (const [block, ids] of Object.entries(context.knownFeatureIdsByBlock ?? {})) {
    if (!isKnownSlot(block) || ids == null) continue;
    ids.forEach((id) => add(block, id));
  }

  for (const [block, featureId] of Object.entries(context.currentFeatureIdsByBlock ?? {})) {
    if (isKnownSlot(block) && typeof featureId === "number" && Number.isInteger(featureId)) {
      add(block, featureId);
    }
  }

  const candidates = Array.isArray(plan.feature_candidates) ? plan.feature_candidates : [];
  for (const candidate of candidates) {
    if (!isRecord(candidate)) continue;
    const block = candidate.block;
    const id = candidate.id ?? candidate.feature_id;
    if (isKnownSlot(block) && typeof id === "number" && Number.isInteger(id)) {
      add(block, id);
    }
  }

  return evidence;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isOneOf<const T extends readonly unknown[]>(
  value: unknown,
  allowed: T,
): value is T[number] {
  return (allowed as readonly unknown[]).includes(value);
}

function isRank(value: unknown): value is Rank {
  return isOneOf(value, VALID_RANKS);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function asOptionalNumber(value: unknown): number | undefined {
  return isFiniteNumber(value) ? value : undefined;
}

function coalesceNumber(first: unknown, second: unknown): number | undefined {
  return asOptionalNumber(first) ?? asOptionalNumber(second);
}

function error(
  code: AgentPlanValidationCode,
  path: string,
  message: string,
  value?: unknown,
): AgentPlanRejection {
  return { code, path, message, value };
}
