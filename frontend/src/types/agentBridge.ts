import type {
  IntensityCurve,
  IntensitySource,
  LinkTarget,
  Rank,
  SpatialMode,
  UpdateBlockConfigMessage,
} from "./sae";
import type {
  DestinationMode,
  DestinationSlot,
  DestinationSpace,
  DestinationType,
  ReactiveConfig,
  ReplaceMode,
} from "./destinations";
import type { AgentIntentIR } from "./agent";

export type AgentBridgeStatus = "idle" | "connecting" | "connected" | "error";

export interface AgentBridgeRequest {
  id: string;
  type: string;
  payload?: unknown;
}

export interface AgentBridgeResult {
  id: string;
  type: "result";
  payload?: unknown;
}

export interface AgentBridgeError {
  id?: string;
  type: "error";
  error: {
    code?: string;
    message: string;
  };
}

export type AgentBridgeMessage = AgentBridgeRequest | AgentBridgeResult | AgentBridgeError;

export interface AgentUpdateBlockConfigAction extends UpdateBlockConfigMessage {
  action: "update_block_config";
  // Slot name from the capability manifest (Hermes dialect keeps the field name `block`)
  block: string;
  link_target?: LinkTarget;
  feature_label?: string;
  feature_id?: number;
  enabled?: boolean;
  auto_config?: boolean;
  sae_rank?: Rank;
  spatial_mode?: SpatialMode;
  spatial_mask?: number[];
  intensity_source?: IntensitySource;
  intensity_curve?: IntensityCurve;
  intensity_gamma?: number;
}

export interface AgentSetDestinationAction {
  action: "set_destination";
  space: DestinationSpace;
  slot: DestinationSlot;
  destination_type: DestinationType;
  seed?: number;
  prompt?: string;
  replace_mode?: ReplaceMode;
}

export interface AgentClearDestinationAction {
  action: "clear_destination";
  space: DestinationSpace;
  slot: DestinationSlot;
}

export interface AgentFreezeBlendAction {
  action: "freeze_blend";
  space: DestinationSpace;
  target_slot: DestinationSlot;
}

export interface AgentSetDestinationModeAction {
  action: "set_destination_mode";
  space: DestinationSpace;
  mode: DestinationMode;
}

export interface AgentSetDestinationLinkAction {
  action: "set_destination_link";
  space: DestinationSpace;
  link_target: LinkTarget;
}

export interface AgentSetReactiveConfigAction {
  action: "set_reactive_config";
  space: DestinationSpace;
  stage_left?: number;
  stage_home?: number;
  stage_right?: number;
  position_source?: ReactiveConfig["positionSource"];
  intensity_source?: ReactiveConfig["intensitySource"];
  position_smoothing_ms?: number;
  silence_behavior?: ReactiveConfig["silenceBehavior"];
  drift_ms?: number;
  intensity_curve?: ReactiveConfig["intensityCurve"];
  intensity_gamma?: number;
  stem_rankings?: ReactiveConfig["stemRankings"];
  rank_weights?: ReactiveConfig["rankWeights"];
  blend_slew_rate?: number;
}

export interface AgentSetBlendPositionAction {
  action: "set_blend_position";
  space: DestinationSpace;
  position: number;
}

export interface AgentSetCompositionConfigAction {
  action: "set_composition_config";
  distance?: number;
  mode?: "auto" | "pulse" | "continuous";
}

export type AgentVisualAction =
  | AgentUpdateBlockConfigAction
  | AgentSetDestinationAction
  | AgentClearDestinationAction
  | AgentFreezeBlendAction
  | AgentSetDestinationModeAction
  | AgentSetDestinationLinkAction
  | AgentSetReactiveConfigAction
  | AgentSetBlendPositionAction
  | AgentSetCompositionConfigAction;

export interface AgentVisualPlanTiming {
  based_on_audio_time?: number | null;
  based_on_wall_time_ms?: number | null;
  max_staleness_sec?: number | null;
}

export interface AgentVisualPlan extends AgentVisualPlanTiming {
  actions: AgentVisualAction[];
  transcript?: string | null;
  provider?: string | null;
  model?: string | null;
  reason?: string | null;
  feature_candidates?: Record<string, unknown>[];
  intent?: AgentIntentIR | null;
}

export interface AgentBridgeController {
  status: AgentBridgeStatus;
  submitDirective: (directive: string) => Promise<unknown>;
}
