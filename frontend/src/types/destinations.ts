/**
 * Destination Modulation Types
 *
 * Type definitions for SLERP-based destination travel in latent and prompt space.
 * Two destinations (A and B) with slider or reactive-mode blending.
 */

import type {
  LinkTarget,
  Rank,
  PositionSource,
  IntensitySource,
  SilenceBehavior,
  IntensityCurve,
} from './sae';

/** Which space the destination modulator operates on */
export type DestinationSpace = 'latent' | 'prompt';

/** Which destination slot */
export type DestinationSlot = 'a' | 'b';

/** How the destination was created */
export type DestinationType = 'seed' | 'prompt';

/** Modulation mode */
export type DestinationMode = 'slider' | 'reactive' | 'linked';

/** How a destination replacement is handled */
export type ReplaceMode = 'direct' | 'from_blend';

/** A destination point in latent or prompt space */
export interface Destination {
  type: DestinationType;
  label: string;
  seed?: number;
  prompt?: string;
}

/** Per-stem ranking for Dancer Ensemble reactive mode */
export interface StemRankings {
  drums: Rank;
  bass: Rank;
  vocals: Rank;
  other: Rank;
}

/** Default stem rankings (all auto) */
export const DEFAULT_STEM_RANKINGS: StemRankings = {
  drums: 1,
  bass: 2,
  vocals: null,
  other: null,
};

/** Configuration for reactive mode */
export interface ReactiveConfig {
  // Stage anchors
  stageLeft: number;
  stageHome: number;
  stageRight: number;
  positionSource: PositionSource;
  intensitySource: IntensitySource;
  positionSmoothingMs: number;
  silenceBehavior: SilenceBehavior;
  driftMs: number;
  intensityCurve: IntensityCurve;
  intensityGamma: number;
  // Direct link target (bypasses driver/stem)
  linkTarget?: LinkTarget;
  // Per-stem rankings for blend contribution
  stemRankings?: StemRankings;
  // Rank weights
  rankWeights?: Record<string, number>;
  // Blend transition speed (max blend change per second, 0 = instant)
  blendSlewRate?: number;
}

/** Default reactive config */
export const DEFAULT_REACTIVE_CONFIG: ReactiveConfig = {
  stageLeft: -30,
  stageHome: 0,
  stageRight: 30,
  positionSource: 'auto',
  intensitySource: 'energy_smooth',
  positionSmoothingMs: 50,
  silenceBehavior: 'hold_last',
  driftMs: 1500,
  intensityCurve: 'linear',
  intensityGamma: 1.0,
  stemRankings: DEFAULT_STEM_RANKINGS,
  rankWeights: {
    '1': 1.0,
    '2': 0.75,
    '3': 0.5,
    '4': 0.25,
    auto: 0.6,
  },
  blendSlewRate: 1.5,
};

/** State of a destination modulator */
export interface DestinationState {
  space: DestinationSpace;
  destinationA: Destination | null;
  destinationB: Destination | null;
  blendPosition: number;
  mode: DestinationMode;
  reactiveConfig: ReactiveConfig;
  /** Link target for 'linked' mode - audio signal that drives blend position */
  linkTarget: LinkTarget | null;
}

/** Default destination state */
export const DEFAULT_DESTINATION_STATE: DestinationState = {
  space: 'latent',
  destinationA: null,
  destinationB: null,
  blendPosition: 0.0,
  mode: 'slider',
  reactiveConfig: DEFAULT_REACTIVE_CONFIG,
  linkTarget: null,
};

// ============================================================================
// WebSocket Message Types (Client → Server)
// ============================================================================

export interface SetDestinationMessage {
  action: 'set_destination';
  space: DestinationSpace;
  slot: DestinationSlot;
  destination_type: DestinationType;
  seed?: number;
  prompt?: string;
  replace_mode: ReplaceMode;
}

export interface FreezeBlendMessage {
  action: 'freeze_blend';
  space: DestinationSpace;
  target_slot: DestinationSlot;
}

export interface SetBlendPositionMessage {
  action: 'set_blend_position';
  space: DestinationSpace;
  position: number;
}

export interface SetDestinationModeMessage {
  action: 'set_destination_mode';
  space: DestinationSpace;
  mode: DestinationMode;
}

export interface SetReactiveConfigMessage {
  action: 'set_reactive_config';
  space: DestinationSpace;
  stage_left?: number;
  stage_home?: number;
  stage_right?: number;
  position_source?: PositionSource;
  intensity_source?: IntensitySource;
  position_smoothing_ms?: number;
  silence_behavior?: SilenceBehavior;
  drift_ms?: number;
  intensity_curve?: IntensityCurve;
  intensity_gamma?: number;
  stem_rankings?: StemRankings;
  rank_weights?: Record<string, number>;
  blend_slew_rate?: number;
}

// ============================================================================
// WebSocket Message Types (Server → Client)
// ============================================================================

export interface DestinationStatusMessage {
  type: 'destination_status';
  space: DestinationSpace;
  destination_a: string | null;
  destination_b: string | null;
  blend_position: number;
  mode: DestinationMode;
}

// ============================================================================
// Union Types
// ============================================================================

/** Set destination link target directly */
export interface SetDestinationLinkMessage {
  action: 'set_destination_link';
  space: DestinationSpace;
  link_target: LinkTarget;
}

export type DestinationClientMessage =
  | SetDestinationMessage
  | FreezeBlendMessage
  | SetBlendPositionMessage
  | SetDestinationModeMessage
  | SetReactiveConfigMessage
  | SetDestinationLinkMessage;
