/**
 * SAE Steering Types
 *
 * Type definitions for SAE feature steering and block-to-stem mapping,
 * including spatial, channel, layer, and physics controls.
 */

/** Audio stem identifiers (4 physical + 5 virtual) */
export type Stem =
  | 'bass'
  | 'drums'
  | 'vocals'
  | 'other'
  | 'drums_low'
  | 'drums_mid'
  | 'drums_high'
  | 'other_mid'
  | 'other_high';

/** Physical stems from Demucs separation */
export type PhysicalStem = 'bass' | 'drums' | 'vocals' | 'other';

/** Virtual stems from bandpass filtering */
export type VirtualStem = 'drums_low' | 'drums_mid' | 'drums_high' | 'other_mid' | 'other_high';

// =============================================================================
// Link Targets
// =============================================================================

/** All possible link targets for audio-reactive system */
export type LinkTarget =
  // Physical stems (Demucs output)
  | 'bass' | 'drums' | 'vocals' | 'other'
  // HPSS components (harmonic/percussive separation)
  | 'drums_harmonic' | 'drums_percussive'
  | 'other_harmonic' | 'other_percussive'
  | 'bass_harmonic' | 'bass_percussive'
  | 'vocals_harmonic' | 'vocals_percussive'
  // Sub-bands (frequency separation)
  | 'drums_low' | 'drums_mid' | 'drums_high'
  | 'other_mid' | 'other_high'
  // Derived
  | 'tension' | 'tonal_distance' | 'global';

/** Physical stems only */
export const PHYSICAL_STEMS: LinkTarget[] = ['bass', 'drums', 'vocals', 'other'];

/** HPSS component targets */
export const HPSS_TARGETS: LinkTarget[] = [
  'drums_harmonic', 'drums_percussive',
  'other_harmonic', 'other_percussive',
  'bass_harmonic', 'bass_percussive',
  'vocals_harmonic', 'vocals_percussive',
];

/** Sub-band targets */
export const SUBBAND_TARGETS: LinkTarget[] = [
  'drums_low', 'drums_mid', 'drums_high',
  'other_mid', 'other_high',
];

/** Derived targets */
export const DERIVED_TARGETS: LinkTarget[] = ['tension', 'tonal_distance', 'global'];

/** All link targets for UI dropdowns */
export const ALL_LINK_TARGETS: LinkTarget[] = [
  ...PHYSICAL_STEMS,
  ...HPSS_TARGETS,
  ...SUBBAND_TARGETS,
  ...DERIVED_TARGETS,
];

// =============================================================================
// Strength Range
// =============================================================================

/** Min/max bounds for SAE steering strength */
export interface StrengthRange {
  strengthMin: number;  // Stage left
  strengthMax: number;  // Stage right
  stageHome: number;    // Stage home (rest)
}

/** Default strength range */
export const DEFAULT_STRENGTH_RANGE: StrengthRange = {
  strengthMin: -30,
  strengthMax: 30,
  stageHome: 0,
};

// =============================================================================
// Dancer Ensemble Architecture: Ranking System
// =============================================================================

/**
 * Rank values for the Dancer Ensemble system.
 * - 1: Main dancer(s) - primary visual focus
 * - 2: Backup dancer(s) - supporting, visible
 * - 3: Background - ambient presence
 * - 4: Barely there - subtle texture
 * - null: Auto/available - can be promoted on surprise moments
 */
export type Rank = 1 | 2 | 3 | 4 | null;

/** Valid rank values for validation */
export const VALID_RANKS: readonly Rank[] = [1, 2, 3, 4, null] as const;

/** Rank display labels for UI */
export const RANK_LABELS: Record<Exclude<Rank, null>, string> = {
  1: 'Main',
  2: 'Backup',
  3: 'Background',
  4: 'Subtle',
};

/** Rank descriptions for tooltips */
export const RANK_DESCRIPTIONS: Record<1 | 2 | 3 | 4, string> = {
  1: 'Main dancer - primary visual focus',
  2: 'Backup dancer - supporting, visible',
  3: 'Background - ambient presence',
  4: 'Barely there - subtle texture',
};

export type SpatialMode = 'draw' | 'pitch_aligned';

/** Position source for dance model */
export type PositionSource =
  | 'auto'
  | 'pitch'
  | 'brightness'
  | 'chroma'
  | 'tension'
  | 'tension_global';

/** Intensity source for dance model */
export type IntensitySource =
  | 'energy_smooth'
  | 'transient'
  | 'flux'
  | 'envelope';

/** Silence behavior */
export type SilenceBehavior = 'drift_center' | 'hold_last';

/** Intensity curve */
export type IntensityCurve = 'linear' | 'gamma' | 'clip';

/** Feature category */
export type FeatureCategory = string;

/** A single SAE feature option */
export interface FeatureOption {
  id: number;
  label: string;
  category: FeatureCategory;
}

/** Configuration for one steering slot (Dancer Ensemble architecture).
 * A slot is whatever the backend steers per unit — a UNet block for SAE,
 * a concept slot for MF-RAE. Slot names come from the capability manifest;
 * nothing in the frontend hardcodes them.
 *
 * Note: SLERP rankings are configured per-destination in ReactiveConfig,
 * not per-slot. This keeps SAE steering separate from destination control.
 */
export interface SlotMapping {
  slot: string;
  linkTarget: LinkTarget;
  featureId: number;
  featureLabel: string;
  strengthRange: StrengthRange;
  enabled: boolean;
  autoConfig: boolean;  // If true, derive channel/layer/spatial from classification

  // Dancer Ensemble ranking (1-4 or null for auto)
  saeRank: Rank;  // Rank for SAE feature steering

  spatialMode: SpatialMode;
  spatialMask: number[];

  // Intensity overrides
  intensitySource?: IntensitySource;
  intensityCurve?: IntensityCurve;
  intensityGamma?: number;
}

/** Server snapshot for slot config sync — the fields we read from the
 * `slot_configs` payload. (The backend also emits a legacy `block` key and a
 * duplicate `block_configs` message for pre-Phase-4 clients; we type
 * neither.) */
export interface SlotConfigSnapshot {
  slot: string;
  link_target: LinkTarget;
  strength_min: number;
  strength_max: number;
  stage_home?: number;
  feature_id: number;
  enabled: boolean;
  auto_config: boolean;
  sae_rank: Rank;
  spatial_mode: SpatialMode;
  spatial_mask?: number[];
  intensity_source?: IntensitySource;
  intensity_curve?: IntensityCurve;
  intensity_gamma?: number;
}

/** Result from audio upload */
export interface AudioUploadResult {
  audioId: string;
  stems: Stem[];
  duration: number;
}

/** Real-time activity levels for all stems */
export interface StemActivity {
  bass: number;
  drums: number;
  vocals: number;
  other: number;
  audioTime: number;
}

/** Track info sent from backend after audio analysis */
export interface TrackInfo {
  type: 'track_info';
  audio_id: string;
  duration: number;
  bpm: number;
  stems: string[];
}

/** WebSocket message: Start SAE steering */
export interface StartSAESteeringMessage {
  action: 'start_sae_steering';
  audio_id: string;
}

/** Audio feature channels from backend analysis */
export type AudioChannel = 'envelope' | 'energy_smooth' | 'transient' | 'flux' | 'brightness' | 'flash' | 'sustain';

/** Per-stem channel data */
export interface StemChannelData {
  envelope: number;      // Raw RMS energy
  energy_smooth: number; // Asymmetric smoothed (fast attack, slow release)
  transient: number;     // Binary peak mask (1.0 at hits)
  flux: number;          // Spectral change / onset strength
  brightness: number;    // Spectral centroid (timbre)
  flash: number;         // Ultra-fast transient "pop"
  sustain: number;       // Slower trailing "glow"
}

/** Per-block activity (physics + raw) */
export interface BlockActivityData {
  raw: number;
  physics: number;
}

/** All stem types (physical + virtual) */
export type AllStems =
  | 'bass' | 'drums' | 'vocals' | 'other'
  | 'drums_low' | 'drums_mid' | 'drums_high' | 'other_mid' | 'other_high';

/** Computed prominence for a stem (from ProminenceEngine) */
export interface StemProminence {
  prominence: number;      // Current computed prominence [0, 1]
  surprise_active: boolean; // True when stem is temporarily promoted
  rank: Rank;              // User-assigned rank (for UI display)
}

/** Server message: Extended activity with all channels for all 8 stems */
export interface ExtendedStemActivityMessage {
  type: 'extended_activity';
  audio_time: number;
  stems: Record<AllStems, StemChannelData>;
  // Dancer Ensemble: computed prominence per stem (optional for backwards compat)
  prominence?: Record<string, StemProminence>;
  // Per-slot activity keyed by slot name (optional; used for UI physics sync)
  blocks?: Record<string, BlockActivityData>;
}

/** Server message: slot config snapshot */
export interface SlotConfigsMessage {
  type: 'slot_configs';
  configs: Record<string, SlotConfigSnapshot>;
}

/** Union type for all activity messages */
export type ActivityMessage = ExtendedStemActivityMessage;

// =============================================================================
// WebSocket Message Types
// =============================================================================

/** WebSocket message: update one steering slot (the vocabulary the app's
 * own controls send). */
export interface UpdateSlotConfigMessage {
  action: 'update_slot_config';
  slot: string;
  link_target?: LinkTarget;
  strength_min?: number;
  strength_max?: number;
  stage_home?: number;
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

/** WebSocket message: legacy update_block_config — the frozen Hermes agent
 * dialect. Only agentPlanApply forwards this; app controls send
 * UpdateSlotConfigMessage. */
export interface UpdateBlockConfigMessage {
  action: 'update_block_config';
  block: string;
  link_target?: LinkTarget;
  strength_min?: number;
  strength_max?: number;
  stage_home?: number;
  feature_id?: number;
  enabled?: boolean;
  auto_config?: boolean;
  // Dancer Ensemble ranking (1-4 or null for auto)
  sae_rank?: 1 | 2 | 3 | 4 | null;
  spatial_mode?: SpatialMode;
  spatial_mask?: number[];
  // Intensity overrides
  stage_left?: number;
  stage_right?: number;
  intensity_source?: IntensitySource;
  intensity_curve?: IntensityCurve;
  intensity_gamma?: number;
}

