/**
 * Block Store - Zustand store for block-centric SAE steering
 *
 * Phase 1-2: Each UNet block is driven by a selected stem with full config.
 * This replaces the stem-centric approach where stems were the primary entity.
 *
 * Features:
 * - In-memory block mappings (persistence currently disabled)
 * - Preset save/load system (per session)
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type {
  BlockCode,
  BlockMapping,
  BlockConfigSnapshot,
  LinkTarget,
  Rank,
  SpatialMode,
  StemActivity,
  StrengthRange,
  TrackInfo,
} from '../types/sae';
import { DEFAULT_STRENGTH_RANGE } from '../types/sae';

// =============================================================================
// TYPES
// =============================================================================

type BlockMappings = Record<BlockCode, BlockMapping>;

/** Steering mode: auto selects params based on stem, manual is fully user-controlled */
export type SteeringMode = 'auto' | 'manual';

// =============================================================================
// DEFAULT MAPPINGS
// =============================================================================

const randFeatureId = () => Math.floor(Math.random() * 5120);

function makeDefaultMappings(): BlockMappings {
  const r = [randFeatureId(), randFeatureId(), randFeatureId(), randFeatureId()];
  return {
    'down.2.1': {
      block: 'down.2.1',
      linkTarget: 'bass',
      featureId: r[0],
      featureLabel: `#${r[0]}`,
      strengthRange: { ...DEFAULT_STRENGTH_RANGE },
      enabled: false,
      autoConfig: true,
      saeRank: 1,
      spatialMode: 'draw',
      spatialMask: Array(256).fill(1),
      intensitySource: 'energy_smooth',
      intensityCurve: 'linear',
      intensityGamma: 1,
    },
    'mid.0': {
      block: 'mid.0',
      linkTarget: 'vocals',
      featureId: r[1],
      featureLabel: `#${r[1]}`,
      strengthRange: { ...DEFAULT_STRENGTH_RANGE },
      enabled: false,
      autoConfig: true,
      saeRank: 2,
      spatialMode: 'draw',
      spatialMask: Array(256).fill(1),
      intensitySource: 'energy_smooth',
      intensityCurve: 'linear',
      intensityGamma: 1,
    },
    'up.0.0': {
      block: 'up.0.0',
      linkTarget: 'drums',
      featureId: r[2],
      featureLabel: `#${r[2]}`,
      strengthRange: { ...DEFAULT_STRENGTH_RANGE },
      enabled: false,
      autoConfig: true,
      saeRank: 1,
      spatialMode: 'draw',
      spatialMask: Array(256).fill(1),
      intensitySource: 'transient',
      intensityCurve: 'linear',
      intensityGamma: 1,
    },
    'up.0.1': {
      block: 'up.0.1',
      linkTarget: 'other_high',
      featureId: r[3],
      featureLabel: `#${r[3]}`,
      strengthRange: { ...DEFAULT_STRENGTH_RANGE },
      enabled: false,
      autoConfig: true,
      saeRank: null,
      spatialMode: 'draw',
      spatialMask: Array(256).fill(1),
      intensitySource: 'energy_smooth',
      intensityCurve: 'linear',
      intensityGamma: 1,
    },
  };
}

const DEFAULT_BLOCK_MAPPINGS: BlockMappings = makeDefaultMappings();

// =============================================================================
// STORE INTERFACE
// =============================================================================

interface BlockState {
  /** Block-centric mappings with LinkTarget, StrengthRange, focusWeight */
  blockMappings: BlockMappings;

  /** Real-time stem activity levels (from server telemetry) */
  activity: StemActivity;

  /** Track metadata from backend (BPM, available stems, duration) */
  trackInfo: TrackInfo | null;

  /** Steering mode: auto auto-selects params on stem change, manual is user-controlled */
  steeringMode: SteeringMode;

  // === Block Actions ===
  setBlockLinkTarget: (block: BlockCode, linkTarget: LinkTarget) => void;
  setBlockStrengthRange: (block: BlockCode, range: StrengthRange) => void;
  setBlockAutoConfig: (block: BlockCode, autoConfig: boolean) => void;
  setBlockFeature: (block: BlockCode, featureId: number, featureLabel: string) => void;
  // Spatial controls
  setBlockSpatialMode: (block: BlockCode, spatialMode: SpatialMode) => void;
  setBlockSpatialMask: (block: BlockCode, mask: number[]) => void;
  setBlockIntensitySource: (block: BlockCode, source: BlockMapping['intensitySource']) => void;
  setBlockIntensityCurve: (block: BlockCode, curve: BlockMapping['intensityCurve']) => void;
  setBlockIntensityGamma: (block: BlockCode, gamma: number) => void;
  setBlockEnabled: (block: BlockCode, enabled: boolean) => void;
  setBlockSaeRank: (block: BlockCode, rank: Rank) => void;
  applyBlockConfigs: (configs: Record<BlockCode, BlockConfigSnapshot>) => void;

  // === Steering Mode ===
  setSteeringMode: (mode: SteeringMode) => void;

  // === Global Actions ===
  setActivity: (activity: StemActivity) => void;
  setTrackInfo: (info: TrackInfo) => void;
  resetToDefaults: () => void;
}

// =============================================================================
// INITIAL STATE
// =============================================================================

const INITIAL_ACTIVITY: StemActivity = {
  bass: 0,
  drums: 0,
  vocals: 0,
  other: 0,
  audioTime: 0,
};

// =============================================================================
// STORE IMPLEMENTATION
// =============================================================================

export const useBlockStore = create<BlockState>()(
  subscribeWithSelector(
      (set) => ({
        blockMappings: { ...DEFAULT_BLOCK_MAPPINGS },
        activity: INITIAL_ACTIVITY,
        trackInfo: null,
        steeringMode: 'auto' as SteeringMode,

        // --- Link Target ---
        setBlockLinkTarget: (block, linkTarget) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                linkTarget,
              },
            },
          })),

        // --- Strength Range (no clamping - allow any values) ---
        setBlockStrengthRange: (block, range) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                strengthRange: {
                  strengthMin: range.strengthMin,
                  strengthMax: range.strengthMax,
                  stageHome: range.stageHome,
                },
              },
            },
          })),

        // --- Auto Config ---
        setBlockAutoConfig: (block, autoConfig) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                autoConfig,
              },
            },
          })),

        // --- Feature Selection ---
        setBlockFeature: (block, featureId, featureLabel) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                featureId,
                featureLabel,
              },
            },
          })),

        setBlockSpatialMode: (block, spatialMode) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                spatialMode,
              },
            },
          })),

        setBlockSpatialMask: (block, mask) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                spatialMask: mask,
              },
            },
          })),

        // --- Intensity Overrides ---
        setBlockIntensitySource: (block, source) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                intensitySource: source,
              },
            },
          })),

        setBlockIntensityCurve: (block, curve) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                intensityCurve: curve,
              },
            },
          })),

        setBlockIntensityGamma: (block, gamma) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                intensityGamma: gamma,
              },
            },
          })),

        // --- Enable/Disable ---
        setBlockEnabled: (block, enabled) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                enabled,
              },
            },
          })),

        // --- SAE Rank ---
        setBlockSaeRank: (block, rank) =>
          set((state) => ({
            blockMappings: {
              ...state.blockMappings,
              [block]: {
                ...state.blockMappings[block],
                saeRank: rank,
              },
            },
          })),

        // --- Apply Server Snapshot ---
        applyBlockConfigs: (configs) =>
          set((state) => {
            const next = { ...state.blockMappings };
            (Object.entries(configs) as [BlockCode, BlockConfigSnapshot][]).forEach(
              ([block, cfg]) => {
                next[block] = {
                  ...next[block],
                  block: cfg.block,
                  linkTarget: cfg.link_target,
                  strengthRange: {
                    strengthMin: cfg.stage_left ?? cfg.strength_min,
                    strengthMax: cfg.stage_right ?? cfg.strength_max,
                    stageHome: cfg.stage_home ?? next[block].strengthRange.stageHome,
                  },
                  featureId: cfg.feature_id,
                  enabled: cfg.enabled,
                  autoConfig: cfg.auto_config,
                  saeRank: cfg.sae_rank,
                  spatialMode: cfg.spatial_mode,
                  spatialMask: cfg.spatial_mask ?? next[block].spatialMask,
                  intensitySource: cfg.intensity_source ?? next[block].intensitySource,
                  intensityCurve: cfg.intensity_curve ?? next[block].intensityCurve,
                  intensityGamma: cfg.intensity_gamma ?? next[block].intensityGamma,
                };
              },
            );
            return { blockMappings: next };
          }),

        // --- Steering Mode ---
        setSteeringMode: (mode) => set({ steeringMode: mode }),

        // --- Global ---
        setActivity: (activity) => set({ activity }),
        setTrackInfo: (info) => set({ trackInfo: info }),

        resetToDefaults: () =>
          set({
            blockMappings: makeDefaultMappings(),
            activity: INITIAL_ACTIVITY,
            trackInfo: null,
            steeringMode: 'auto',
          }),
      })
  )
);

// =============================================================================
// ATOMIC SELECTORS (efficient re-renders)
// =============================================================================

export const useBlockMappings = () => useBlockStore((s) => s.blockMappings);
export const useBlockMapping = (block: BlockCode) =>
  useBlockStore((s) => s.blockMappings[block]);
export const useStemActivity = () => useBlockStore((s) => s.activity);
export const useTrackInfo = () => useBlockStore((s) => s.trackInfo);
export const useSteeringMode = () => useBlockStore((s) => s.steeringMode);

// =============================================================================
// DIRECT ACTIONS (stable references, no hook subscription)
// =============================================================================

export const blockActions = {
  setBlockLinkTarget: (...args: Parameters<BlockState['setBlockLinkTarget']>) =>
    useBlockStore.getState().setBlockLinkTarget(...args),
  setBlockStrengthRange: (...args: Parameters<BlockState['setBlockStrengthRange']>) =>
    useBlockStore.getState().setBlockStrengthRange(...args),
  setBlockAutoConfig: (...args: Parameters<BlockState['setBlockAutoConfig']>) =>
    useBlockStore.getState().setBlockAutoConfig(...args),
  setBlockFeature: (...args: Parameters<BlockState['setBlockFeature']>) =>
    useBlockStore.getState().setBlockFeature(...args),
  // Spatial controls
  setBlockSpatialMode: (...args: Parameters<BlockState['setBlockSpatialMode']>) =>
    useBlockStore.getState().setBlockSpatialMode(...args),
  setBlockSpatialMask: (...args: Parameters<BlockState['setBlockSpatialMask']>) =>
    useBlockStore.getState().setBlockSpatialMask(...args),
  setBlockIntensitySource: (...args: Parameters<BlockState['setBlockIntensitySource']>) =>
    useBlockStore.getState().setBlockIntensitySource(...args),
  setBlockIntensityCurve: (...args: Parameters<BlockState['setBlockIntensityCurve']>) =>
    useBlockStore.getState().setBlockIntensityCurve(...args),
  setBlockIntensityGamma: (...args: Parameters<BlockState['setBlockIntensityGamma']>) =>
    useBlockStore.getState().setBlockIntensityGamma(...args),
  setBlockEnabled: (...args: Parameters<BlockState['setBlockEnabled']>) =>
    useBlockStore.getState().setBlockEnabled(...args),
  setBlockSaeRank: (...args: Parameters<BlockState['setBlockSaeRank']>) =>
    useBlockStore.getState().setBlockSaeRank(...args),
  applyBlockConfigs: (...args: Parameters<BlockState['applyBlockConfigs']>) =>
    useBlockStore.getState().applyBlockConfigs(...args),
  setActivity: (...args: Parameters<BlockState['setActivity']>) =>
    useBlockStore.getState().setActivity(...args),
  setTrackInfo: (...args: Parameters<BlockState['setTrackInfo']>) =>
    useBlockStore.getState().setTrackInfo(...args),
  resetToDefaults: () => useBlockStore.getState().resetToDefaults(),
  setSteeringMode: (...args: Parameters<BlockState['setSteeringMode']>) =>
    useBlockStore.getState().setSteeringMode(...args),
};

// =============================================================================
// COMPUTED SELECTORS
// =============================================================================

/** Get all enabled blocks */
export const useEnabledBlocks = () =>
  useBlockStore((s) =>
    (Object.entries(s.blockMappings) as [BlockCode, BlockMapping][])
      .filter(([, m]) => m.enabled)
      .map(([code]) => code)
  );

/** Get activity for a block's link target */
export const useBlockActivity = (block: BlockCode) =>
  useBlockStore((s) => {
    const mapping = s.blockMappings[block];
    const linkTarget = mapping.linkTarget;
    // Derived targets don't have direct activity
    if (linkTarget === 'tension' || linkTarget === 'tonal_distance' || linkTarget === 'global') {
      return 0;
    }
    // Extract physical stem from compound names
    for (const base of ['drums', 'vocals', 'bass', 'other'] as const) {
      if (linkTarget.startsWith(base)) {
        return s.activity[base] ?? 0;
      }
    }
    return 0;
  });
