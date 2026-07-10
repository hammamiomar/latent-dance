/**
 * Slot Store — steering slot configuration, keyed by backend slot name.
 *
 * The slot vocabulary comes from the capability manifest (useSessionStore
 * calls initFromCapabilities whenever a manifest arrives); nothing here
 * hardcodes slot names or counts. `order` is THE single index-identity
 * source: physics body i, orb DOM element i, 3D flower i, and tendril i all
 * mean slots[order[i]]. No other module may define slot ordering.
 */

import { create } from 'zustand';
import type { BackendCapabilities } from '../types/wire/capabilities';
import type {
  SlotMapping,
  SlotConfigSnapshot,
  IntensitySource,
  LinkTarget,
  Rank,
  SpatialMode,
  StrengthRange,
} from '../types/sae';
import { DEFAULT_STRENGTH_RANGE } from '../types/sae';

// =============================================================================
// DEFAULT SEEDS
// =============================================================================

/**
 * Per-index defaults for a fresh rig, cycling past four slots. Index-based
 * (not name-based) so any backend gets a musically sensible starting spread:
 * foundation, voice, hits, air — the historical SAE defaults.
 */
export const SLOT_SEEDS: ReadonlyArray<{
  linkTarget: LinkTarget;
  saeRank: Rank;
  intensitySource: IntensitySource;
}> = [
  { linkTarget: 'bass', saeRank: 1, intensitySource: 'energy_smooth' },
  { linkTarget: 'vocals', saeRank: 2, intensitySource: 'energy_smooth' },
  { linkTarget: 'drums', saeRank: 1, intensitySource: 'transient' },
  { linkTarget: 'other_high', saeRank: null, intensitySource: 'energy_smooth' },
];

function makeSlotMapping(
  name: string,
  index: number,
  capabilities: BackendCapabilities,
): SlotMapping {
  const seed = SLOT_SEEDS[index % SLOT_SEEDS.length];
  const [idMin, idMax] = capabilities.feature_id_range;
  const featureId = idMin + Math.floor(Math.random() * (idMax - idMin + 1));
  const [maskH, maskW] = capabilities.spatial_mask_shape;
  return {
    slot: name,
    linkTarget: seed.linkTarget,
    featureId,
    featureLabel: `#${featureId}`,
    strengthRange: { ...DEFAULT_STRENGTH_RANGE },
    enabled: false,
    autoConfig: true,
    saeRank: seed.saeRank,
    spatialMode: 'draw',
    spatialMask: Array(maskH * maskW).fill(1),
    intensitySource: seed.intensitySource,
    intensityCurve: 'linear',
    intensityGamma: 1,
  };
}

// =============================================================================
// STORE
// =============================================================================

interface SlotState {
  /** Slot configs keyed by slot name; empty until a manifest arrives. */
  slots: Record<string, SlotMapping>;
  /** Manifest slot order — the only index↔slot mapping in the app. */
  order: string[];

  /**
   * Materialize slots from a manifest. Idempotent by vocabulary: when the
   * slot names match the current order (e.g. the WS hello re-confirming the
   * bootstrap fetch), the user's current config is kept untouched.
   */
  initFromCapabilities: (capabilities: BackendCapabilities) => void;

  setSlotLinkTarget: (slot: string, linkTarget: LinkTarget) => void;
  setSlotStrengthRange: (slot: string, range: StrengthRange) => void;
  setSlotAutoConfig: (slot: string, autoConfig: boolean) => void;
  setSlotFeature: (slot: string, featureId: number, featureLabel: string) => void;
  setSlotSpatialMode: (slot: string, spatialMode: SpatialMode) => void;
  setSlotSpatialMask: (slot: string, mask: number[]) => void;
  setSlotIntensitySource: (slot: string, source: SlotMapping['intensitySource']) => void;
  setSlotIntensityCurve: (slot: string, curve: SlotMapping['intensityCurve']) => void;
  setSlotIntensityGamma: (slot: string, gamma: number) => void;
  setSlotEnabled: (slot: string, enabled: boolean) => void;
  setSlotSaeRank: (slot: string, rank: Rank) => void;

  /** Merge a server config snapshot (ACK) into the local slots. */
  applyConfigSnapshots: (configs: Record<string, SlotConfigSnapshot>) => void;
}

export const useSlotStore = create<SlotState>()((set) => {
  /** All single-field setters funnel here; unknown slot names are ignored
   * (the agent seam and UI can only name slots that exist). */
  const patchSlot = (slot: string, patch: Partial<SlotMapping>) =>
    set((state) => {
      const current = state.slots[slot];
      if (!current) return state;
      return { slots: { ...state.slots, [slot]: { ...current, ...patch } } };
    });

  return {
    slots: {},
    order: [],

    initFromCapabilities: (capabilities) =>
      set((state) => {
        const names = capabilities.slots.map((slot) => slot.name);
        const sameVocabulary =
          names.length === state.order.length &&
          names.every((name, i) => name === state.order[i]);
        if (sameVocabulary) return state;
        return {
          order: names,
          slots: Object.fromEntries(
            names.map((name, i) => [name, makeSlotMapping(name, i, capabilities)]),
          ),
        };
      }),

    setSlotLinkTarget: (slot, linkTarget) => patchSlot(slot, { linkTarget }),
    // No clamping — any strength values are allowed
    setSlotStrengthRange: (slot, range) =>
      patchSlot(slot, {
        strengthRange: {
          strengthMin: range.strengthMin,
          strengthMax: range.strengthMax,
          stageHome: range.stageHome,
        },
      }),
    setSlotAutoConfig: (slot, autoConfig) => patchSlot(slot, { autoConfig }),
    setSlotFeature: (slot, featureId, featureLabel) =>
      patchSlot(slot, { featureId, featureLabel }),
    setSlotSpatialMode: (slot, spatialMode) => patchSlot(slot, { spatialMode }),
    setSlotSpatialMask: (slot, mask) => patchSlot(slot, { spatialMask: mask }),
    setSlotIntensitySource: (slot, source) => patchSlot(slot, { intensitySource: source }),
    setSlotIntensityCurve: (slot, curve) => patchSlot(slot, { intensityCurve: curve }),
    setSlotIntensityGamma: (slot, gamma) => patchSlot(slot, { intensityGamma: gamma }),
    setSlotEnabled: (slot, enabled) => patchSlot(slot, { enabled }),
    setSlotSaeRank: (slot, rank) => patchSlot(slot, { saeRank: rank }),

    applyConfigSnapshots: (configs) =>
      set((state) => {
        const next = { ...state.slots };
        for (const [slot, cfg] of Object.entries(configs)) {
          const current = next[slot];
          if (!current) continue;
          next[slot] = {
            ...current,
            linkTarget: cfg.link_target,
            strengthRange: {
              // strength_* is authoritative — stage_* are legacy dance-model
              // fields stuck at backend defaults (±30) since the mirror was
              // removed; preferring them clobbered user ranges on every ACK
              strengthMin: cfg.strength_min,
              strengthMax: cfg.strength_max,
              stageHome: cfg.stage_home ?? current.strengthRange.stageHome,
            },
            featureId: cfg.feature_id,
            enabled: cfg.enabled,
            autoConfig: cfg.auto_config,
            saeRank: cfg.sae_rank,
            spatialMode: cfg.spatial_mode,
            spatialMask: cfg.spatial_mask ?? current.spatialMask,
            intensitySource: cfg.intensity_source ?? current.intensitySource,
            intensityCurve: cfg.intensity_curve ?? current.intensityCurve,
            intensityGamma: cfg.intensity_gamma ?? current.intensityGamma,
          };
        }
        return { slots: next };
      }),
  };
});

// =============================================================================
// ATOMIC SELECTORS (efficient re-renders)
// =============================================================================

export const useSlots = () => useSlotStore((s) => s.slots);
export const useSlotOrder = () => useSlotStore((s) => s.order);
