/**
 * slotControls — the single write path for slot configuration.
 *
 * Each function performs the optimistic local store update and sends the
 * matching wire message. Plain module functions (no React): components bind
 * them directly, so no handler props need to travel through the tree.
 *
 * These send the `update_slot_config` wire vocabulary; only the frozen
 * Hermes agent dialect still speaks update_block_config.
 */

import { useSlotStore } from "../stores/useSlotStore";
import { sendUpdateSlotConfig } from "./wire";
import type {
  IntensityCurve,
  IntensitySource,
  LinkTarget,
  Rank,
  SpatialMode,
  StrengthRange,
} from "../types/sae";

export function handleSlotLinkTargetChange(slot: string, linkTarget: LinkTarget): void {
  useSlotStore.getState().setSlotLinkTarget(slot, linkTarget);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, link_target: linkTarget });
}

export function handleSlotFeatureChange(
  slot: string,
  featureId: number,
  featureLabel: string,
): void {
  useSlotStore.getState().setSlotFeature(slot, featureId, featureLabel);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, feature_id: featureId });
}

export function handleSlotStrengthRangeChange(slot: string, range: StrengthRange): void {
  useSlotStore.getState().setSlotStrengthRange(slot, range);
  // strength_* is the steering amplitude; stage_* belongs to the
  // destination dance model and must not mirror strengths.
  sendUpdateSlotConfig({
    action: "update_slot_config",
    slot,
    strength_min: range.strengthMin,
    strength_max: range.strengthMax,
    stage_home: range.stageHome,
  });
}

export function handleSlotAutoConfigChange(slot: string, autoConfig: boolean): void {
  useSlotStore.getState().setSlotAutoConfig(slot, autoConfig);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, auto_config: autoConfig });
}

export function handleSlotSpatialModeChange(slot: string, spatialMode: SpatialMode): void {
  useSlotStore.getState().setSlotSpatialMode(slot, spatialMode);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, spatial_mode: spatialMode });
}

export function handleSlotSpatialMaskChange(slot: string, mask: number[]): void {
  useSlotStore.getState().setSlotSpatialMask(slot, mask);
  sendUpdateSlotConfig({
    action: "update_slot_config",
    slot,
    spatial_mode: "draw",
    spatial_mask: mask,
  });
}

export function handleSlotIntensitySourceChange(slot: string, source: IntensitySource): void {
  useSlotStore.getState().setSlotIntensitySource(slot, source);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, intensity_source: source });
}

export function handleSlotIntensityCurveChange(slot: string, curve: IntensityCurve): void {
  useSlotStore.getState().setSlotIntensityCurve(slot, curve);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, intensity_curve: curve });
}

export function handleSlotIntensityGammaChange(slot: string, gamma: number): void {
  useSlotStore.getState().setSlotIntensityGamma(slot, gamma);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, intensity_gamma: gamma });
}

export function handleSlotSaeRankChange(slot: string, rank: Rank): void {
  useSlotStore.getState().setSlotSaeRank(slot, rank);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, sae_rank: rank });
}

export function handleToggleSlot(slot: string): void {
  const mapping = useSlotStore.getState().slots[slot];
  if (!mapping) return;
  const newEnabled = !mapping.enabled;
  useSlotStore.getState().setSlotEnabled(slot, newEnabled);
  sendUpdateSlotConfig({ action: "update_slot_config", slot, enabled: newEnabled });
}
