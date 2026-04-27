import { useCallback } from "react";
import {
  useBlockStore,
  blockActions,
} from "../stores/useBlockStore";
import type { UpdateBlockConfigMessage } from "../types/sae";
import type {
  BlockCode,
  LinkTarget,
  Rank,
  StrengthRange,
  SpatialMode,
  IntensitySource,
  IntensityCurve,
} from "../types/sae";

interface BlockConfigHandlers {
  handleBlockLinkTargetChange: (block: BlockCode, linkTarget: LinkTarget) => void;
  handleBlockFeatureChange: (block: BlockCode, featureId: number, featureLabel: string) => void;
  handleBlockStrengthRangeChange: (block: BlockCode, range: StrengthRange) => void;
  handleBlockAutoConfigChange: (block: BlockCode, autoConfig: boolean) => void;
  handleBlockSpatialModeChange: (block: BlockCode, spatialMode: SpatialMode) => void;
  handleBlockSpatialMaskChange: (block: BlockCode, mask: number[]) => void;
  handleBlockIntensitySourceChange: (block: BlockCode, source: IntensitySource) => void;
  handleBlockIntensityCurveChange: (block: BlockCode, curve: IntensityCurve) => void;
  handleBlockIntensityGammaChange: (block: BlockCode, gamma: number) => void;
  handleBlockSaeRankChange: (block: BlockCode, rank: Rank) => void;
  handleToggleBlock: (block: BlockCode) => void;
}

export function useBlockConfigHandlers(
  sendUpdateBlockConfig: (message: UpdateBlockConfigMessage) => void
): BlockConfigHandlers {
  const handleBlockLinkTargetChange = useCallback(
    (block: BlockCode, linkTarget: LinkTarget) => {
      blockActions.setBlockLinkTarget(block, linkTarget);
      sendUpdateBlockConfig({ action: 'update_block_config', block, link_target: linkTarget });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockFeatureChange = useCallback(
    (block: BlockCode, featureId: number, featureLabel: string) => {
      blockActions.setBlockFeature(block, featureId, featureLabel);
      sendUpdateBlockConfig({ action: 'update_block_config', block, feature_id: featureId });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockStrengthRangeChange = useCallback(
    (block: BlockCode, range: StrengthRange) => {
      blockActions.setBlockStrengthRange(block, range);
      sendUpdateBlockConfig({
        action: 'update_block_config',
        block,
        strength_min: range.strengthMin,
        strength_max: range.strengthMax,
        stage_left: range.strengthMin,
        stage_right: range.strengthMax,
        stage_home: range.stageHome,
      });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockAutoConfigChange = useCallback(
    (block: BlockCode, autoConfig: boolean) => {
      blockActions.setBlockAutoConfig(block, autoConfig);
      sendUpdateBlockConfig({ action: 'update_block_config', block, auto_config: autoConfig });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockSpatialModeChange = useCallback(
    (block: BlockCode, spatialMode: SpatialMode) => {
      blockActions.setBlockSpatialMode(block, spatialMode);
      sendUpdateBlockConfig({ action: 'update_block_config', block, spatial_mode: spatialMode });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockSpatialMaskChange = useCallback(
    (block: BlockCode, mask: number[]) => {
      blockActions.setBlockSpatialMask(block, mask);
      sendUpdateBlockConfig({
        action: 'update_block_config',
        block,
        spatial_mode: 'draw',
        spatial_mask: mask,
      });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockIntensitySourceChange = useCallback(
    (block: BlockCode, source: IntensitySource) => {
      blockActions.setBlockIntensitySource(block, source);
      sendUpdateBlockConfig({ action: 'update_block_config', block, intensity_source: source });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockIntensityCurveChange = useCallback(
    (block: BlockCode, curve: IntensityCurve) => {
      blockActions.setBlockIntensityCurve(block, curve);
      sendUpdateBlockConfig({ action: 'update_block_config', block, intensity_curve: curve });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockIntensityGammaChange = useCallback(
    (block: BlockCode, gamma: number) => {
      blockActions.setBlockIntensityGamma(block, gamma);
      sendUpdateBlockConfig({ action: 'update_block_config', block, intensity_gamma: gamma });
    },
    [sendUpdateBlockConfig]
  );

  const handleBlockSaeRankChange = useCallback(
    (block: BlockCode, rank: Rank) => {
      blockActions.setBlockSaeRank(block, rank);
      sendUpdateBlockConfig({ action: 'update_block_config', block, sae_rank: rank });
    },
    [sendUpdateBlockConfig]
  );

  const handleToggleBlock = useCallback(
    (block: BlockCode) => {
      const mapping = useBlockStore.getState().blockMappings[block];
      const newEnabled = !mapping.enabled;
      blockActions.setBlockEnabled(block, newEnabled);
      sendUpdateBlockConfig({ action: 'update_block_config', block, enabled: newEnabled });
    },
    [sendUpdateBlockConfig]
  );

  return {
    handleBlockLinkTargetChange,
    handleBlockFeatureChange,
    handleBlockStrengthRangeChange,
    handleBlockAutoConfigChange,
    handleBlockSpatialModeChange,
    handleBlockSpatialMaskChange,
    handleBlockIntensitySourceChange,
    handleBlockIntensityCurveChange,
    handleBlockIntensityGammaChange,
    handleBlockSaeRankChange,
    handleToggleBlock,
  };
}
