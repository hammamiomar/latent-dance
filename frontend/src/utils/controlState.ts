import { BLOCK_CONTROL_METADATA, CONTROL_STATE_VERSION } from '../data/controlSurface';
import type { CompositionStateSnapshot } from '../types/composition';
import type { DestinationState } from '../types/destinations';
import type { BlockCode, BlockMapping } from '../types/sae';

interface BuildControlStateArgs {
  blockMappings: Record<BlockCode, BlockMapping>;
  latent: DestinationState;
  prompt: DestinationState;
  composition: CompositionStateSnapshot;
}

function destinationValue(destination: DestinationState['destinationA']) {
  if (!destination) return null;
  return {
    type: destination.type,
    label: destination.label,
    seed: destination.seed,
    prompt: destination.prompt,
  };
}

function promptUiMode(mode: DestinationState['mode']) {
  if (mode === 'reactive') return 'GLOBAL';
  if (mode === 'linked') return 'LINKED';
  return 'SLIDER';
}

function countActiveMaskCells(mask: number[]) {
  return mask.reduce((count, value) => count + (value > 0 ? 1 : 0), 0);
}

function blockSummary(block: BlockCode, mapping: BlockMapping) {
  const metadata = BLOCK_CONTROL_METADATA[block];
  return {
    block,
    label: metadata.label,
    enabled: mapping.enabled,
    rank: mapping.saeRank,
    link_target: mapping.linkTarget,
    feature_id: mapping.featureId,
    feature_label: mapping.featureLabel,
    intensity_source: mapping.intensitySource ?? 'energy_smooth',
    strength_min: mapping.strengthRange.strengthMin,
    strength_max: mapping.strengthRange.strengthMax,
    spatial_mode: mapping.spatialMode,
  };
}

function blockDetail(block: BlockCode, mapping: BlockMapping) {
  const metadata = BLOCK_CONTROL_METADATA[block];
  return {
    identity: {
      block,
      label: metadata.label,
      description: metadata.description,
      role: metadata.role,
      feature_count: metadata.featureCount,
    },
    enabled: mapping.enabled,
    rank: mapping.saeRank,
    auto_config: mapping.autoConfig,
    feature: {
      id: mapping.featureId,
      label: mapping.featureLabel,
    },
    link: {
      target: mapping.linkTarget,
    },
    response: {
      runtime_model: 'sae_min_max_strength',
      strength_min: mapping.strengthRange.strengthMin,
      strength_max: mapping.strengthRange.strengthMax,
      stage_left: mapping.strengthRange.strengthMin,
      stage_right: mapping.strengthRange.strengthMax,
      stage_home: mapping.strengthRange.stageHome,
      stage_home_runtime_effect: 'stored_not_used',
      intensity_source: mapping.intensitySource ?? 'energy_smooth',
      intensity_curve: mapping.intensityCurve ?? 'linear',
      intensity_gamma: mapping.intensityGamma ?? 1,
    },
    spatial: {
      mode: mapping.spatialMode,
      mask_active_cells: countActiveMaskCells(mapping.spatialMask),
      mask_size: mapping.spatialMask.length,
      mask: mapping.spatialMask,
    },
  };
}

function promptDetail(prompt: DestinationState) {
  const cfg = prompt.reactiveConfig;
  return {
    space: 'prompt',
    mode: prompt.mode,
    ui_mode: promptUiMode(prompt.mode),
    destination_a: destinationValue(prompt.destinationA),
    destination_b: destinationValue(prompt.destinationB),
    blend_position: prompt.blendPosition,
    link_target: prompt.linkTarget,
    reactive_config: {
      runtime_model: prompt.mode === 'slider'
        ? 'manual_crossfade'
        : 'prompt_stage_position_intensity',
      stage: {
        left: cfg.stageLeft,
        home: cfg.stageHome,
        right: cfg.stageRight,
      },
      position_source: cfg.positionSource,
      intensity_source: cfg.intensitySource,
      position_smoothing_ms: cfg.positionSmoothingMs,
      silence_behavior: cfg.silenceBehavior,
      drift_ms: cfg.driftMs,
      intensity_curve: cfg.intensityCurve,
      intensity_gamma: cfg.intensityGamma,
      stem_rankings: cfg.stemRankings,
      rank_weights: cfg.rankWeights,
      blend_slew_rate: cfg.blendSlewRate,
    },
  };
}

function compositionDetail(latent: DestinationState, composition: CompositionStateSnapshot) {
  return {
    runtime_model: 'latent_noise_circle_walk',
    seed_a: latent.destinationA?.seed ?? null,
    seed_b: latent.destinationB?.seed ?? null,
    destination_a: destinationValue(latent.destinationA),
    destination_b: destinationValue(latent.destinationB),
    distance: composition.distance,
    mode: composition.mode,
  };
}

export function buildControlState({
  blockMappings,
  latent,
  prompt,
  composition,
}: BuildControlStateArgs) {
  const entries = Object.entries(blockMappings) as [BlockCode, BlockMapping][];
  const promptState = promptDetail(prompt);
  const compositionState = compositionDetail(latent, composition);

  return {
    version: CONTROL_STATE_VERSION,
    summary: {
      enabled_block_count: entries.filter(([, mapping]) => mapping.enabled).length,
      blocks: entries.map(([block, mapping]) => blockSummary(block, mapping)),
      prompt: {
        mode: promptState.mode,
        ui_mode: promptState.ui_mode,
        destination_a: promptState.destination_a?.label ?? null,
        destination_b: promptState.destination_b?.label ?? null,
        position_source: promptState.reactive_config.position_source,
        intensity_source: promptState.reactive_config.intensity_source,
        stage: promptState.reactive_config.stage,
        blend_slew_rate: promptState.reactive_config.blend_slew_rate,
      },
      composition: {
        seed_a: compositionState.seed_a,
        seed_b: compositionState.seed_b,
        distance: compositionState.distance,
        mode: compositionState.mode,
      },
    },
    blocks: Object.fromEntries(
      entries.map(([block, mapping]) => [block, blockDetail(block, mapping)]),
    ),
    prompt: promptState,
    composition: compositionState,
    latent_boundary: {
      note: 'Latent uses seeds plus composition config. Prompt reactive/link controls are ignored by the backend for latent space.',
      raw_shared_destination_state: {
        mode: latent.mode,
        blend_position: latent.blendPosition,
        link_target: latent.linkTarget,
      },
    },
  };
}
