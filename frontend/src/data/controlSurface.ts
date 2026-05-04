import { BLOCKS } from './features';
import {
  DERIVED_TARGETS,
  HPSS_TARGETS,
  PHYSICAL_STEMS,
  RANK_DESCRIPTIONS,
  SUBBAND_TARGETS,
  DEFAULT_STRENGTH_RANGE,
  type BlockCode,
} from '../types/sae';
import { DEFAULT_REACTIVE_CONFIG, DEFAULT_STEM_RANKINGS } from '../types/destinations';
import {
  INTENSITY_CURVE_OPTIONS,
  INTENSITY_SOURCE_OPTIONS,
  POSITION_SOURCE_OPTIONS,
  SILENCE_OPTIONS,
} from './options';

export const CONTROL_STATE_VERSION = 'hamba-control-state/v1';
export const CONTROL_SURFACE_VERSION = 'hamba-control-surface/v1';

export const BLOCK_CONTROL_METADATA: Record<BlockCode, {
  label: string;
  description: string;
  role: string;
  featureCount: number;
}> = {
  'down.2.1': {
    label: BLOCKS['down.2.1'].name,
    description: BLOCKS['down.2.1'].description,
    role: 'composition, scene, mood, character, global visual world',
    featureCount: 5120,
  },
  'mid.0': {
    label: BLOCKS['mid.0'].name,
    description: BLOCKS['mid.0'].description,
    role: 'abstract structure, spatial layout, density, contrast, symmetry, depth',
    featureCount: 5120,
  },
  'up.0.0': {
    label: BLOCKS['up.0.0'].name,
    description: BLOCKS['up.0.0'].description,
    role: 'local detail, object details, faces, body parts, accessories, edges',
    featureCount: 5120,
  },
  'up.0.1': {
    label: BLOCKS['up.0.1'].name,
    description: BLOCKS['up.0.1'].description,
    role: 'style, texture, pattern, lighting, material, color palette',
    featureCount: 5120,
  },
};

const DEFAULT_BLOCK_SURFACE: Record<BlockCode, {
  enabled: boolean;
  link_target: string;
  sae_rank: number | null;
  intensity_source: string;
  intensity_curve: string;
  intensity_gamma: number;
  strength_min: number;
  stage_home: number;
  strength_max: number;
  spatial_mode: string;
  spatial_mask: string;
  auto_config: boolean;
}> = {
  'down.2.1': {
    enabled: false,
    link_target: 'bass',
    sae_rank: 1,
    intensity_source: 'energy_smooth',
    intensity_curve: 'linear',
    intensity_gamma: 1,
    strength_min: DEFAULT_STRENGTH_RANGE.strengthMin,
    stage_home: DEFAULT_STRENGTH_RANGE.stageHome,
    strength_max: DEFAULT_STRENGTH_RANGE.strengthMax,
    spatial_mode: 'draw',
    spatial_mask: 'full 16x16 mask',
    auto_config: true,
  },
  'mid.0': {
    enabled: false,
    link_target: 'vocals',
    sae_rank: 2,
    intensity_source: 'energy_smooth',
    intensity_curve: 'linear',
    intensity_gamma: 1,
    strength_min: DEFAULT_STRENGTH_RANGE.strengthMin,
    stage_home: DEFAULT_STRENGTH_RANGE.stageHome,
    strength_max: DEFAULT_STRENGTH_RANGE.strengthMax,
    spatial_mode: 'draw',
    spatial_mask: 'full 16x16 mask',
    auto_config: true,
  },
  'up.0.0': {
    enabled: false,
    link_target: 'drums',
    sae_rank: 1,
    intensity_source: 'transient',
    intensity_curve: 'linear',
    intensity_gamma: 1,
    strength_min: DEFAULT_STRENGTH_RANGE.strengthMin,
    stage_home: DEFAULT_STRENGTH_RANGE.stageHome,
    strength_max: DEFAULT_STRENGTH_RANGE.strengthMax,
    spatial_mode: 'draw',
    spatial_mask: 'full 16x16 mask',
    auto_config: true,
  },
  'up.0.1': {
    enabled: false,
    link_target: 'other_high',
    sae_rank: null,
    intensity_source: 'energy_smooth',
    intensity_curve: 'linear',
    intensity_gamma: 1,
    strength_min: DEFAULT_STRENGTH_RANGE.strengthMin,
    stage_home: DEFAULT_STRENGTH_RANGE.stageHome,
    strength_max: DEFAULT_STRENGTH_RANGE.strengthMax,
    spatial_mode: 'draw',
    spatial_mask: 'full 16x16 mask',
    auto_config: true,
  },
};

export function buildControlSurface() {
  return {
    version: CONTROL_SURFACE_VERSION,
    state_contract: {
      live_state_tool: 'hamba_get_state',
      static_surface_tool: 'hamba_get_control_surface',
      live_state_shape: {
        summary: 'quick glance: active blocks, prompt mode, composition motion',
        blocks: 'full current SAE block parameters keyed by block id',
        prompt: 'full current prompt destination parameters',
        composition: 'current latent seed and noise circle-walk parameters',
      },
    },
    defaults: {
      blank_start: {
        meaning: 'No visual has been staged yet: all SAE blocks disabled, prompt A/B empty, latent seed A/B empty.',
        agent_guidance: [
          'This is not a blocker. A fresh plan should initialize prompt A/B, latent seed A/B, composition, and selected SAE layers.',
          'Do not ask the user to press play just to set durable controls.',
          'If no song analysis exists, say that the song has not exposed DSP evidence yet and build a conservative starter rig only from user intent/control defaults.',
        ],
      },
      blocks: DEFAULT_BLOCK_SURFACE,
      prompt: {
        mode: 'reactive',
        ui_mode: 'GLOBAL',
        destination_a: null,
        destination_b: null,
        stage: {
          left: DEFAULT_REACTIVE_CONFIG.stageLeft,
          home: DEFAULT_REACTIVE_CONFIG.stageHome,
          right: DEFAULT_REACTIVE_CONFIG.stageRight,
        },
        position_source: DEFAULT_REACTIVE_CONFIG.positionSource,
        intensity_source: DEFAULT_REACTIVE_CONFIG.intensitySource,
        intensity_curve: DEFAULT_REACTIVE_CONFIG.intensityCurve,
        intensity_gamma: DEFAULT_REACTIVE_CONFIG.intensityGamma,
        stem_rankings: DEFAULT_STEM_RANKINGS,
        rank_weights: DEFAULT_REACTIVE_CONFIG.rankWeights,
        blend_slew_rate: DEFAULT_REACTIVE_CONFIG.blendSlewRate,
      },
      latent: {
        seed_a: null,
        seed_b: null,
        guidance: 'Set latent A and B with seed destinations; use composition distance/mode for noise motion.',
      },
      composition: {
        distance: 1.0,
        mode: 'auto',
      },
    },
    blocks: Object.fromEntries(
      Object.entries(BLOCK_CONTROL_METADATA).map(([block, metadata]) => [
        block,
        {
          ...metadata,
          runtime_model: 'sae_min_max_strength',
          parameters: {
            enabled: 'turn SAE steering for this block on/off',
            link_target: 'audio source that drives this block',
            feature_id: 'numeric SAE feature id in [0, 5119]',
            feature_label: 'frontend/operator label; backend receives the numeric id',
            sae_rank: {
              1: RANK_DESCRIPTIONS[1],
              2: RANK_DESCRIPTIONS[2],
              3: RANK_DESCRIPTIONS[3],
              4: RANK_DESCRIPTIONS[4],
              null: 'Auto/available for surprise promotion',
            },
            strength_min: 'SAE strength when block physics is 0',
            strength_max: 'SAE strength when block physics is 1',
            stage_home: 'stored for SAE blocks, but not used by current SAE runtime',
            spatial_mode: {
              draw: 'use the 16x16 spatial mask',
              pitch_aligned: 'derive spatial placement from pitch',
            },
            spatial_mask: '256 values for the 16x16 draw mask',
            intensity_source: Object.fromEntries(
              INTENSITY_SOURCE_OPTIONS.map((option) => [option.value, option.description]),
            ),
            intensity_curve: Object.fromEntries(
              INTENSITY_CURVE_OPTIONS.map((option) => [option.value, option.description]),
            ),
            intensity_gamma: 'gamma exponent used when intensity_curve is gamma',
            auto_config: 'derive some defaults from the selected audio target when available',
          },
          stage_math: {
            note: 'SAE blocks currently do not use prompt-style stage math.',
            formula: 'strength = strength_min + physics_value * (strength_max - strength_min)',
          },
        },
      ]),
    ),
    link_targets: {
      signal_model: [
        'Link targets are normalized, precomputed audio-analysis curves sampled at runtime.',
        'They are not fixed semantic labels. A synth, pluck, vocal chop, or noisy texture may land in different targets depending on the source separator and the mix.',
        'Choose targets by the measured behavior you want to drive, then use music-window/activity data when the song makes the label ambiguous.',
      ],
      physical: PHYSICAL_STEMS,
      hpss: HPSS_TARGETS,
      subband: SUBBAND_TARGETS,
      derived: DERIVED_TARGETS,
      groups: {
        physical: {
          targets: PHYSICAL_STEMS,
          calculation: 'Source-separated stems estimated from the uploaded mix before feature extraction.',
          runtime_signal: 'sample_stem(link_target, time, layer) using flash, sustain, or combined energy.',
          guidance: 'Use for broad musical roles when the whole separated source should move together.',
          notes: {
            bass: 'separator estimate of the bass source; often low-end bassline, but may include synth bass or sub-heavy material depending on the mix.',
            drums: 'separator estimate of drum/percussion source; can include cymbals, kicks, snares, loops, and other transient percussion.',
            vocals: 'separator estimate of vocal source; may include lead vocals, backing vocals, chops, and vocal-like material.',
            other: 'separator remainder; often keys, guitars, pads, synths, strings, ambience, and anything not confidently bass/drums/vocals.',
          },
        },
        hpss: {
          targets: HPSS_TARGETS,
          calculation: 'Each physical stem is split with harmonic-percussive source separation; harmonic targets track sustained tonal energy, percussive targets track attack/noise energy.',
          runtime_signal: 'sample_harmonic_energy(base_stem, time) or sample_percussive_energy(base_stem, time).',
          guidance: 'Use when the user describes body vs hits inside the same stem, such as bassline sustain vs bass attacks or vocal melody vs consonant chops.',
          suffixes: {
            harmonic: 'sustained, tonal, ringing, pitched, or washed component inside the parent stem.',
            percussive: 'transient, noisy, struck, plucked, consonant, or attack component inside the parent stem.',
          },
        },
        subband: {
          targets: SUBBAND_TARGETS,
          calculation: 'Bandpass-filtered virtual stems derived from physical stems with offline Linkwitz-Riley/Butterworth filters.',
          runtime_signal: 'sample_stem(virtual_target, time, layer).',
          guidance: 'Use for frequency-specific routing when the visual idea names low impact, mid punch, high shimmer, air, or sparkle.',
          bands_hz: {
            drums_low: [20, 200],
            drums_mid: [200, 5000],
            drums_high: [5000, 16000],
            other_mid: [200, 4000],
            other_high: [4000, 16000],
          },
          parents: {
            drums_low: 'drums',
            drums_mid: 'drums',
            drums_high: 'drums',
            other_mid: 'other',
            other_high: 'other',
          },
        },
        derived: {
          targets: DERIVED_TARGETS,
          calculation: 'Aggregate curves computed from the analyzed song rather than one separated source.',
          runtime_signal: 'sample_aggregate_tension, sample_aggregate_tonal_distance, or average physical-stem activity.',
          guidance: 'Use for whole-track motion, harmonic journey, release, surprise, or global activity.',
          notes: {
            tension: 'energy-weighted aggregate harmonic tension across stems with tension curves.',
            tonal_distance: 'energy-weighted aggregate departure from the track tonal center.',
            global: 'average activity across physical stems.',
          },
        },
      },
      aliases: {
        kick: 'drums_low',
        thump: 'drums_low',
        snare: 'drums_mid',
        clap: 'drums_mid',
        hats: 'drums_high',
        cymbals: 'drums_high',
        sparkle: 'other_high',
        air: 'other_high',
        bass_hits: 'bass_percussive',
        bassline: 'bass_harmonic',
        vocal_melody: 'vocals_harmonic',
      },
    },
    prompt: {
      runtime_model: 'prompt_stage_position_intensity',
      modes: {
        slider: 'manual 0..1 crossfader; bypasses stage anchors',
        reactive: 'UI GLOBAL; ranked stems compute position and intensity',
        linked: 'one selected link target computes position and intensity',
      },
      stage_math: {
        applies_to: ['reactive', 'linked'],
        bypassed_by: ['slider'],
        formula: [
          'pos_value = stage_left + position * (stage_right - stage_left)',
          'output = stage_home + intensity * (pos_value - stage_home)',
          'blend = (output - stage_left) / (stage_right - stage_left)',
        ],
      },
      position_source: Object.fromEntries(
        POSITION_SOURCE_OPTIONS.map((option) => [option.value, option.description]),
      ),
      intensity_source: Object.fromEntries(
        INTENSITY_SOURCE_OPTIONS.map((option) => [option.value, option.description]),
      ),
      silence_behavior: Object.fromEntries(
        SILENCE_OPTIONS.map((option) => [option.value, option.description]),
      ),
      intensity_curve: Object.fromEntries(
        INTENSITY_CURVE_OPTIONS.map((option) => [option.value, option.description]),
      ),
      rank_semantics: {
        1: RANK_DESCRIPTIONS[1],
        2: RANK_DESCRIPTIONS[2],
        3: RANK_DESCRIPTIONS[3],
        4: RANK_DESCRIPTIONS[4],
        null: 'Auto/available',
      },
    },
    composition: {
      runtime_model: 'latent_noise_circle_walk',
      controls: {
        seed_a: 'latent/noise anchor A',
        seed_b: 'latent/noise anchor B',
        distance: 'circle radius in [0, 4]; lower is steadier, higher is more churn',
        mode: {
          auto: 'adaptive beat pulse plus continuous drift',
          pulse: 'beat-synced angular jumps',
          continuous: 'smooth drift without beat jumps',
        },
      },
      boundary: 'Do not use prompt reactive/link controls for latent space.',
    },
  };
}
