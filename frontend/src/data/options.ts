/**
 * Shared option arrays for UI selectors.
 *
 * Single source of truth — used by SlotConfigPanel and ReactiveConfigSection.
 */

import type {
  IntensityCurve,
  IntensitySource,
  PositionSource,
  SilenceBehavior,
} from '../types/sae';

export const POSITION_SOURCE_OPTIONS: Array<{ value: PositionSource; label: string; description: string }> = [
  { value: 'auto', label: 'Auto', description: 'Stem-based defaults' },
  { value: 'pitch', label: 'Pitch', description: 'Monophonic pitch height' },
  { value: 'chroma', label: 'Chroma', description: 'Polyphonic chroma centroid' },
  { value: 'brightness', label: 'Brightness', description: 'Spectral centroid position' },
  { value: 'tension', label: 'Tension', description: 'Per-stem harmonic tension' },
  { value: 'tension_global', label: 'Tension (Global)', description: 'Aggregate harmonic tension' },
];

export const INTENSITY_SOURCE_OPTIONS: Array<{ value: IntensitySource; label: string; description: string }> = [
  { value: 'energy_smooth', label: 'Energy', description: 'Smoothed envelope' },
  { value: 'transient', label: 'Transient', description: 'Onset/peak detection' },
  { value: 'flux', label: 'Flux', description: 'Spectral change rate' },
  { value: 'envelope', label: 'Envelope', description: 'Raw RMS energy' },
];

export const SILENCE_OPTIONS: Array<{ value: SilenceBehavior; label: string; description: string }> = [
  { value: 'hold_last', label: 'Hold', description: 'Freeze last position' },
  { value: 'drift_center', label: 'Drift', description: 'Relax toward center' },
];

export const INTENSITY_CURVE_OPTIONS: Array<{ value: IntensityCurve; label: string; description: string }> = [
  { value: 'linear', label: 'Linear', description: 'Direct — no shaping' },
  { value: 'gamma', label: 'Gamma', description: 'Power curve — reshape response' },
  { value: 'clip', label: 'Clip', description: 'Boost 1.5× — more aggressive' },
];
