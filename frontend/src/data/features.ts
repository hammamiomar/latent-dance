/**
 * Audio + labeling vocabulary for the UI: stem options, stem colors, and SAE
 * label-category colors. Slot identity (names, display names, slot colors)
 * lives in the capability manifest — never here.
 */

import type { Stem } from '../types/sae';

// =============================================================================
// DROPDOWN OPTIONS
// =============================================================================

/** Stem selector options (9 total: 4 physical + 5 virtual) */
export const STEM_OPTIONS: { value: Stem; label: string; description: string }[] = [
  { value: 'bass', label: 'Bass', description: 'Full bass stem' },
  { value: 'drums', label: 'Drums', description: 'Full drum kit' },
  { value: 'vocals', label: 'Vocals', description: 'Vocal track' },
  { value: 'other', label: 'Other', description: 'Instruments & synths' },
  { value: 'drums_low', label: 'Kick', description: '<200 Hz drum thump' },
  { value: 'drums_mid', label: 'Snare', description: '200-5000 Hz rhythmic crack' },
  { value: 'drums_high', label: 'Hi-hats', description: '>5 kHz cymbal shimmer' },
  { value: 'other_mid', label: 'Harmony', description: '200-4000 Hz melody' },
  { value: 'other_high', label: 'Air', description: '>4 kHz atmosphere' },
];


// =============================================================================
// UI COLORS
// =============================================================================

/** Category colors (27 pipeline categories) */
export const CATEGORY_COLORS: Record<string, string> = {
  // Composition & Scene
  scene: '#b8863a',
  composition: '#a87a3a',
  setting: '#c8963a',

  // Subject & Character
  character: '#a85a7a',
  face: '#c86a8a',
  body: '#8a4a6a',
  action: '#b86a6a',
  subject: '#9a5a7a',

  // Objects & Details
  object: '#6a6a9a',
  object_detail: '#5a5a8a',
  accessory: '#7a7aaa',
  edge: '#4a4a7a',

  // Style & Mood
  mood: '#b8863a',
  style: '#7a5a9a',
  form: '#4a8a6a',
  expression: '#a85a7a',
  abstract: '#8a5a8a',

  // Texture & Pattern
  texture: '#4a8a7a',
  pattern: '#b86a3a',
  material: '#3a7a6a',
  shape: '#5a9a8a',

  // Color & Lighting
  color: '#5a7a9a',
  lighting: '#9a8a3a',

  // Spatial & Structure (mid.0 block)
  spatial: '#4a8a6a',
  symmetry: '#3a7a5a',
  border: '#5a9a7a',
  depth: '#2a6a4a',
  density: '#6aaa8a',
  contrast: '#8a9a5a',

  // Catch-all
  unclear: '#6a6a6a',
  unknown: '#5a5a5a',
};

/** Color with fallback for unknown categories */
export function getCategoryColor(category: string): string {
  return CATEGORY_COLORS[category] ?? CATEGORY_COLORS['unknown'];
}

/** Stem colors for UI (Muted earthy palette - matches index.css) */
export const STEM_COLORS: Record<Stem, string> = {
  // Physical stems - use CSS variables where possible
  bass: '#c45a2a',      // Muted orange - foundational (--color-stem-bass)
  drums: '#a84070',     // Muted magenta - rhythmic (--color-stem-drums)
  vocals: '#4a9eb0',    // Muted cyan - human presence (--color-stem-vocals)
  other: '#5a8a4a',     // Muted green - atmosphere (--color-stem-other)
  // Virtual stems - darker/lighter variants
  drums_low: '#a84a28', // Darker orange-red - kick impact
  drums_mid: '#b86060', // Muted coral - snare crack
  drums_high: '#985070', // Lighter magenta - sparkle
  other_mid: '#4a8a60', // Mid green - harmony
  other_high: '#6a9098', // Muted teal - air/texture
};

