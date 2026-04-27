/**
 * useEffectsStore - Visual effects overlay state.
 *
 * Extracted from useAppCore to eliminate 15-prop drill through
 * DesktopApp → ModeBar and BrowserApp → FxPanel. Any component
 * can now read/toggle effects directly.
 */

import { create } from "zustand";

type EffectKey =
  | "showCrt"
  | "showDither"
  | "showEffectsPanel"
  | "showChromatic"
  | "showBloom"
  | "showVhsTracking"
  | "showHeavyGrain";

interface EffectsState {
  showCrt: boolean;
  showDither: boolean;
  showEffectsPanel: boolean;
  showChromatic: boolean;
  showBloom: boolean;
  showVhsTracking: boolean;
  showHeavyGrain: boolean;
  toggle: (key: EffectKey) => void;
}

export const useEffectsStore = create<EffectsState>((set) => ({
  showCrt: false,
  showDither: false,
  showEffectsPanel: false,
  showChromatic: false,
  showBloom: false,
  showVhsTracking: false,
  showHeavyGrain: false,
  toggle: (key) => set((s) => ({ [key]: !s[key] })),
}));
