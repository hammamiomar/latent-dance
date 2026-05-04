import { create } from 'zustand';
import type { CompositionMode, CompositionStateSnapshot } from '../types/composition';

interface CompositionStore extends CompositionStateSnapshot {
  setDistance: (distance: number) => void;
  setMode: (mode: CompositionMode) => void;
  setConfig: (config: Partial<CompositionStateSnapshot>) => void;
  reset: () => void;
}

const DEFAULT_COMPOSITION: CompositionStateSnapshot = {
  distance: 1.0,
  mode: 'auto',
};

function clampDistance(distance: number) {
  return Math.max(0, Math.min(4, distance));
}

export const useCompositionStore = create<CompositionStore>((set) => ({
  ...DEFAULT_COMPOSITION,

  setDistance: (distance) => set({ distance: clampDistance(distance) }),

  setMode: (mode) => set({ mode }),

  setConfig: (config) =>
    set((state) => ({
      distance: config.distance == null ? state.distance : clampDistance(config.distance),
      mode: config.mode ?? state.mode,
    })),

  reset: () => set(DEFAULT_COMPOSITION),
}));
