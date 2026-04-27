/**
 * Canvas Lighting Store
 *
 * Zustand store for video canvas color/brightness sampling.
 * Updated by useCanvasSampling hook at 10Hz.
 * Consumed by CrystalHeart and FlowerOrb shaders.
 *
 * Data flow:
 *   Canvas.tsx (ref) → useCanvasSampling → This Store → Shader uniforms
 */

import { create } from 'zustand';

export interface HotSpot {
  x: number; // Normalized 0-1
  y: number; // Normalized 0-1
  intensity: number; // 0-1
}

interface CanvasLightingState {
  /** Dominant color from canvas (RGB, 0-1) */
  dominantColor: [number, number, number];

  /** Overall brightness (0-1) */
  brightness: number;

  /** Two brightest regions (normalized positions) */
  hotSpots: [HotSpot, HotSpot];

  /** Timestamp of last sample */
  lastSampleTime: number;

  /** Whether sampling is active */
  isSampling: boolean;

  /** Update from sampling */
  updateFromSample: (data: {
    dominantColor: [number, number, number];
    brightness: number;
    hotSpots: [HotSpot, HotSpot];
  }) => void;

  /** Start/stop sampling */
  setIsSampling: (value: boolean) => void;

  /** Reset to defaults */
  reset: () => void;
}

// Default neutral values (earthy baseline)
const DEFAULT_STATE = {
  dominantColor: [0.3, 0.28, 0.22] as [number, number, number], // Muted earthy
  brightness: 0.4,
  hotSpots: [
    { x: 0.5, y: 0.5, intensity: 0.3 },
    { x: 0.5, y: 0.5, intensity: 0.2 },
  ] as [HotSpot, HotSpot],
  lastSampleTime: 0,
  isSampling: false,
};

export const useCanvasLightingStore = create<CanvasLightingState>((set) => ({
  ...DEFAULT_STATE,

  updateFromSample: (data) => {
    set({
      dominantColor: data.dominantColor,
      brightness: data.brightness,
      hotSpots: data.hotSpots,
      lastSampleTime: Date.now(),
    });
  },

  setIsSampling: (value) => {
    set({ isSampling: value });
  },

  reset: () => {
    set(DEFAULT_STATE);
  },
}));

/**
 * Selector for dominant color as THREE-compatible array
 */
export function useDominantColor(): [number, number, number] {
  return useCanvasLightingStore((state) => state.dominantColor);
}

/**
 * Selector for brightness
 */
export function useCanvasBrightness(): number {
  return useCanvasLightingStore((state) => state.brightness);
}

/**
 * Selector for hot spots
 */
export function useHotSpots(): [HotSpot, HotSpot] {
  return useCanvasLightingStore((state) => state.hotSpots);
}
