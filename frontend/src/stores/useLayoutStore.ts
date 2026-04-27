/**
 * Layout Store — Zone rects for the body shader.
 *
 * FaceScreen and BellyScreen write their bounding rects (normalized 0-1)
 * on resize. BodyCanvas reads them in useFrame via .getState() — no
 * React re-renders.
 *
 * Rect format: [left, top, right, bottom] in viewport-normalized coords.
 */

import { create } from "zustand";

type Rect = [number, number, number, number];

interface LayoutState {
  faceRect: Rect;
  bellyRect: Rect;
  setFaceRect: (rect: Rect) => void;
  setBellyRect: (rect: Rect) => void;
}

// Default rects (approximate — overwritten on first resize)
const DEFAULT_FACE: Rect = [0.2, 0.0, 0.8, 0.15];
const DEFAULT_BELLY: Rect = [0.05, 0.2, 0.95, 0.85];

export const useLayoutStore = create<LayoutState>((set) => ({
  faceRect: DEFAULT_FACE,
  bellyRect: DEFAULT_BELLY,
  setFaceRect: (rect) => set({ faceRect: rect }),
  setBellyRect: (rect) => set({ bellyRect: rect }),
}));
