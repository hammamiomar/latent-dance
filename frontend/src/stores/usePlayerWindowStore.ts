/**
 * Player window store — open/minimized state of the audio player.
 *
 * Consumed across three subtrees (CrystalHeart click, AudioPlayerWindow,
 * BellyScene's heart pose) plus the Escape-key handler, so it lives in a
 * store rather than threading through the render tree.
 */

import { create } from "zustand";

interface PlayerWindowState {
  isOpen: boolean;
  isMinimized: boolean;

  /** Heart click: restore if minimized, otherwise open. */
  openFromHeart: () => void;
  close: () => void;
  minimize: () => void;
}

export const usePlayerWindowStore = create<PlayerWindowState>((set) => ({
  isOpen: false,
  isMinimized: false,

  openFromHeart: () =>
    set((s) => (s.isMinimized ? { isMinimized: false } : { isOpen: true })),
  close: () => set({ isOpen: false, isMinimized: false }),
  minimize: () => set({ isMinimized: true }),
}));
