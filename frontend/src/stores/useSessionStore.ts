/**
 * Session Store - What the connected backend told us about itself.
 *
 * Everything here arrives over the wire and describes the session, not the
 * user's steering choices: the capability manifest (WS hello today, the
 * bootstrap fetch or the WS hello), track metadata after audio analysis,
 * and the steering mode synced to the backend on play. Slot mappings stay
 * in useSlotStore, which setCapabilities initializes from the manifest.
 */

import { create } from 'zustand';
import { useSlotStore } from './useSlotStore';
import type { BackendCapabilities } from '../types/wire/capabilities';
import type { TrackInfo } from '../types/sae';

/** Steering mode: auto derives per-slot params from the linked stem, manual leaves them alone. */
export type SteeringMode = 'auto' | 'manual';

interface SessionState {
  /** Active backend's manifest; null until the first hello arrives. */
  capabilities: BackendCapabilities | null;

  /** Track metadata from backend (BPM, available stems, duration). */
  trackInfo: TrackInfo | null;

  steeringMode: SteeringMode;

  setCapabilities: (capabilities: BackendCapabilities) => void;
  setTrackInfo: (info: TrackInfo) => void;
  setSteeringMode: (mode: SteeringMode) => void;
}

export const useSessionStore = create<SessionState>()((set) => ({
  capabilities: null,
  trackInfo: null,
  steeringMode: 'auto',

  setCapabilities: (capabilities) => {
    set({ capabilities });
    // The slot vocabulary IS session state: any manifest entering the app
    // (bootstrap fetch or WS hello) materializes the slot store, so nothing
    // can render slots the backend didn't declare.
    useSlotStore.getState().initFromCapabilities(capabilities);
  },
  setTrackInfo: (trackInfo) => set({ trackInfo }),
  setSteeringMode: (steeringMode) => set({ steeringMode }),
}));

// =============================================================================
// ATOMIC SELECTORS (efficient re-renders)
// =============================================================================

export const useCapabilities = () => useSessionStore((s) => s.capabilities);
export const useTrackInfo = () => useSessionStore((s) => s.trackInfo);
export const useSteeringMode = () => useSessionStore((s) => s.steeringMode);
