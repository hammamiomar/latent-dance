/**
 * Destination Modulation Store
 *
 * Zustand store for destination-based SLERP travel between A/B points
 * in latent and prompt space. Receives DestinationStatus messages at ~2Hz.
 *
 * Two independent modulators:
 *   - Latent space: Controls visual composition (seed-based destinations)
 *   - Prompt space: Controls semantic meaning (prompt-based destinations)
 *
 * Each supports:
 *   - Slider mode: User controls blend via crossfader
 *   - Reactive mode: Audio drives blend position via physics
 */

import { create } from 'zustand';
import type {
  DestinationSpace,
  DestinationSlot,
  DestinationMode,
  DestinationState,
  ReactiveConfig,
  Destination,
  DestinationStatusMessage,
} from '../types/destinations';
import { DEFAULT_REACTIVE_CONFIG } from '../types/destinations';
import type { LinkTarget } from '../types/sae';

interface DestinationStore {
  /** State for latent space modulator */
  latent: DestinationState;

  /** State for prompt space modulator */
  prompt: DestinationState;

  /** Which panel is currently open (null = none) */
  selectedSpace: DestinationSpace | null;

  /** Set a destination for a slot */
  setDestination: (
    space: DestinationSpace,
    slot: DestinationSlot,
    destination: Destination | null
  ) => void;

  /** Set blend position (slider mode) */
  setBlendPosition: (space: DestinationSpace, position: number) => void;

  /** Set modulation mode */
  setMode: (space: DestinationSpace, mode: DestinationMode) => void;

  /** Set reactive config */
  setReactiveConfig: (space: DestinationSpace, config: Partial<ReactiveConfig>) => void;

  /** Set link target for 'linked' mode (also sets mode to 'linked') */
  setLinkTarget: (space: DestinationSpace, linkTarget: LinkTarget) => void;

  /** Set which panel is open */
  setSelectedSpace: (space: DestinationSpace | null) => void;

  /** Update from server status message */
  updateFromStatus: (msg: DestinationStatusMessage) => void;

  /** Reset all state */
  reset: () => void;
}

const createDefaultState = (): DestinationState => ({
  space: 'latent',
  destinationA: null,
  destinationB: null,
  blendPosition: 0.0,
  mode: 'reactive',  // Default to reactive (audio-driven blend)
  reactiveConfig: { ...DEFAULT_REACTIVE_CONFIG },
  linkTarget: null,
});

export const useDestinationStore = create<DestinationStore>((set) => ({
  latent: { ...createDefaultState(), space: 'latent' },
  prompt: { ...createDefaultState(), space: 'prompt' },
  selectedSpace: null,

  setDestination: (space, slot, destination) => {
    set((state) => ({
      [space]: {
        ...state[space],
        [slot === 'a' ? 'destinationA' : 'destinationB']: destination,
      },
    }));
  },

  setBlendPosition: (space, position) => {
    set((state) => ({
      [space]: {
        ...state[space],
        blendPosition: Math.max(0, Math.min(1, position)),
      },
    }));
  },

  setMode: (space, mode) => {
    set((state) => ({
      [space]: {
        ...state[space],
        mode,
      },
    }));
  },

  setReactiveConfig: (space, config) => {
    set((state) => ({
      [space]: {
        ...state[space],
        reactiveConfig: {
          ...state[space].reactiveConfig,
          ...config,
        },
      },
    }));
  },

  setLinkTarget: (space, linkTarget) => {
    set((state) => ({
      [space]: {
        ...state[space],
        linkTarget,
        mode: 'linked', // Automatically switch to linked mode
      },
    }));
  },

  setSelectedSpace: (space) => {
    set({ selectedSpace: space });
  },

  updateFromStatus: (msg) => {
    const space = msg.space;
    set((state) => {
      // In slider mode, user controls blend position - don't overwrite from server
      // In reactive/linked mode, server controls blend position via physics/audio
      // Use LOCAL mode (frontend is authoritative), not server's mode
      const currentMode = state[space].mode;
      const shouldUpdateBlend = currentMode === 'reactive' || currentMode === 'linked';

      return {
        [space]: {
          ...state[space],
          // Update labels from server (keep local destinations, update labels)
          destinationA: msg.destination_a
            ? { ...state[space].destinationA, label: msg.destination_a }
            : state[space].destinationA,
          destinationB: msg.destination_b
            ? { ...state[space].destinationB, label: msg.destination_b }
            : state[space].destinationB,
          // Only update blend position in reactive/linked mode (server-driven)
          blendPosition: shouldUpdateBlend ? msg.blend_position : state[space].blendPosition,
          // Don't update mode from server - frontend is authoritative for mode
          // This prevents race condition where server's stale mode overwrites user's choice
        },
      };
    });
  },

  reset: () => {
    set({
      latent: { ...createDefaultState(), space: 'latent' },
      prompt: { ...createDefaultState(), space: 'prompt' },
      selectedSpace: null,
    });
  },
}));
