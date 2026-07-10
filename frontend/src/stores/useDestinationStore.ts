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

  /** Clear a destination slot, promoting B to A when clearing A */
  clearDestination: (space: DestinationSpace, slot: DestinationSlot) => void;

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

  clearDestination: (space, slot) => {
    set((state) => {
      const current = state[space];
      if (slot === 'a') {
        return {
          [space]: {
            ...current,
            destinationA: current.destinationB,
            destinationB: null,
            blendPosition: 0,
          },
        };
      }

      return {
        [space]: {
          ...current,
          destinationB: null,
          blendPosition: 0,
        },
      };
    });
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
      const current = state[space];
      // In slider mode, user controls blend position - don't overwrite from server
      // In reactive/linked mode, server controls blend position via physics/audio
      // Use LOCAL mode (frontend is authoritative), not server's mode
      const shouldUpdateBlend = current.mode === 'reactive' || current.mode === 'linked';

      // Update labels from server (keep local destinations, update labels);
      // keep the same object when the label already matches
      const destinationA =
        msg.destination_a && current.destinationA?.label !== msg.destination_a
          ? { ...current.destinationA, label: msg.destination_a }
          : current.destinationA;
      const destinationB =
        msg.destination_b && current.destinationB?.label !== msg.destination_b
          ? { ...current.destinationB, label: msg.destination_b }
          : current.destinationB;
      const blendPosition = shouldUpdateBlend ? msg.blend_position : current.blendPosition;

      // Most ~2Hz status echoes change nothing (paused, slider mode, same
      // labels) — return the same state so no subscriber re-renders.
      if (
        destinationA === current.destinationA &&
        destinationB === current.destinationB &&
        blendPosition === current.blendPosition
      ) {
        return state;
      }

      // Mode is deliberately NOT taken from the server - frontend is
      // authoritative, preventing a stale echo from overwriting user's choice
      return {
        [space]: { ...current, destinationA, destinationB, blendPosition },
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
