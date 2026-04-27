/**
 * Shared types for destination panels
 */

import type {
  DestinationSlot,
  DestinationMode,
  Destination,
  ReactiveConfig,
} from '../../types/destinations';

/** Panel height based on mode */
export const PANEL_HEIGHT_SLIDER = 320;
export const PANEL_HEIGHT_REACTIVE = 600;  // Increased to fit blend range controls

/** Color schemes for different spaces */
export const SPACE_COLORS = {
  latent: { accent: '#8a6aaa' },
  prompt: { accent: '#aa8a6a' },
} as const;

/** Debounce delays for auto-save */
export const SEED_DEBOUNCE_MS = 400;
export const PROMPT_DEBOUNCE_MS = 600;

/** Base props shared by both panels */
export interface BaseDestinationPanelProps {
  // State
  destinationA: Destination | null;
  destinationB: Destination | null;
  blendPosition: number;
  mode: DestinationMode;
  reactiveConfig: ReactiveConfig;
  // UI
  isOpen: boolean;
  onClose: () => void;
  orbPosition?: { x: number; y: number };
  containerSize?: { width: number; height: number };
  // Common callbacks
  onClearDestination: (slot: DestinationSlot) => void;
  onFreezeBlend: (targetSlot: DestinationSlot) => void;
  onSetBlendPosition: (position: number) => void;
  onSetMode: (mode: DestinationMode) => void;
  onSetReactiveConfig: (config: Partial<ReactiveConfig>) => void;
}
