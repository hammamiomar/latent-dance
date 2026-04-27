/**
 * Scene Panel Components
 *
 * CompositionPanel: latent space (seeds + circle-walking)
 * PromptDestinationPanel: prompt space (text prompts + reactive modes)
 */

export { CompositionPanel } from './CompositionPanel';
export { PromptDestinationPanel } from './PromptDestinationPanel';

// Re-export types (BaseDestinationPanelProps still used by PromptDestinationPanel)
export type { BaseDestinationPanelProps } from './types';
export { SPACE_COLORS, PANEL_HEIGHT_SLIDER, PANEL_HEIGHT_REACTIVE } from './types';
