/**
 * useDestinationHandlers - Prompt destination event handlers.
 *
 * Extracts inline arrow callbacks from App.tsx into a reusable hook,
 * matching the pattern of useBlockConfigHandlers.
 *
 * Each handler: update Zustand store + send WebSocket message.
 */

import { useCallback } from 'react';
import { useDestinationStore } from '../stores/useDestinationStore';
import type { DestinationSlot, DestinationSpace, DestinationMode, DestinationType, ReactiveConfig } from '../types/destinations';
import type { LinkTarget } from '../types/sae';

interface DestinationSenders {
  sendSetDestination: (space: DestinationSpace, slot: DestinationSlot, destinationType: DestinationType, value: { seed?: number; prompt?: string }) => void;
  sendFreezeBlend: (space: DestinationSpace, slot: DestinationSlot) => void;
  sendSetBlendPosition: (space: DestinationSpace, position: number) => void;
  sendSetDestinationMode: (space: DestinationSpace, mode: DestinationMode) => void;
  sendSetReactiveConfig: (space: DestinationSpace, config: Partial<ReactiveConfig>) => void;
  sendSetDestinationLink: (space: DestinationSpace, linkTarget: LinkTarget) => void;
}

export function useDestinationHandlers(senders: DestinationSenders) {
  const {
    sendSetDestination,
    sendFreezeBlend,
    sendSetBlendPosition,
    sendSetDestinationMode,
    sendSetReactiveConfig,
    sendSetDestinationLink,
  } = senders;

  const handleSetPrompt = useCallback((slot: DestinationSlot, prompt: string) => {
    sendSetDestination('prompt', slot, 'prompt', { prompt });
    const label = prompt.length > 20 ? prompt.slice(0, 20) + '...' : prompt;
    useDestinationStore.getState().setDestination('prompt', slot, { type: 'prompt', label, prompt });
  }, [sendSetDestination]);

  const handleClearPromptDestination = useCallback((slot: DestinationSlot) => {
    useDestinationStore.getState().setDestination('prompt', slot, null);
  }, []);

  const handlePromptFreezeBlend = useCallback((targetSlot: DestinationSlot) => {
    sendFreezeBlend('prompt', targetSlot);
  }, [sendFreezeBlend]);

  const handlePromptSetBlendPosition = useCallback((pos: number) => {
    sendSetBlendPosition('prompt', pos);
    useDestinationStore.getState().setBlendPosition('prompt', pos);
  }, [sendSetBlendPosition]);

  const handlePromptSetMode = useCallback((mode: DestinationMode) => {
    sendSetDestinationMode('prompt', mode);
    useDestinationStore.getState().setMode('prompt', mode);
  }, [sendSetDestinationMode]);

  const handlePromptSetReactiveConfig = useCallback((config: Partial<ReactiveConfig>) => {
    sendSetReactiveConfig('prompt', config);
    useDestinationStore.getState().setReactiveConfig('prompt', config);
  }, [sendSetReactiveConfig]);

  const handlePromptSetLinkTarget = useCallback((linkTarget: LinkTarget) => {
    sendSetDestinationLink('prompt', linkTarget);
    useDestinationStore.getState().setLinkTarget('prompt', linkTarget);
  }, [sendSetDestinationLink]);

  return {
    handleSetPrompt,
    handleClearPromptDestination,
    handlePromptFreezeBlend,
    handlePromptSetBlendPosition,
    handlePromptSetMode,
    handlePromptSetReactiveConfig,
    handlePromptSetLinkTarget,
  };
}
