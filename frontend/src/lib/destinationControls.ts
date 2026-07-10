/**
 * destinationControls — the single write path for prompt destinations.
 *
 * Each function performs the optimistic local store update and sends the
 * matching wire message. Plain module functions (no React), mirroring
 * lib/slotControls.ts.
 */

import { useDestinationStore } from "../stores/useDestinationStore";
import {
  sendClearDestination,
  sendFreezeBlend,
  sendSetBlendPosition,
  sendSetDestination,
  sendSetDestinationLink,
  sendSetDestinationMode,
  sendSetReactiveConfig,
} from "./wire";
import type { DestinationMode, DestinationSlot, ReactiveConfig } from "../types/destinations";
import type { LinkTarget } from "../types/sae";

export function handleSetPrompt(slot: DestinationSlot, prompt: string): void {
  sendSetDestination("prompt", slot, "prompt", { prompt });
  const label = prompt.length > 20 ? prompt.slice(0, 20) + "..." : prompt;
  useDestinationStore.getState().setDestination("prompt", slot, { type: "prompt", label, prompt });
}

export function handleClearPromptDestination(slot: DestinationSlot): void {
  sendClearDestination("prompt", slot);
  useDestinationStore.getState().clearDestination("prompt", slot);
}

export function handlePromptFreezeBlend(targetSlot: DestinationSlot): void {
  sendFreezeBlend("prompt", targetSlot);
}

export function handlePromptSetBlendPosition(position: number): void {
  sendSetBlendPosition("prompt", position);
  useDestinationStore.getState().setBlendPosition("prompt", position);
}

export function handlePromptSetMode(mode: DestinationMode): void {
  sendSetDestinationMode("prompt", mode);
  useDestinationStore.getState().setMode("prompt", mode);
}

export function handlePromptSetReactiveConfig(config: Partial<ReactiveConfig>): void {
  sendSetReactiveConfig("prompt", config);
  useDestinationStore.getState().setReactiveConfig("prompt", config);
}

export function handlePromptSetLinkTarget(linkTarget: LinkTarget): void {
  sendSetDestinationLink("prompt", linkTarget);
  useDestinationStore.getState().setLinkTarget("prompt", linkTarget);
}
