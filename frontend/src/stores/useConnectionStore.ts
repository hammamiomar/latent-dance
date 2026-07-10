/**
 * Connection store — reactive view of the streaming WebSocket state.
 *
 * Written by useWebSocket (lifecycle) and lib/wire.ts (optimistic
 * isGenerating on start/stop). Read by anything that renders connection
 * state or gates on it at call time via getState().
 */

import { create } from "zustand";
import { ConnectionStatus } from "../types";

interface ConnectionState {
  status: ConnectionStatus;
  isGenerating: boolean;

  setStatus: (status: ConnectionStatus) => void;
  setGenerating: (isGenerating: boolean) => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: ConnectionStatus.DISCONNECTED,
  isGenerating: false,

  setStatus: (status) => set({ status }),
  setGenerating: (isGenerating) => set({ isGenerating }),
}));

export const useWsStatus = () => useConnectionStore((s) => s.status);
export const useIsGenerating = () => useConnectionStore((s) => s.isGenerating);
