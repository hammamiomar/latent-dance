/**
 * Perf overlay store (backend telemetry + client drift)
 */
import { create } from "zustand";

export interface PerfStats {
  genFps?: number;
  queueDepth?: number;
  encodeBusy?: boolean;
  encodeMs?: number;
  pendingAgeMs?: number;
  avgSteerMs?: number;
  avgInferMs?: number;
  avgD2hMs?: number;
  avgTotalMs?: number;
  lastUpdated?: number;
  // SLO delivery metrics
  deliveryP50Ms?: number;
  deliveryP95Ms?: number;
  jitterMeanMs?: number;
  jitterP95Ms?: number;
  dropRate?: number;
  measuredFps?: number;
  lookaheadMs?: number;
}

interface PerfState {
  stats: PerfStats;
  setStats: (stats: PerfStats) => void;
}

export const usePerfStore = create<PerfState>((set) => ({
  stats: {},
  setStats: (stats) => set({ stats }),
}));
