import { create } from "zustand";
import type { AgentBridgeStatus } from "../types/agentBridge";
import type { AgentEvent, AgentMode, AgentPhase } from "../types/agent";

const AGENT_EVENT_LIMIT = 200;

interface AgentStore {
  armed: boolean;
  mode: AgentMode;
  latestEvent: AgentEvent | null;
  events: AgentEvent[];
  eventStatus: AgentBridgeStatus;
  error: string | null;
  setArmed: (armed: boolean, mode: AgentMode, event?: Partial<AgentEvent>) => void;
  setPhase: (phase: AgentPhase, event?: Partial<AgentEvent>) => void;
  ingestEvent: (event: AgentEvent) => void;
  clearEvents: () => void;
  setBridgeStatus: (status: AgentBridgeStatus) => void;
  setError: (error: string | null) => void;
}

function makeEvent(mode: AgentMode, phase: AgentPhase, event: Partial<AgentEvent> = {}): AgentEvent {
  const eventId = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    type: "agent_event",
    event_id: eventId,
    timestamp: new Date().toISOString(),
    mode,
    phase,
    ...event,
  };
}

export const useAgentStore = create<AgentStore>((set, get) => ({
  armed: false,
  mode: "off",
  latestEvent: null,
  events: [],
  eventStatus: "idle",
  error: null,

  setArmed: (armed, mode, eventPatch = {}) => {
    const nextMode = armed ? mode : "off";
    const event = makeEvent(
      nextMode,
      armed ? "armed" : "off",
      {
        summary: armed ? "Agent armed locally" : "Agent disarmed",
        ...eventPatch,
      },
    );
    get().ingestEvent(event);
    set({ armed, mode: nextMode, error: null });
  },

  setPhase: (phase, eventPatch = {}) => {
    const mode = get().mode === "off" ? "directive" : get().mode;
    get().ingestEvent(makeEvent(mode, phase, eventPatch));
  },

  ingestEvent: (event) => {
    set((state) => ({
      latestEvent: event,
      events: [...state.events, event].slice(-AGENT_EVENT_LIMIT),
      armed: event.phase === "off" ? false : state.armed,
      mode: event.mode,
      error: event.error ?? null,
    }));
  },

  clearEvents: () => set({ latestEvent: null, events: [], error: null }),

  setBridgeStatus: (eventStatus) => set({ eventStatus }),

  setError: (error) => set({ error }),
}));
