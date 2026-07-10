/**
 * BrainApp — "Aurora" studio talkback module.
 *
 * Runs in a separate Electrobun BrowserWindow (transparent, frameless,
 * non-resizable). Connects to the desktop bridge as role=brain.
 *
 * Same machined-metal manufacturer DNA as hamba (engraved labels, LED
 * indicators, 3D pushbuttons) but its own product line: portrait,
 * teal-accented, communication-focused. A piece of post-apocalyptic
 * studio gear with visible screws and a phosphor CRT chat screen.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import "../brain.css";
import { SHELL_BRIDGE_PORT } from "../constants";
import type { AgentEvent, AgentPhase, AgentStateResponse } from "../types/agent";
import {
  AUTO_DANCE_RECENT_HUMAN_STEER_MS,
  AUTO_DANCE_TICK_MS,
  buildAutoDanceDirective,
  sectionInfo,
  sectionsFromState,
  shouldTriggerAutoDance,
  type AutoDanceLastCheckpoint,
} from "../utils/autoDance";

type BrainStatus = "connecting" | "connected" | "error" | "closed";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: number;
}

interface BrainBridgeStatus {
  frontend_connected: boolean;
  mcp_connections: number;
  brain_connections: number;
  hermes_model?: string;
  hermes_endpoint?: string;
}

interface ChatMessage {
  id: string;
  kind: "user" | "hamba" | "action" | "tool" | "error" | "system";
  text: string;
  timestamp: number;
}

const LINE_HEIGHT = 18;
const TEXTAREA_MIN_LINES = 4;
const TEXTAREA_MAX_LINES = 8;
// Box-sizing: border-box means style.height includes border + padding.
// scrollHeight excludes border (includes padding). Add border manually.
const TEXTAREA_PADDING_Y = 20;  // 10px top + 10px bottom
const TEXTAREA_BORDER_Y = 18;   // 9px top + 9px bottom
const PENDING_TRANSCRIPT_TTL_MS = 300_000;
const RECENT_SUMMARY_TTL_MS = 10_000;
const REQUEST_TIMEOUT_MS = 300_000;
const DEFAULT_DIVERGENCE = 0.85;
const AUTO_DANCE_HUMAN_COOLDOWN_MS = 5_000;
const DIVERGENCE_STORAGE_KEY = "hamba.brain.divergence";
const BRAIN_MODE = "directive";

function clamp01(value: number) {
  if (!Number.isFinite(value)) return DEFAULT_DIVERGENCE;
  return Math.max(0, Math.min(1, value));
}

function readStoredDivergence() {
  if (typeof window === "undefined") return DEFAULT_DIVERGENCE;
  const raw = window.localStorage.getItem(DIVERGENCE_STORAGE_KEY);
  return raw == null ? DEFAULT_DIVERGENCE : clamp01(Number(raw));
}

function requestId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shortTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatCompactJson(value: unknown, maxLength = 700) {
  if (value == null) return "";
  try {
    const text = JSON.stringify(value, null, 2);
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
  } catch {
    return String(value);
  }
}

function eventToMessages(
  event: AgentEvent,
  pendingTranscripts: Map<string, number>,
  recentSummaries: Map<string, number>,
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const ts = Date.parse(event.timestamp) || Date.now();
  const now = Date.now();

  for (const [text, expiry] of pendingTranscripts) {
    if (expiry <= now) pendingTranscripts.delete(text);
  }
  for (const [text, expiry] of recentSummaries) {
    if (expiry <= now) recentSummaries.delete(text);
  }

	  if (event.transcript) {
	    const expiry = pendingTranscripts.get(event.transcript);
	    if (expiry && expiry > now) {
	      // Already shown locally or by an earlier phase event. Keep suppressing
	      // repeated phase transcripts until the directive request window expires.
	    } else if (event.phase === "listening" || event.phase === "transcribing") {
	      if (expiry) pendingTranscripts.delete(event.transcript);
	      messages.push({ id: `${event.event_id}-t`, kind: "user", text: event.transcript, timestamp: ts });
	      pendingTranscripts.set(event.transcript, now + PENDING_TRANSCRIPT_TTL_MS);
	    } else if (expiry) {
	      pendingTranscripts.delete(event.transcript);
	    }
	  }

  const hasEntryContext = event.changes?.some((change) => change.action === "agent_entry_context");
  if (event.summary && event.phase !== "off" && (event.phase !== "armed" || hasEntryContext)) {
    const expiry = recentSummaries.get(event.summary);
    if (!expiry || expiry <= now) {
      messages.push({ id: `${event.event_id}-s`, kind: "hamba", text: event.summary, timestamp: ts });
      recentSummaries.set(event.summary, now + RECENT_SUMMARY_TTL_MS);
    }
  }

  if (event.tool) {
    const args = formatCompactJson(event.tool.arguments, 500);
    const result = formatCompactJson(event.tool.result_summary, 700);
    const toolText = event.tool.status === "started"
      ? `${event.tool.name}...`
      : `${event.tool.name} ${event.tool.status}`;
    messages.push({
      id: `${event.event_id}-tool`,
      kind: "tool",
      text: [
        toolText,
        args ? `args: ${args}` : "",
        result ? `result: ${result}` : "",
      ].filter(Boolean).join("\n"),
      timestamp: ts,
    });
  }

  if (event.feature_candidates && event.feature_candidates.length > 0) {
    const candidateText = event.feature_candidates
      .slice(0, 6)
      .map((c) => `${c.block ?? "?"}#${c.id ?? c.feature_id ?? "?"} ${c.label ?? ""}`)
      .join("  ");
    messages.push({ id: `${event.event_id}-fc`, kind: "action", text: `candidates: ${candidateText}`, timestamp: ts });
  }

  if (event.changes && event.changes.length > 0) {
    for (const [index, change] of event.changes.entries()) {
      const after = formatCompactJson(change.after, 700);
      const changeText = [
        `${change.action ?? "change"} ${change.target ?? ""}`.trim(),
        after ? `after: ${after}` : "",
      ].filter(Boolean).join("\n");
      messages.push({
        id: `${event.event_id}-c-${index}`,
        kind: "action",
        text: changeText,
        timestamp: ts,
      });
    }
  }

  if (event.error) {
    messages.push({ id: `${event.event_id}-err`, kind: "error", text: event.error, timestamp: ts });
  }

  return messages;
}

function phaseLabel(phase: AgentPhase): string {
  switch (phase) {
    case "thinking": return "THINKING";
    case "searching_features": return "SEARCHING";
    case "planning": return "PLANNING";
    case "applying": return "APPLYING";
    case "watching": return "WATCHING";
    case "dj_deciding": return "DJ DECIDING";
    case "listening": return "LISTENING";
    case "transcribing": return "TRANSCRIBING";
    case "cooldown": return "COOLDOWN";
    case "error": return "ERROR";
    // "armed" is already conveyed by the ARM LED — no chat line, no phase line
    case "armed": return "";
    default: return "";
  }
}

export function BrainApp() {
  const [status, setStatus] = useState<BrainStatus>("connecting");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [armed, setArmedState] = useState(false);
  const [phase, setPhase] = useState<AgentPhase>("off");
  const [directive, setDirective] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [autoDanceEnabled, setAutoDanceEnabled] = useState(false);
  const [showLogs, setShowLogs] = useState(true);
  const [divergence, setDivergence] = useState(readStoredDivergence);
  const [bridgeStatus, setBridgeStatus] = useState<BrainBridgeStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pendingRef = useRef(new Map<string, PendingRequest>());
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const activeSubmitId = useRef<string | null>(null);
  const latestState = useRef<AgentStateResponse | null>(null);
  const lastAutoDance = useRef<AutoDanceLastCheckpoint>({
    wallTimeMs: Date.now(),
    audioTime: 0,
    sectionIndex: null,
  });
  const lastHumanDirective = useRef<{ text: string; wallTimeMs: number } | null>(null);
  const autoDanceHumanCooldownUntil = useRef(0);
  const seenEventIds = useRef(new Set<string>());
  const userAtBottom = useRef(true);
  const pendingTranscripts = useRef(new Map<string, number>());
  const recentSummaries = useRef(new Map<string, number>());

  // Transparent background for frameless Electrobun window. No border-radius:
  // the brain is industrial, blocky. Hamba is rounded (organic).
  const frontendOnline = bridgeStatus?.frontend_connected ?? false;

  useEffect(() => {
    document.documentElement.style.background = "transparent";
    document.body.style.background = "transparent";
    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";
    const root = document.getElementById("root");
    if (root) {
      root.style.background = "transparent";
      root.style.overflow = "hidden";
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(DIVERGENCE_STORAGE_KEY, String(divergence));
  }, [divergence]);

  // Auto-resize textarea — 4 line minimum, 8 line max, internal scroll beyond.
  // scrollHeight reflects content + padding; we add border to match border-box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const min = LINE_HEIGHT * TEXTAREA_MIN_LINES + TEXTAREA_PADDING_Y + TEXTAREA_BORDER_Y;
    const max = LINE_HEIGHT * TEXTAREA_MAX_LINES + TEXTAREA_PADDING_Y + TEXTAREA_BORDER_Y;
    const target = el.scrollHeight + TEXTAREA_BORDER_Y;
    el.style.height = Math.min(max, Math.max(min, target)) + "px";
  }, [directive]);

  const scrollToBottom = useCallback(() => {
    if (userAtBottom.current) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, []);

  const handleChatScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    userAtBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }, []);

  const appendMessages = useCallback((newMessages: ChatMessage[]) => {
    if (newMessages.length === 0) return;
    setChatMessages((prev) => [...prev, ...newMessages].slice(-500));
    setTimeout(scrollToBottom, 50);
  }, [scrollToBottom]);

  const reportError = useCallback((msg: string) => {
    appendMessages([{
      id: requestId(),
      kind: "error",
      text: msg,
      timestamp: Date.now(),
    }]);
  }, [appendMessages]);

  const reportSystem = useCallback((msg: string) => {
    appendMessages([{
      id: requestId(),
      kind: "system",
      text: msg,
      timestamp: Date.now(),
    }]);
  }, [appendMessages]);

  const request = useCallback((type: string, payload?: unknown) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Brain bridge is not connected"));
    }
    const id = requestId();
    ws.send(JSON.stringify({ id, type, payload }));
    return new Promise<unknown>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        pendingRef.current.delete(id);
        reject(new Error(`Brain request timed out: ${type}`));
      }, REQUEST_TIMEOUT_MS);
      pendingRef.current.set(id, { resolve, reject, timeout });
    });
  }, []);

  const refreshBridgeStatus = useCallback(async () => {
    try {
      const state = await request("brain.get_bridge_status") as BrainBridgeStatus;
      setBridgeStatus(state);
    } catch {
      setBridgeStatus(null);
    }
  }, [request]);

  const refreshState = useCallback(async () => {
    try {
      const state = await request("brain.get_state") as AgentStateResponse;
      latestState.current = state;
      setArmedState(Boolean(state.armed));
      if (state.event_log) {
        const newMessages: ChatMessage[] = [];
        for (const event of state.event_log) {
          if (seenEventIds.current.has(event.event_id)) continue;
          seenEventIds.current.add(event.event_id);
          newMessages.push(...eventToMessages(event, pendingTranscripts.current, recentSummaries.current));
        }
        if (newMessages.length > 0) appendMessages(newMessages);
      }
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    }
    void refreshBridgeStatus();
  }, [appendMessages, refreshBridgeStatus, reportError, request]);

  const readStateQuietly = useCallback(async () => {
    const state = await request("brain.get_state") as AgentStateResponse;
    latestState.current = state;
    setArmedState(Boolean(state.armed));
    return state;
  }, [request]);

  // Poll bridge status every 2s so HAMBA LED stays live as the main frontend
  // connects/disconnects. BRIDGE LED is already driven by our own WS status.
  useEffect(() => {
    const interval = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        void refreshBridgeStatus();
      }
    }, 2000);
    return () => window.clearInterval(interval);
  }, [refreshBridgeStatus]);

  useEffect(() => {
    const ws = new WebSocket(`ws://127.0.0.1:${SHELL_BRIDGE_PORT}/agent/ws?role=brain`);
    const pending = pendingRef.current;
    wsRef.current = ws;
    setStatus("connecting");

    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      setStatus("connected");
      void refreshState();
    };
    ws.onerror = () => {
      if (wsRef.current !== ws) return;
      setStatus("error");
    };
    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      setStatus("closed");
      wsRef.current = null;
      for (const req of pending.values()) {
        window.clearTimeout(req.timeout);
        req.reject(new Error("Brain bridge disconnected"));
      }
      pending.clear();
    };
    ws.onmessage = (messageEvent) => {
      if (wsRef.current !== ws) return;
      let message: Record<string, unknown>;
      try {
        message = JSON.parse(String(messageEvent.data)) as Record<string, unknown>;
      } catch {
        return;
      }

      if (message.type === "brain.agent_event") {
        const event = message.payload as AgentEvent;
        if (!seenEventIds.current.has(event.event_id)) {
          seenEventIds.current.add(event.event_id);
          appendMessages(eventToMessages(event, pendingTranscripts.current, recentSummaries.current));
        }
        setPhase(event.phase);
        setArmedState((current) => event.phase === "armed" ? true : event.phase === "off" ? false : current);
        return;
      }

      if (message.type !== "result" && message.type !== "error") return;
      const id = typeof message.id === "string" ? message.id : "";
      const pending = pendingRef.current.get(id);
      if (!pending) return;
      pendingRef.current.delete(id);
      window.clearTimeout(pending.timeout);
      if (message.type === "error") {
        const bridgeError = message.error as { message?: string } | undefined;
        pending.reject(new Error(bridgeError?.message ?? "Brain request failed"));
      } else {
        pending.resolve(message.payload);
      }
    };

    return () => {
      ws.onopen = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.onmessage = null;
      if (wsRef.current === ws) wsRef.current = null;
      ws.close(1000, "Brain window closed");
      for (const req of pending.values()) {
        window.clearTimeout(req.timeout);
        req.reject(new Error("Brain window closed"));
      }
      pending.clear();
    };
  }, [appendMessages, refreshState]);

  const toggleArmed = useCallback(async () => {
    const nextArmed = !armed;
    try {
      const result = await request("brain.set_armed", { armed: nextArmed, mode: BRAIN_MODE }) as {
        armed?: boolean;
      };
      setArmedState(Boolean(result.armed));
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    }
  }, [armed, reportError, request]);

  const markAutoDancePause = useCallback(() => {
    const state = latestState.current;
    const audioTime = state?.entry_context?.audio.current_time ?? 0;
    const { sectionIndex } = sectionInfo(
      sectionsFromState(state),
      state?.entry_context?.audio.duration ?? null,
      audioTime,
    );
    lastAutoDance.current = {
      wallTimeMs: Date.now(),
      audioTime,
      sectionIndex,
    };
  }, []);

  const markHumanSteerCooldown = useCallback(() => {
    autoDanceHumanCooldownUntil.current = Date.now() + AUTO_DANCE_HUMAN_COOLDOWN_MS;
  }, []);

	  const submitAutoDanceDirective = useCallback(async (text: string) => {
	    const submitId = requestId();
	    activeSubmitId.current = submitId;
	    pendingTranscripts.current.set(text, Date.now() + PENDING_TRANSCRIPT_TTL_MS);
	    setSubmitting(true);
    try {
      await request("brain.submit_directive", {
        directive: text,
        mode: BRAIN_MODE,
        creative: {
          divergence,
        },
      });
      if (autoDanceEnabled) markAutoDancePause();
    } catch (err) {
      if (activeSubmitId.current !== submitId) return;
      const message = err instanceof Error ? err.message : String(err);
      if (/cancel|abort|interrupt/i.test(message)) {
        reportSystem("auto dance redirected");
      } else {
        reportError(message);
      }
    } finally {
	      if (activeSubmitId.current === submitId) {
	        activeSubmitId.current = null;
	        setSubmitting(false);
	      }
	    }
	  }, [autoDanceEnabled, divergence, markAutoDancePause, reportError, reportSystem, request]);

  // Optimistic submit: clear input + show "you:" message immediately, regardless
  // of backend latency or errors. The textarea always clears on Enter.
  const submitDirective = useCallback(async () => {
    const text = directive.trim();
    if (!text) return;
    if (!armed) {
      reportError("Arm Hamba Brain before sending a directive");
      return;
    }
    if (status !== "connected") {
      reportError("Brain bridge is not connected");
      return;
    }

    const wasSubmitting = submitting;
    const submitId = requestId();
    activeSubmitId.current = submitId;

    // Optimistic UI — happens BEFORE the await so it's never blocked
    setDirective("");
    appendMessages([{
      id: `local-${requestId()}`,
      kind: "user",
      text,
      timestamp: Date.now(),
    }]);
    lastHumanDirective.current = { text, wallTimeMs: Date.now() };
	    pendingTranscripts.current.set(text, Date.now() + PENDING_TRANSCRIPT_TTL_MS);
	    if (wasSubmitting) reportSystem("redirecting Hermes to latest directive");
	    if (autoDanceEnabled) markHumanSteerCooldown();

	    setSubmitting(true);
	    try {
	      await request("brain.submit_directive", {
        directive: text,
        mode: BRAIN_MODE,
        creative: {
          divergence,
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (activeSubmitId.current !== submitId) return;
      if (/cancel|abort|interrupt/i.test(message)) {
        reportSystem("directive stopped");
      } else {
        reportError(message);
      }
    } finally {
	      if (activeSubmitId.current === submitId) {
	        activeSubmitId.current = null;
	        setSubmitting(false);
	      }
	      if (autoDanceEnabled) markHumanSteerCooldown();
	    }
	  }, [appendMessages, armed, autoDanceEnabled, directive, divergence, markHumanSteerCooldown, reportError, reportSystem, request, status, submitting]);

  const cancelDirective = useCallback(async () => {
    if (!submitting) return;
    const submitId = activeSubmitId.current;
    try {
      await request("brain.cancel_directive", { reason: "user_stop" });
      reportSystem("stop sent");
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    } finally {
      if (activeSubmitId.current === submitId) {
        activeSubmitId.current = null;
        setSubmitting(false);
      }
    }
  }, [reportError, reportSystem, request, submitting]);

  const toggleAutoDance = useCallback(() => {
    const next = !autoDanceEnabled;
    if (next) {
      markAutoDancePause();
      reportSystem("auto dance armed");
    } else {
      reportSystem("auto dance off");
    }
    setAutoDanceEnabled(next);
  }, [autoDanceEnabled, markAutoDancePause, reportSystem]);

  useEffect(() => {
    if (!autoDanceEnabled) return;
    let canceled = false;
    let checking = false;

    const tick = async () => {
      if (checking || canceled) return;
      checking = true;
      try {
        const now = Date.now();
        if (now < autoDanceHumanCooldownUntil.current) return;
        const state = await readStateQuietly();
        const decision = shouldTriggerAutoDance({
          enabled: autoDanceEnabled,
          bridgeConnected: status === "connected",
          frontendConnected: frontendOnline,
          submitting: submitting || Boolean(activeSubmitId.current),
          state,
          nowWallTimeMs: now,
          last: lastAutoDance.current,
        });

        if (!decision.shouldTrigger || activeSubmitId.current) return;

        lastAutoDance.current = {
          wallTimeMs: now,
          audioTime: decision.audioTime,
          sectionIndex: decision.sectionIndex,
        };
        reportSystem(`auto dance checkpoint: ${decision.reason}`);
        const recentHumanDirective = lastHumanDirective.current;
        const recentHumanDirectiveAgeMs = recentHumanDirective
          ? now - recentHumanDirective.wallTimeMs
          : null;
        await submitAutoDanceDirective(buildAutoDanceDirective(decision, {
          recentHumanDirective:
            recentHumanDirective && recentHumanDirectiveAgeMs != null
            && recentHumanDirectiveAgeMs <= AUTO_DANCE_RECENT_HUMAN_STEER_MS
              ? recentHumanDirective.text
              : null,
          recentHumanDirectiveAgeMs,
        }));
      } catch (err) {
        if (!canceled) reportError(err instanceof Error ? err.message : String(err));
      } finally {
        checking = false;
      }
    };

    const interval = window.setInterval(() => {
      void tick();
    }, AUTO_DANCE_TICK_MS);
    void tick();

    return () => {
      canceled = true;
      window.clearInterval(interval);
    };
  }, [
    autoDanceEnabled,
    frontendOnline,
    readStateQuietly,
    reportError,
    reportSystem,
    status,
    submitting,
    submitAutoDanceDirective,
  ]);

  const canSend = armed && status === "connected" && directive.trim().length > 0;
  const phaseText = phaseLabel(phase);
  const visibleMessages = showLogs
    ? chatMessages
    : chatMessages.filter((msg) => msg.kind === "user" || msg.kind === "hamba");

  return (
    <div className="brain brain-chassis">
      {/* Phillips-head screws at four corners */}
      <div className="brain-screw brain-screw--tl" />
      <div className="brain-screw brain-screw--tr" />
      <div className="brain-screw brain-screw--bl" />
      <div className="brain-screw brain-screw--br" />

      {/* ═══ Title strip — engraved BRAIN, drag region (native FFI) ═══ */}
      <div className="brain-title-strip">
        <span className="brain-title-strip__text">BRAIN</span>
      </div>

      {/* ═══ Header row — status card (left) + ARM hero (right) ═══ */}
      <div className="brain-header-row">
        <div className="brain-status-card">
          <div className="brain-status-card__model">HAMBA BRAIN</div>
          <div className="brain-status-card__sub">NEURAL CTRL · MK I</div>
          <div className="brain-status-card__serial">SERIAL 27000-B</div>
          <div className="brain-status-card__divider" />
          <div className="brain-status-card__leds">
            <div className="brain-status-card__led-group">
              <span className={`brain-led ${
                status === "connected" && frontendOnline ? "brain-led--green" : "brain-led--off"
              }`} />
              <span className="brain-status-card__led-label">HAMBA</span>
            </div>
            <div className="brain-status-card__led-group">
              <span className={`brain-led ${
                status === "connected" ? "brain-led--verdant"
                  : status === "error" ? "brain-led--red"
                  : "brain-led--off"
              }`} />
              <span className="brain-status-card__led-label">BRIDGE</span>
            </div>
          </div>
          {/* Always rendered (reserved-height slot) — no layout shift on phase change */}
          <div className="brain-status-card__phase">
            {phaseText && armed ? `▸ ${phaseText}` : ""}
          </div>
        </div>

        <button
          className={`brain-arm ${armed ? "brain-arm--armed" : ""}`}
          onClick={() => void toggleArmed()}
          title={armed ? "Disarm agent" : "Arm agent"}
        >
          <span className="brain-arm__edge" />
          <span className="brain-arm__face">
            <span className="brain-arm__led" />
            <span className="brain-arm__label">ARM</span>
          </span>
        </button>
      </div>

      <div className="brain-creative-panel">
        <div className="brain-divergence-row">
          <span className="brain-divergence-label">DIVERGENCE</span>
          <input
            className="brain-divergence-slider"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={divergence}
            onChange={(e) => setDivergence(clamp01(Number(e.target.value)))}
            title="Agent creative divergence"
          />
          <span className="brain-divergence-value">{Math.round(divergence * 100)}</span>
          <button
            className={`brain-log-toggle ${showLogs ? "brain-log-toggle--active" : ""}`}
            type="button"
            onClick={() => setShowLogs((current) => !current)}
            title={showLogs ? "Hide tool/action/system logs" : "Show tool/action/system logs"}
          >
            {showLogs ? "LOGS ON" : "LOGS OFF"}
          </button>
        </div>
      </div>

      {/* ═══ Phosphor chat screen ═══ */}
      <div className="brain-chat" onScroll={handleChatScroll}>
        {visibleMessages.length === 0 ? (
          <div className="brain-empty">
            <div className="brain-empty__title">HAMBA BRAIN</div>
            <div className="brain-empty__sub">NEURAL CTRL · MK I</div>
            <div className={`brain-empty__cta ${armed ? "brain-empty__cta--armed" : ""}`}>
              {armed ? "READY · SPEAK OR TYPE" : "ARM TO BEGIN"}
            </div>
          </div>
        ) : (
          visibleMessages.map((msg) => (
            <div key={msg.id} className={`brain-msg brain-msg--${msg.kind}`}>
              {msg.kind === "user" && (
                <>
                  <span className="brain-msg__timestamp">{shortTime(msg.timestamp)}</span>
                  <span className="brain-msg__prefix">you: </span>
                  <span className="brain-msg__text">{msg.text}</span>
                </>
              )}
              {msg.kind === "hamba" && (
                <>
                  <span className="brain-msg__timestamp">{shortTime(msg.timestamp)}</span>
                  <span className="brain-msg__prefix">hamba: </span>
                  <span className="brain-msg__text">{msg.text}</span>
                </>
              )}
              {msg.kind === "action" && (
                <>
                  <span className="brain-msg__prefix">&gt; </span>
                  <span className="brain-msg__text">{msg.text}</span>
                </>
              )}
              {msg.kind === "tool" && (
                <>
                  <span className="brain-msg__prefix">:: </span>
                  <span className="brain-msg__text">{msg.text}</span>
                </>
              )}
              {msg.kind === "error" && (
                <>
                  <span className="brain-msg__prefix">! </span>
                  <span className="brain-msg__text">{msg.text}</span>
                </>
              )}
              {msg.kind === "system" && (
                <span className="brain-msg__text">&mdash; {msg.text}</span>
              )}
            </div>
          ))
        )}
        <div ref={chatEndRef} />
      </div>

      {/* ═══ Input zone — etched DIRECTIVE label + textarea + TRANSMIT ═══ */}
      <div className="brain-input-zone">
        <div className="brain-input-label">
          <span>DIRECTIVE</span>
        </div>
        <div className="brain-input-strip">
          <textarea
            ref={textareaRef}
            className="brain-textarea"
            value={directive}
            rows={1}
            disabled={status !== "connected"}
            onChange={(e) => setDirective(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void submitDirective();
              }
            }}
            placeholder="type a directive, or hold REC to speak..."
          />
          <button
            className="brain-transmit"
            disabled={!canSend}
            onClick={() => void submitDirective()}
            title="Transmit directive (Enter)"
          >
            <span className="brain-transmit__edge" />
            <span className="brain-transmit__face">
              <span className="brain-transmit__glyph">{submitting ? "!" : "▶"}</span>
              <span className="brain-transmit__label">
                {submitting ? <>STE<br />ER</> : <>TRANS<br />MIT</>}
              </span>
            </span>
          </button>
        </div>
      </div>

      {/* ═══ Live action controls ═══ */}
      <div className="brain-controls-row">
        <button
          className={`brain-btn-record ${phase === "listening" || submitting ? "brain-btn-record--active" : ""}`}
          disabled={!submitting}
          onClick={() => void cancelDirective()}
          title={submitting ? "Stop current Hermes request" : "Voice recording is not wired yet"}
        >
          <span className="brain-btn-record__edge" />
          <span className="brain-btn-record__face">
            <span className="brain-btn-record__led" />
            {submitting ? "STOP" : "REC"}
          </span>
        </button>
        <button
          className={`brain-btn-dance ${autoDanceEnabled ? "brain-btn-dance--active" : ""}`}
          disabled={status !== "connected"}
          onClick={toggleAutoDance}
          title={autoDanceEnabled ? "Disable autonomous checkpoint steering" : "Enable autonomous checkpoint steering"}
        >
          <span className="brain-btn-dance__edge" />
          <span className="brain-btn-dance__face">
            {autoDanceEnabled ? "DANCING" : "AUTO DANCE"}
          </span>
        </button>
      </div>
    </div>
  );
}
