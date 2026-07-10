/**
 * useWebSocket — lifecycle and demux for the streaming connection.
 *
 * Owns the socket (connect, reconnect, teardown) and routes every backend
 * message to its store. Sending lives in lib/wire.ts via lib/transport.ts;
 * this hook attaches the live socket to the transport so those module
 * functions reach it. The only callback left is onFrame — binary frames go
 * straight to the canvas, everything else is store state.
 *
 * The WS endpoint is /ws/stream/{mode}, where mode comes from the
 * capabilities manifest (lib/bootstrap.ts). connect() refuses until the
 * manifest is known — the server would reject a mode mismatch anyway.
 */

import { useEffect, useRef, useCallback } from "react";
import { ConnectionStatus } from "../types";
import { getWsUrl, WS_CONFIG, PERF_CONFIG } from "../constants";
import { attachSocket, detachSocket } from "../lib/transport";
import { clientPerf } from "../lib/clientPerf";
import { usePerfStore } from "../stores/usePerfStore";
import { useSlotStore } from "../stores/useSlotStore";
import { useConnectionStore } from "../stores/useConnectionStore";
import { useSessionStore } from "../stores/useSessionStore";
import { useAudioStore } from "../stores/useAudioStore";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import { parseBackendCapabilities } from "../types/wire/capabilities";
import {
  useSongIntelligenceStore,
  type DecodedSongCurves,
  type SongAnalysis,
  type SongProfile,
} from "../stores/useSongIntelligenceStore";
import type {
  LinkTarget,
  SlotConfigsMessage,
  ExtendedStemActivityMessage,
} from "../types/sae";
import type { DestinationStatusMessage } from "../types/destinations";

interface UseWebSocketOptions {
  onFrame?: (data: ArrayBuffer) => void;
  autoConnect?: boolean;
  enableReconnect?: boolean;
}

interface UseWebSocketReturn {
  connect: () => void;
  disconnect: () => void;
}

export const BINARY_KIND_JPEG_FRAME = 0x01;
export const BINARY_KIND_SONG_CURVES = 0x02;

type DemuxedBinaryPayload =
  | { kind: 'frame'; payload: ArrayBuffer }
  | { kind: 'curves'; curves: DecodedSongCurves }
  | { kind: 'unknown'; kindByte: number };

function exactBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function isJpegPayload(bytes: Uint8Array): boolean {
  return bytes.byteLength >= 2 && bytes[0] === 0xff && bytes[1] === 0xd8;
}

function requireBytes(offset: number, size: number, total: number) {
  if (offset + size > total) {
    throw new Error('Truncated song curve payload');
  }
}

export function decodeSongCurvePayload(buffer: ArrayBuffer): DecodedSongCurves {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const decoder = new TextDecoder();
  const curves: DecodedSongCurves = { lock_index: {}, target_curves: {} };
  let offset = 0;

  requireBytes(offset, 4, bytes.byteLength);
  const curveCount = view.getUint32(offset, true);
  offset += 4;

  for (let i = 0; i < curveCount; i += 1) {
    requireBytes(offset, 4, bytes.byteLength);
    const nameLength = view.getUint32(offset, true);
    offset += 4;

    requireBytes(offset, nameLength, bytes.byteLength);
    const name = decoder.decode(bytes.slice(offset, offset + nameLength));
    offset += nameLength;

    requireBytes(offset, 4, bytes.byteLength);
    const floatCount = view.getUint32(offset, true);
    offset += 4;

    const byteCount = floatCount * Float32Array.BYTES_PER_ELEMENT;
    requireBytes(offset, byteCount, bytes.byteLength);
    const values = new Float32Array(bytes.slice(offset, offset + byteCount).buffer);
    offset += byteCount;

    if (name === 'timestamps') {
      curves.timestamps = values;
    } else if (name === 'tension') {
      curves.tension = values;
    } else if (name === 'tonal_distance') {
      curves.tonal_distance = values;
    } else if (name === 'novelty_long') {
      curves.novelty_long = values;
    } else if (name.startsWith('lock_index:')) {
      curves.lock_index[name.slice('lock_index:'.length)] = values;
    } else if (name.startsWith('target:')) {
      const [, target, channel] = name.split(':', 3);
      if (target && channel) {
        const linkTarget = target as LinkTarget;
        curves.target_curves[linkTarget] = {
          ...(curves.target_curves[linkTarget] ?? {}),
          [channel]: values,
        };
      }
    }
  }

  return curves;
}

export function demuxBinaryPayload(buffer: ArrayBuffer): DemuxedBinaryPayload {
  const bytes = new Uint8Array(buffer);
  const kindByte = bytes[0] ?? -1;
  if (isJpegPayload(bytes)) {
    return { kind: 'frame', payload: exactBuffer(bytes) };
  }
  if (kindByte === BINARY_KIND_JPEG_FRAME) {
    return { kind: 'frame', payload: exactBuffer(bytes.slice(1)) };
  }
  if (kindByte === BINARY_KIND_SONG_CURVES) {
    return { kind: 'curves', curves: decodeSongCurvePayload(exactBuffer(bytes.slice(1))) };
  }
  return { kind: 'unknown', kindByte };
}

/** Backend perf telemetry (~2Hz); mirrors PerfStatsMessage in app/schemas.py. */
interface PerfStatsMessage {
  gen_fps: number;
  queue_depth: number;
  encode_busy: boolean;
  encode_ms: number;
  pending_age_ms: number;
  avg_steer_ms: number;
  avg_infer_ms: number;
  avg_d2h_ms: number;
  avg_total_ms: number;
  delivery_p50_ms: number;
  delivery_p95_ms: number;
  jitter_mean_ms: number;
  jitter_p95_ms: number;
  drop_rate: number;
  measured_fps: number;
  lookahead_ms: number;
}

/** Route one parsed JSON message to its store. */
function routeJsonMessage(msg: Record<string, unknown> & { type?: string }) {
  switch (msg.type) {
    case "capabilities":
      // First message on every connection: the backend's control-input
      // manifest. Validated at the boundary so a contract drift fails
      // here with a field path, not as undefined reads in render code.
      try {
        useSessionStore.getState().setCapabilities(
          parseBackendCapabilities(msg.capabilities),
        );
      } catch (error) {
        console.error("[useWebSocket] Rejected capabilities manifest:", error);
      }
      break;
    case "perf_stats": {
      const stats = msg as unknown as PerfStatsMessage;
      usePerfStore.getState().setStats({
        genFps: stats.gen_fps,
        queueDepth: stats.queue_depth,
        encodeBusy: stats.encode_busy,
        encodeMs: stats.encode_ms,
        pendingAgeMs: stats.pending_age_ms,
        avgSteerMs: stats.avg_steer_ms,
        avgInferMs: stats.avg_infer_ms,
        avgD2hMs: stats.avg_d2h_ms,
        avgTotalMs: stats.avg_total_ms,
        deliveryP50Ms: stats.delivery_p50_ms,
        deliveryP95Ms: stats.delivery_p95_ms,
        jitterMeanMs: stats.jitter_mean_ms,
        jitterP95Ms: stats.jitter_p95_ms,
        dropRate: stats.drop_rate,
        measuredFps: stats.measured_fps,
        lookaheadMs: stats.lookahead_ms,
        lastUpdated: Date.now(),
      });
      break;
    }
    case "slot_configs":
      useSlotStore.getState().applyConfigSnapshots((msg as unknown as SlotConfigsMessage).configs);
      break;
    case "error":
      console.error("[WS] Backend error:", msg.message);
      break;
    case "extended_activity": {
      const activity = msg as unknown as ExtendedStemActivityMessage;
      useAudioActivityStore.getState().updateFromMessage(activity);

      // Drift detection: compare backend time vs frontend time
      const frontendTime = useAudioStore.getState().currentTime;
      const drift = Math.abs(frontendTime - activity.audio_time);
      clientPerf.driftSec = drift;
      if (drift > 0.5) {
        console.warn(
          `[Sync] Drift detected: ${drift.toFixed(2)}s (frontend=${frontendTime.toFixed(2)}, backend=${activity.audio_time.toFixed(2)})`
        );
      }
      break;
    }
    case "track_info":
      useSessionStore.getState().setTrackInfo({
        type: "track_info",
        audio_id: msg.audio_id as string,
        duration: msg.duration as number,
        bpm: msg.bpm as number,
        stems: msg.stems as string[],
      });
      break;
    case "song_intelligence":
      if (typeof msg.audio_id === "string" && msg.profile) {
        useSongIntelligenceStore.getState().setProfile(
          msg.audio_id,
          msg.profile as SongProfile,
          Array.isArray(msg.sections) ? msg.sections : [],
          msg.analysis ? msg.analysis as SongAnalysis : null,
        );
      }
      break;
    case "destination_status":
      useDestinationStore.getState().updateFromStatus(
        msg as unknown as DestinationStatusMessage,
      );
      break;
  }
}

export function useWebSocket({
  onFrame,
  autoConnect = false,
  enableReconnect = true,
}: UseWebSocketOptions): UseWebSocketReturn {
  const ws = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Reconnection — use ref to avoid stale closure
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Callback ref (avoids reconnections on parent re-render)
  const onFrameRef = useRef(onFrame);
  useEffect(() => {
    onFrameRef.current = onFrame;
  });

  const connect = useCallback(() => {
    if (
      ws.current?.readyState === WebSocket.OPEN ||
      ws.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    const mode = useSessionStore.getState().capabilities?.mode;
    if (!mode) {
      console.warn("[useWebSocket] No capabilities manifest yet — cannot pick a stream mode");
      return;
    }

    const connection = useConnectionStore.getState();
    connection.setStatus(ConnectionStatus.CONNECTING);
    const socket = new WebSocket(getWsUrl(mode));
    socket.binaryType = "arraybuffer";
    ws.current = socket;
    attachSocket(socket);

    socket.onopen = () => {
      if (ws.current !== socket) return;
      useConnectionStore.getState().setStatus(ConnectionStatus.CONNECTED);
      reconnectAttemptsRef.current = 0;
    };

    socket.onmessage = (event: MessageEvent) => {
      if (ws.current !== socket) return;
      // Binary frame
      if (event.data instanceof ArrayBuffer) {
        let demuxed: DemuxedBinaryPayload;
        try {
          demuxed = demuxBinaryPayload(event.data);
        } catch (error) {
          console.warn("[useWebSocket] Failed to decode binary payload:", error);
          return;
        }

        if (demuxed.kind === 'curves') {
          useSongIntelligenceStore.getState().setCurves(demuxed.curves);
          return;
        }

        if (demuxed.kind === 'unknown') {
          console.warn("[useWebSocket] Unknown binary payload kind:", demuxed.kindByte);
          return;
        }

        onFrameRef.current?.(demuxed.payload);

        frameCountRef.current++;
        const now = Date.now();
        const elapsed = now - lastFpsUpdateRef.current;
        if (elapsed >= PERF_CONFIG.FPS_UPDATE_INTERVAL) {
          clientPerf.wsFps = frameCountRef.current / (elapsed / 1000);
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }
        return;
      }

      // JSON messages
      if (typeof event.data !== "string") return;
      try {
        routeJsonMessage(JSON.parse(event.data));
      } catch (e) {
        console.warn("[useWebSocket] Failed to parse JSON message:", e);
      }
    };

    socket.onerror = () => {
      if (ws.current !== socket) return;
      useConnectionStore.getState().setStatus(ConnectionStatus.ERROR);
    };

    socket.onclose = (event) => {
      if (ws.current !== socket) return;
      ws.current = null;
      detachSocket(socket);
      const state = useConnectionStore.getState();
      state.setStatus(ConnectionStatus.DISCONNECTED);
      state.setGenerating(false); // Reset so handlePlayAll re-sends start on reconnect
      clientPerf.wsFps = 0;

      // Reconnect on abnormal closure (uses ref to avoid stale closure)
      if (
        enableReconnect &&
        event.code !== 1000 &&
        reconnectAttemptsRef.current < WS_CONFIG.MAX_RECONNECT_ATTEMPTS
      ) {
        reconnectAttemptsRef.current += 1;
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, WS_CONFIG.RECONNECT_DELAY);
      }
    };
  }, [enableReconnect]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (ws.current) {
      const socket = ws.current;
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      socket.close(1000, "Client requested disconnect");
      if (ws.current === socket) ws.current = null;
      detachSocket(socket);
      const state = useConnectionStore.getState();
      state.setStatus(ConnectionStatus.DISCONNECTED);
      state.setGenerating(false);
      clientPerf.wsFps = 0;
      reconnectAttemptsRef.current = 0;
    }
  }, []);

  // Auto-connect
  useEffect(() => {
    if (autoConnect) connect();
    return () => { disconnect(); };
  }, [autoConnect, connect, disconnect]);

  return { connect, disconnect };
}
