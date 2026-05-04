import { useEffect, useRef, useCallback, useState } from "react";
import { ConnectionStatus } from "../types";
import { WS_CONFIG, PERF_CONFIG } from "../constants";
import { usePerfStore } from "../stores/usePerfStore";
import { blockActions } from "../stores/useBlockStore";
import {
  songIntelligenceActions,
  type DecodedSongCurves,
  type SongAnalysis,
  type SongProfile,
} from "../stores/useSongIntelligenceStore";
import type {
  LinkTarget,
  TrackInfo,
  ExtendedStemActivityMessage,
  UpdateBlockConfigMessage,
  BlockConfigsMessage,
} from "../types/sae";
import type {
  DestinationSpace,
  DestinationSlot,
  DestinationType,
  DestinationMode,
  ReactiveConfig,
  DestinationStatusMessage,
} from "../types/destinations";

interface UseWebSocketOptions {
  url: string;
  onFrame?: (data: ArrayBuffer) => void;
  onExtendedActivity?: (data: ExtendedStemActivityMessage) => void;
  onTrackInfo?: (info: TrackInfo) => void;
  onDestinationStatus?: (status: DestinationStatusMessage) => void;
  autoConnect?: boolean;
  enableReconnect?: boolean;
}

interface UseWebSocketReturn {
  connect: () => void;
  disconnect: () => void;
  sendStopGeneration: () => void;

  // SAE steering
  sendStartSAESteering: (audioId: string) => void;
  sendUpdateBlockConfig: (message: UpdateBlockConfigMessage) => void;

  // Audio sync
  sendAudioTimeUpdate: (time: number) => void;
  sendAudioPlay: (time: number) => void;
  sendAudioPause: () => void;
  sendAudioSeek: (time: number) => void;

  // Steering mode
  sendSetSteeringMode: (mode: 'auto' | 'manual') => void;

  // Destination modulation
  sendSetDestination: (
    space: DestinationSpace,
    slot: DestinationSlot,
    destinationType: DestinationType,
    value: { seed?: number; prompt?: string },
    replaceMode?: 'direct' | 'from_blend'
  ) => void;
  sendClearDestination: (space: DestinationSpace, slot: DestinationSlot) => void;
  sendFreezeBlend: (space: DestinationSpace, targetSlot: DestinationSlot) => void;
  sendSetBlendPosition: (space: DestinationSpace, position: number) => void;
  sendSetDestinationMode: (space: DestinationSpace, mode: DestinationMode) => void;
  sendSetReactiveConfig: (space: DestinationSpace, config: Partial<ReactiveConfig>) => void;
  sendSetDestinationLink: (space: DestinationSpace, linkTarget: LinkTarget) => void;

  // Composition engine
  sendSetCompositionConfig: (config: { distance?: number; mode?: string }) => void;

  // State
  status: ConnectionStatus;
  fpsRef: React.RefObject<number>;
  isGenerating: boolean;
  reconnectAttempts: number;
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

export function useWebSocket({
  url,
  onFrame,
  onExtendedActivity,
  onTrackInfo,
  onDestinationStatus,
  autoConnect = false,
  enableReconnect = true,
}: UseWebSocketOptions): UseWebSocketReturn {
  const ws = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [isGenerating, setIsGenerating] = useState(false);
  const fpsRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Reconnection — use ref to avoid stale closure
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Callback refs (avoids reconnections on parent re-render)
  const onFrameRef = useRef(onFrame);
  const onExtendedActivityRef = useRef(onExtendedActivity);
  const onTrackInfoRef = useRef(onTrackInfo);
  const onDestinationStatusRef = useRef(onDestinationStatus);
  const setPerfStats = usePerfStore((s) => s.setStats);
  useEffect(() => {
    onFrameRef.current = onFrame;
    onExtendedActivityRef.current = onExtendedActivity;
    onTrackInfoRef.current = onTrackInfo;
    onDestinationStatusRef.current = onDestinationStatus;
  });

  // Helper: send JSON if connected
  const send = useCallback((payload: Record<string, unknown>) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(payload));
    }
  }, []);

  const connect = useCallback(() => {
    if (
      ws.current?.readyState === WebSocket.OPEN ||
      ws.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    setStatus(ConnectionStatus.CONNECTING);
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    ws.current = socket;

    socket.onopen = () => {
      if (ws.current !== socket) return;
      setStatus(ConnectionStatus.CONNECTED);
      reconnectAttemptsRef.current = 0;
      setReconnectAttempts(0);
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
          songIntelligenceActions.setCurves(demuxed.curves);
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
          fpsRef.current = frameCountRef.current / (elapsed / 1000);
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }
        return;
      }

      // JSON messages
      if (typeof event.data !== "string") return;
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "perf_stats":
            setPerfStats({
              genFps: msg.gen_fps,
              queueDepth: msg.queue_depth,
              encodeBusy: msg.encode_busy,
              encodeMs: msg.encode_ms,
              pendingAgeMs: msg.pending_age_ms,
              avgSteerMs: msg.avg_steer_ms,
              avgInferMs: msg.avg_infer_ms,
              avgD2hMs: msg.avg_d2h_ms,
              avgTotalMs: msg.avg_total_ms,
              deliveryP50Ms: msg.delivery_p50_ms,
              deliveryP95Ms: msg.delivery_p95_ms,
              jitterMeanMs: msg.jitter_mean_ms,
              jitterP95Ms: msg.jitter_p95_ms,
              dropRate: msg.drop_rate,
              measuredFps: msg.measured_fps,
              lookaheadMs: msg.lookahead_ms,
              lastUpdated: Date.now(),
            });
            break;
          case "block_configs":
            blockActions.applyBlockConfigs((msg as BlockConfigsMessage).configs);
            break;
          case "error":
            console.error("[WS] Backend error:", msg.message);
            break;
          case "extended_activity":
            onExtendedActivityRef.current?.({
              type: "extended_activity",
              audio_time: msg.audio_time,
              stems: msg.stems,
              prominence: msg.prominence,
              blocks: msg.blocks,
            });
            break;
          case "track_info":
            onTrackInfoRef.current?.({
              type: "track_info",
              audio_id: msg.audio_id,
              duration: msg.duration,
              bpm: msg.bpm,
              stems: msg.stems,
            });
            break;
          case "song_intelligence":
            if (typeof msg.audio_id === "string" && msg.profile) {
              songIntelligenceActions.setProfile(
                msg.audio_id,
                msg.profile as SongProfile,
                Array.isArray(msg.sections) ? msg.sections : [],
                msg.analysis ? msg.analysis as SongAnalysis : null,
              );
            }
            break;
          case "destination_status":
            onDestinationStatusRef.current?.({
              type: "destination_status",
              space: msg.space,
              destination_a: msg.destination_a,
              destination_b: msg.destination_b,
              blend_position: msg.blend_position,
              mode: msg.mode,
            });
            break;
        }
      } catch (e) {
        console.warn("[useWebSocket] Failed to parse JSON message:", e);
      }
    };

    socket.onerror = () => {
      if (ws.current !== socket) return;
      setStatus(ConnectionStatus.ERROR);
    };

    socket.onclose = (event) => {
      if (ws.current !== socket) return;
      ws.current = null;
      setStatus(ConnectionStatus.DISCONNECTED);
      fpsRef.current = 0;
      setIsGenerating(false); // Reset so handlePlayAll re-sends start on reconnect

      // Reconnect on abnormal closure (uses ref to avoid stale closure)
      if (
        enableReconnect &&
        event.code !== 1000 &&
        reconnectAttemptsRef.current < WS_CONFIG.MAX_RECONNECT_ATTEMPTS
      ) {
        reconnectAttemptsRef.current += 1;
        setReconnectAttempts(reconnectAttemptsRef.current);
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, WS_CONFIG.RECONNECT_DELAY);
      }
    };
  }, [url, enableReconnect, setPerfStats]); // reconnectAttempts uses a ref to avoid reconnect loops

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
      setStatus(ConnectionStatus.DISCONNECTED);
      fpsRef.current = 0;
      setIsGenerating(false);
      reconnectAttemptsRef.current = 0;
      setReconnectAttempts(0);
    }
  }, []);

  // === Send methods ===

  const sendStopGeneration = useCallback(() => {
    send({ action: "stop_generation" });
    setIsGenerating(false);
  }, [send]);

  const sendStartSAESteering = useCallback(
    (audioId: string) => {
      songIntelligenceActions.clear();
      send({ action: "start_sae_steering", audio_id: audioId });
      setIsGenerating(true);
    },
    [send],
  );

  const sendUpdateBlockConfig = useCallback(
    (message: UpdateBlockConfigMessage) => { send(message as unknown as Record<string, unknown>); },
    [send],
  );

  const sendAudioTimeUpdate = useCallback(
    (time: number) => { send({ action: "audio_timeupdate", time }); },
    [send],
  );

  const sendAudioPlay = useCallback(
    (time: number) => { send({ action: "audio_play", time }); },
    [send],
  );

  const sendAudioPause = useCallback(
    () => { send({ action: "audio_pause" }); },
    [send],
  );

  const sendAudioSeek = useCallback(
    (time: number) => { send({ action: "audio_seek", time }); },
    [send],
  );

  const sendSetSteeringMode = useCallback(
    (mode: 'auto' | 'manual') => { send({ action: "set_steering_mode", mode }); },
    [send],
  );

  const sendSetDestination = useCallback(
    (
      space: DestinationSpace,
      slot: DestinationSlot,
      destinationType: DestinationType,
      value: { seed?: number; prompt?: string },
      replaceMode: 'direct' | 'from_blend' = 'direct'
    ) => {
      send({
        action: "set_destination",
        space,
        slot,
        destination_type: destinationType,
        seed: value.seed,
        prompt: value.prompt,
        replace_mode: replaceMode,
      });
    },
    [send],
  );

  const sendClearDestination = useCallback(
    (space: DestinationSpace, slot: DestinationSlot) => {
      send({ action: "clear_destination", space, slot });
    },
    [send],
  );

  const sendFreezeBlend = useCallback(
    (space: DestinationSpace, targetSlot: DestinationSlot) => {
      send({ action: "freeze_blend", space, target_slot: targetSlot });
    },
    [send],
  );

  const sendSetBlendPosition = useCallback(
    (space: DestinationSpace, position: number) => {
      send({ action: "set_blend_position", space, position });
    },
    [send],
  );

  const sendSetDestinationMode = useCallback(
    (space: DestinationSpace, mode: DestinationMode) => {
      if (mode === 'linked') {
        console.warn('Use sendSetDestinationLink() for linked mode');
        return;
      }
      send({ action: "set_destination_mode", space, mode });
    },
    [send],
  );

  const sendSetReactiveConfig = useCallback(
    (space: DestinationSpace, config: Partial<ReactiveConfig>) => {
      send({
        action: "set_reactive_config",
        space,
        stage_left: config.stageLeft,
        stage_home: config.stageHome,
        stage_right: config.stageRight,
        position_source: config.positionSource,
        intensity_source: config.intensitySource,
        position_smoothing_ms: config.positionSmoothingMs,
        silence_behavior: config.silenceBehavior,
        drift_ms: config.driftMs,
        intensity_curve: config.intensityCurve,
        intensity_gamma: config.intensityGamma,
        stem_rankings: config.stemRankings,
        rank_weights: config.rankWeights,
        blend_slew_rate: config.blendSlewRate,
      });
    },
    [send],
  );

  const sendSetDestinationLink = useCallback(
    (space: DestinationSpace, linkTarget: LinkTarget) => {
      send({ action: "set_destination_link", space, link_target: linkTarget });
    },
    [send],
  );

  const sendSetCompositionConfig = useCallback(
    (config: { distance?: number; mode?: string }) => {
      send({ action: "set_composition_config", ...config });
    },
    [send],
  );

  // Auto-connect
  useEffect(() => {
    if (autoConnect) connect();
    return () => { disconnect(); };
  }, [autoConnect, connect, disconnect]);

  return {
    connect,
    disconnect,
    sendStopGeneration,
    sendStartSAESteering,
    sendUpdateBlockConfig,
    sendAudioTimeUpdate,
    sendAudioPlay,
    sendAudioPause,
    sendAudioSeek,
    sendSetSteeringMode,
    sendSetDestination,
    sendClearDestination,
    sendFreezeBlend,
    sendSetBlendPosition,
    sendSetDestinationMode,
    sendSetReactiveConfig,
    sendSetDestinationLink,
    sendSetCompositionConfig,
    status,
    fpsRef,
    isGenerating,
    reconnectAttempts,
  };
}
