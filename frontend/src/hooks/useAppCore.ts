/**
 * useAppCore — orchestration for a generation shell (Browser/Desktop).
 *
 * After the transport inversion this hook only owns what genuinely needs a
 * single React home: the canvas ref + frame callback, the physics world,
 * the WS lifecycle, the play-all sequence, and global keyboard/connection
 * effects. All sending lives in lib/wire.ts, all shared state in stores —
 * components subscribe to what they render.
 */

import { useRef, useCallback, useState, useEffect } from "react";
import { loadFeatures, getFeature } from "../data/featureLoader";
import { useSlotStore } from "../stores/useSlotStore";
import { useAudioStore } from "../stores/useAudioStore";
import { useSessionStore, useCapabilities } from "../stores/useSessionStore";
import { useCompositionStore } from "../stores/useCompositionStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import { useConnectionStore } from "../stores/useConnectionStore";
import { usePlayerWindowStore } from "../stores/usePlayerWindowStore";
import { notify } from "../stores/useNotificationStore";
import { useWebSocket } from "./useWebSocket";
import { useMatterPhysics } from "./useMatterPhysics";
import { useAgentBridge } from "./useAgentBridge";
import { useCanvasSamplingManual } from "./useCanvasSampling";
import { ensureCapabilities } from "../lib/bootstrap";
import {
  sendStartSAESteering,
  sendSetSteeringMode,
  sendUpdateSlotConfig,
  sendSetCompositionConfig,
  sendSetDestination,
  sendSetDestinationMode,
  sendSetReactiveConfig,
  sendSetDestinationLink,
  sendAudioPlay,
  sendAudioPause,
} from "../lib/wire";
import type { CanvasHandle } from "../components/Canvas";

/** Poll a condition with timeout (connection/generation handshakes). */
async function waitFor(
  condition: () => boolean,
  timeoutMs: number = 2000,
  pollMs: number = 50,
): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (condition()) return true;
    await new Promise((r) => setTimeout(r, pollMs));
  }
  return false; // Timeout
}

export function useAppCore(dimensions: { width: number; height: number }) {
  const canvasRef = useRef<CanvasHandle>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch the capability manifest (mode + slot vocabulary). The shells show
  // a gate until it lands; connect() below refuses without it.
  useEffect(() => {
    void ensureCapabilities();
  }, []);

  // Feature labels need the manifest's slot names; slots whose label file
  // 404s stay numeric (FeaturePicker degrades to its ID spinner).
  const capabilities = useCapabilities();
  useEffect(() => {
    if (!capabilities) return;
    loadFeatures(capabilities.slots.map((slot) => slot.name)).then(() => {
      const { slots, setSlotFeature } = useSlotStore.getState();
      for (const mapping of Object.values(slots)) {
        const entry = getFeature(mapping.slot, mapping.featureId);
        if (entry) setSlotFeature(mapping.slot, mapping.featureId, entry.label);
      }
    });
  }, [capabilities]);

  const physics = useMatterPhysics({
    width: dimensions.width,
    height: dimensions.height,
    slotCount: capabilities?.slot_count ?? 0,
    containerRef,
  });

  // Frame delivery: render + sample every 3rd frame (~10Hz) for video sync
  const { sampleCanvas } = useCanvasSamplingManual();
  const frameCountRef = useRef(0);
  const handleFrame = useCallback(async (data: ArrayBuffer) => {
    await canvasRef.current?.renderFrame(data);
    frameCountRef.current++;
    if (frameCountRef.current >= 3) {
      frameCountRef.current = 0;
      const canvas = canvasRef.current?.getCanvas();
      if (canvas) sampleCanvas(canvas);
    }
  }, [sampleCanvas]);

  const { connect } = useWebSocket({
    onFrame: handleFrame,
    autoConnect: false,
    enableReconnect: true,
  });

  // Runs for its WS side effects (agent control plane)
  useAgentBridge();

  // Perf overlay (toggle with Shift+D)
  const [showPerfOverlay, setShowPerfOverlay] = useState(false);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.shiftKey && event.key === "D") {
        setShowPerfOverlay((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const handleAudioReady = useCallback(
    (_newAudioId: string) => {
      if (useConnectionStore.getState().status === "disconnected") {
        connect();
      }
    },
    [connect]
  );

  // Unified handler: connect -> start generation -> sync staged state -> play
  const handlePlayAll = useCallback(async () => {
    const audioId = useAudioStore.getState().audioId;
    if (!audioId) return;
    const connection = () => useConnectionStore.getState();

    // 1. Connect if needed (wait for the actual connection)
    if (connection().status !== "connected") {
      connect();
      const connected = await waitFor(() => connection().status === "connected", 2000);
      if (!connected) {
        console.warn("[handlePlayAll] Connection timeout - proceeding anyway");
      }
    }

    // 2. Start generation if not already generating
    if (!connection().isGenerating) {
      sendStartSAESteering(audioId);
      const generating = await waitFor(() => connection().isGenerating, 1000);
      if (!generating) {
        console.warn("[handlePlayAll] Generation start timeout - proceeding anyway");
      }

      // 3. Sync steering mode from frontend to backend
      sendSetSteeringMode(useSessionStore.getState().steeringMode);

      // 4. Sync staged frontend control state to backend. This is what lets the
      // agent create a complete visual while generation is idle.
      const currentSlots = useSlotStore.getState().slots;
      for (const mapping of Object.values(currentSlots)) {
        sendUpdateSlotConfig({
          action: 'update_slot_config',
          slot: mapping.slot,
          link_target: mapping.linkTarget,
          feature_id: mapping.featureId,
          enabled: mapping.enabled,
          auto_config: mapping.autoConfig,
          sae_rank: mapping.saeRank,
          spatial_mode: mapping.spatialMode,
          spatial_mask: mapping.spatialMask,
          strength_min: mapping.strengthRange.strengthMin,
          strength_max: mapping.strengthRange.strengthMax,
          stage_home: mapping.strengthRange.stageHome,
          intensity_source: mapping.intensitySource,
          intensity_curve: mapping.intensityCurve,
          intensity_gamma: mapping.intensityGamma,
        });
      }

      const composition = useCompositionStore.getState();
      sendSetCompositionConfig({
        distance: composition.distance,
        mode: composition.mode,
      });

      // 5. Sync destination state to backend
      const latentDest = useDestinationStore.getState().latent;
      const promptDest = useDestinationStore.getState().prompt;

      if (latentDest.destinationA?.seed !== undefined) {
        sendSetDestination('latent', 'a', 'seed', { seed: latentDest.destinationA.seed });
      }
      if (latentDest.destinationB?.seed !== undefined) {
        sendSetDestination('latent', 'b', 'seed', { seed: latentDest.destinationB.seed });
      }
      if (promptDest.destinationA?.prompt) {
        sendSetDestination('prompt', 'a', 'prompt', { prompt: promptDest.destinationA.prompt });
      }
      if (promptDest.destinationB?.prompt) {
        sendSetDestination('prompt', 'b', 'prompt', { prompt: promptDest.destinationB.prompt });
      }

      sendSetDestinationMode('prompt', promptDest.mode);
      if (promptDest.mode === 'reactive') {
        sendSetReactiveConfig('prompt', promptDest.reactiveConfig);
      }
      if (promptDest.mode === 'linked' && promptDest.linkTarget) {
        sendSetDestinationLink('prompt', promptDest.linkTarget);
      }
    }
  }, [connect]);

  // Global keyboard shortcuts: Escape closes the player, Space toggles playback
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

      switch (e.key) {
        case "Escape":
          if (usePlayerWindowStore.getState().isOpen) {
            usePlayerWindowStore.getState().close();
          }
          break;

        case " ": {
          e.preventDefault();
          const { audioId, isPlaying, play, pause, currentTime } = useAudioStore.getState();
          if (!audioId) break;
          if (isPlaying) {
            pause();
            sendAudioPause();
          } else {
            play();
            sendAudioPlay(currentTime);
          }
          break;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Pause audio + notify when the WS drops
  const status = useConnectionStore((s) => s.status);
  useEffect(() => {
    if (status === "error") {
      const { isPlaying, pause } = useAudioStore.getState();
      if (isPlaying) pause();
      notify.error("Connection error - please try reconnecting");
    } else if (status === "disconnected") {
      const { isPlaying, pause } = useAudioStore.getState();
      if (isPlaying) {
        console.warn("[App] WebSocket disconnected - pausing audio");
        pause();
        notify.warning("Connection lost - audio paused");
      }
    }
  }, [status]);

  return {
    canvasRef,
    containerRef,
    dimensions,
    physics,
    showPerfOverlay,
    handlePlayAll,
    handleAudioReady,
  };
}

export type AppCore = ReturnType<typeof useAppCore>;
