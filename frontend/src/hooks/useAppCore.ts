/**
 * useAppCore - All application state, hooks, callbacks, and effects.
 *
 * Extracted from App.tsx to make the visualizer embeddable.
 * Dimensions are a parameter (caller decides: window vs. container).
 */

import { useRef, useCallback, useMemo, useState, useEffect } from "react";
import { loadFeatures, getFeature } from "../data/featureLoader";
import { useBlockStore } from "../stores/useBlockStore";
import type { BlockCode } from "../types/sae";
import type { CanvasHandle } from "../components/Canvas";
import { useMatterPhysics } from "./useMatterPhysics";
import { useBlockConfigHandlers } from "./useBlockConfigHandlers";
import { useAudioStore } from "../stores/useAudioStore";
import { useWebSocket } from "./useWebSocket";
import { notify } from "../stores/useNotificationStore";
import { WS_CONFIG } from "../constants";
import {
  useStemActivity,
  useBlockMappings,
  useTrackInfo,
  useSteeringMode,
  blockActions,
} from "../stores/useBlockStore";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import { useCanvasSamplingManual } from "./useCanvasSampling";
import { useDestinationStore } from "../stores/useDestinationStore";
import { useCompositionStore } from "../stores/useCompositionStore";
import { useShallow } from "zustand/shallow";
import { usePerfStore } from "../stores/usePerfStore";
import type {
  DestinationSpace,
  DestinationStatusMessage,
} from "../types/destinations";
import { useDestinationHandlers } from "./useDestinationHandlers";
import { useAgentBridge } from "./useAgentBridge";
import type { ExtendedStemActivityMessage } from "../types/sae";

// ============================================================================
// useAppCore
// ============================================================================

export function useAppCore(dimensions: { width: number; height: number }) {
  // ========================================
  // Refs & State
  // ========================================

  const canvasRef = useRef<CanvasHandle>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load SAE features on mount
  useEffect(() => {
    loadFeatures().then(() => {
      const { blockMappings, setBlockFeature } = useBlockStore.getState();
      for (const [block, m] of Object.entries(blockMappings) as [BlockCode, typeof blockMappings[BlockCode]][]) {
        const entry = getFeature(block, m.featureId);
        if (entry) setBlockFeature(block, m.featureId, entry.label);
      }
    });
  }, []);

  // UI state
  const [isPlayerOpen, setIsPlayerOpen] = useState(false);
  const [isPlayerMinimized, setIsPlayerMinimized] = useState(false);

  // Audio store
  const audioId = useAudioStore((s) => s.audioId);

  // Block store
  const stemActivity = useStemActivity();
  const blockMappings = useBlockMappings();
  const trackInfo = useTrackInfo();
  const stemProminence = useAudioActivityStore((s) => s.prominence);

  // Steering mode (for syncing to backend on connect)
  const steeringMode = useSteeringMode();

  // Canvas sampling for video sync
  const { sampleCanvas } = useCanvasSamplingManual();

  // Destination modulation store (shallow compare to avoid re-render on blendPosition updates)
  const latentDestinations = useDestinationStore(useShallow((s) => s.latent));
  const promptDestinations = useDestinationStore(useShallow((s) => s.prompt));
  const selectedSpace = useDestinationStore((s) => s.selectedSpace);

  // Perf overlay (toggle with Shift+D)
  const [showPerfOverlay, setShowPerfOverlay] = useState(false);
  const perfStats = usePerfStore((s) => s.stats);
  const driftRef = useRef(0);

  // Frame counter for throttled sampling (every 3rd frame ~ 10Hz at 30fps)
  const frameCountRef = useRef(0);

  // ========================================
  // Physics World
  // ========================================

  const physics = useMatterPhysics({
    width: dimensions.width,
    height: dimensions.height,
    containerRef,
  });

  // ========================================
  // WebSocket Callbacks
  // ========================================

  const handleFrame = useCallback(async (data: ArrayBuffer) => {
    await canvasRef.current?.renderFrame(data);

    // Sample canvas for video sync (throttled to ~10Hz)
    frameCountRef.current++;
    if (frameCountRef.current >= 3) {
      frameCountRef.current = 0;
      const canvas = canvasRef.current?.getCanvas();
      if (canvas) {
        sampleCanvas(canvas);
      }
    }
  }, [sampleCanvas]);

  // Shift+D toggles perf overlay
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.shiftKey && event.key === "D") {
        setShowPerfOverlay((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Handle TrackInfo from backend
  const handleTrackInfo = useCallback(
    (info: { type: string; audio_id: string; duration: number; bpm: number; stems: string[] }) => {
      console.log("[App] TrackInfo received:", info);
      blockActions.setTrackInfo({
        type: "track_info",
        audio_id: info.audio_id,
        duration: info.duration,
        bpm: info.bpm,
        stems: info.stems,
      });
    },
    []
  );

  // Handle ExtendedStemActivity from backend (all 8 stems, 7 channels + drift detection)
  const handleExtendedActivity = useCallback(
    (data: ExtendedStemActivityMessage) => {
      useAudioActivityStore.getState().updateFromMessage(data);

      // Drift detection: compare backend time vs frontend time
      const frontendTime = useAudioStore.getState().currentTime;
      const backendTime = data.audio_time;
      const drift = Math.abs(frontendTime - backendTime);
      driftRef.current = drift;

      if (drift > 0.5) {
        console.warn(
          `[Sync] Drift detected: ${drift.toFixed(2)}s (frontend=${frontendTime.toFixed(2)}, backend=${backendTime.toFixed(2)})`
        );
      }
    },
    []
  );

  // Destination status from backend (blend position, mode, labels)
  const handleDestinationStatus = useCallback(
    (status: DestinationStatusMessage) => {
      useDestinationStore.getState().updateFromStatus(status);
    },
    []
  );

  // ========================================
  // WebSocket Hook
  // ========================================

  const {
    connect,
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
  } = useWebSocket({
    url: WS_CONFIG.URL,
    onFrame: handleFrame,
    onExtendedActivity: handleExtendedActivity,
    onTrackInfo: handleTrackInfo,
    onDestinationStatus: handleDestinationStatus,
    autoConnect: false,
    enableReconnect: true,
  });

  // Refs to track current values for async polling (avoids closure trap)
  const statusRef = useRef(status);
  const isGeneratingRef = useRef(isGenerating);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);
  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  // ========================================
  // Audio Handlers
  // ========================================

  const handleAudioReady = useCallback(
    (newAudioId: string) => {
      console.log("[App] Audio ready:", newAudioId);
      if (status === "disconnected") {
        connect();
      }
    },
    [status, connect]
  );

  // ========================================
  // Generation Handlers
  // ========================================

  // Helper: wait for a condition with timeout
  const waitFor = useCallback(
    async (condition: () => boolean, timeoutMs: number = 2000, pollMs: number = 50): Promise<boolean> => {
      const start = Date.now();
      while (Date.now() - start < timeoutMs) {
        if (condition()) return true;
        await new Promise((r) => setTimeout(r, pollMs));
      }
      return false; // Timeout
    },
    []
  );

  // Unified handler: connect -> start generation -> play
  const handlePlayAll = useCallback(async () => {
    if (!audioId) return;

    // 1. Connect if needed (wait for actual connection via ref polling)
    if (statusRef.current !== "connected") {
      connect();
      const connected = await waitFor(() => statusRef.current === "connected", 2000);
      if (!connected) {
        console.warn("[handlePlayAll] Connection timeout - proceeding anyway");
      }
    }

    // 2. Start generation if not already generating
    if (!isGeneratingRef.current) {
      sendStartSAESteering(audioId);
      const generating = await waitFor(() => isGeneratingRef.current, 1000);
      if (!generating) {
        console.warn("[handlePlayAll] Generation start timeout - proceeding anyway");
      }

      // 3. Sync steering mode from frontend to backend
      sendSetSteeringMode(steeringMode);

      // 4. Sync staged frontend control state to backend. This is what lets the
      // agent create a complete visual while generation is idle.
      const currentBlockMappings = useBlockStore.getState().blockMappings;
      for (const mapping of Object.values(currentBlockMappings)) {
        sendUpdateBlockConfig({
          action: 'update_block_config',
          block: mapping.block,
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

      // Sync latent destinations (seeds)
      if (latentDest.destinationA?.seed !== undefined) {
        sendSetDestination('latent', 'a', 'seed', { seed: latentDest.destinationA.seed });
      }
      if (latentDest.destinationB?.seed !== undefined) {
        sendSetDestination('latent', 'b', 'seed', { seed: latentDest.destinationB.seed });
      }

      // Sync prompt destinations (prompts)
      if (promptDest.destinationA?.prompt) {
        sendSetDestination('prompt', 'a', 'prompt', { prompt: promptDest.destinationA.prompt });
      }
      if (promptDest.destinationB?.prompt) {
        sendSetDestination('prompt', 'b', 'prompt', { prompt: promptDest.destinationB.prompt });
      }

      // Sync prompt mode + reactive/linked config
      sendSetDestinationMode('prompt', promptDest.mode);

      if (promptDest.mode === 'reactive') {
        sendSetReactiveConfig('prompt', promptDest.reactiveConfig);
      }

      if (promptDest.mode === 'linked' && promptDest.linkTarget) {
        sendSetDestinationLink('prompt', promptDest.linkTarget);
      }

      console.log(`[handlePlayAll] Synced destinations: latent=${latentDest.destinationA?.label ?? 'none'}->${latentDest.destinationB?.label ?? 'none'}, prompt=${promptDest.destinationA?.label ?? 'none'}->${promptDest.destinationB?.label ?? 'none'}`);
    }
  }, [audioId, connect, sendStartSAESteering, waitFor, steeringMode, sendSetSteeringMode, sendUpdateBlockConfig, sendSetCompositionConfig, sendSetDestination, sendSetDestinationMode, sendSetReactiveConfig, sendSetDestinationLink]);

  // ========================================
  // Block Config Handlers (extracted hook)
  // ========================================

  const {
    handleBlockLinkTargetChange,
    handleBlockFeatureChange,
    handleBlockStrengthRangeChange,
    handleBlockAutoConfigChange,
    handleBlockSpatialModeChange,
    handleBlockSpatialMaskChange,
    handleBlockIntensitySourceChange,
    handleBlockIntensityCurveChange,
    handleBlockIntensityGammaChange,
    handleBlockSaeRankChange,
    handleToggleBlock,
  } = useBlockConfigHandlers(sendUpdateBlockConfig);

  // ========================================
  // Destination Handlers (extracted hook)
  // ========================================

  const {
    handleSetPrompt,
    handleClearPromptDestination,
    handlePromptFreezeBlend,
    handlePromptSetBlendPosition,
    handlePromptSetMode,
    handlePromptSetReactiveConfig,
    handlePromptSetLinkTarget,
  } = useDestinationHandlers({
    sendSetDestination,
    sendClearDestination,
    sendFreezeBlend,
    sendSetBlendPosition,
    sendSetDestinationMode,
    sendSetReactiveConfig,
    sendSetDestinationLink,
  });

  const agentBridge = useAgentBridge({
    generationStatus: status,
    isGenerating,
    sendUpdateBlockConfig,
    sendSetDestination,
    sendClearDestination,
    sendFreezeBlend,
    sendSetBlendPosition,
    sendSetDestinationMode,
    sendSetReactiveConfig,
    sendSetDestinationLink,
    sendSetCompositionConfig,
  });

  // ========================================
  // Player Handlers
  // ========================================

  const handleHeartClick = useCallback(() => {
    if (isPlayerMinimized) {
      setIsPlayerMinimized(false);
    } else {
      setIsPlayerOpen(true);
    }
  }, [isPlayerMinimized]);

  const handlePlayerClose = useCallback(() => {
    setIsPlayerOpen(false);
    setIsPlayerMinimized(false);
  }, []);

  const handlePlayerMinimize = useCallback(() => {
    setIsPlayerMinimized(true);
  }, []);

  // ========================================
  // Destination Handlers
  // ========================================

  const handleDestinationOrbClick = useCallback((space: DestinationSpace) => {
    useDestinationStore.getState().setSelectedSpace(space);
  }, []);

  const handleDestinationPanelClose = useCallback(() => {
    useDestinationStore.getState().setSelectedSpace(null);
  }, []);

  // Overall activity — computed once per render (not a callback called 4x)
  const overallActivity = useMemo(() => {
    // External audio activity lives in Zustand; this prop acts as the render tick.
    void stemProminence;
    const enabledMappings = Object.values(blockMappings).filter(m => m.enabled);
    if (enabledMappings.length === 0) return 0;
    const audioStems = useAudioActivityStore.getState().stems;
    const sum = enabledMappings.reduce((acc, m) => {
      const baseStem = m.linkTarget.split('_')[0];
      return acc + (audioStems[baseStem as keyof typeof audioStems]?.energy_smooth ?? 0);
    }, 0);
    return sum / enabledMappings.length;
  }, [blockMappings, stemProminence]);

  // ========================================
  // Global Keyboard Shortcuts
  // ========================================

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input field
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
        return;
      }

      switch (e.key) {
        case "Escape":
          if (isPlayerOpen) {
            setIsPlayerOpen(false);
            setIsPlayerMinimized(false);
          }
          break;

        case " ": // Space = play/pause
          e.preventDefault();
          if (audioId) {
            const { isPlaying, play, pause, currentTime } = useAudioStore.getState();
            if (isPlaying) {
              pause();
              sendAudioPause();
            } else {
              play();
              sendAudioPlay(currentTime);
            }
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPlayerOpen, audioId, sendAudioPlay, sendAudioPause]);

  // ========================================
  // Disconnect Handler - Pause audio when WS disconnects
  // ========================================

  useEffect(() => {
    if (status === "error") {
      const { isPlaying, pause } = useAudioStore.getState();
      if (isPlaying) {
        pause();
      }
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

  // ========================================
  // Return
  // ========================================

  return {
    // Refs
    canvasRef,
    containerRef,
    driftRef,

    // Dimensions (passed through)
    dimensions,

    // Physics
    physics,

    // UI State
    isPlayerOpen,
    isPlayerMinimized,
    showPerfOverlay,

    // Store data
    stemActivity,
    blockMappings,
    trackInfo,
    stemProminence,
    latentDestinations,
    promptDestinations,
    selectedSpace,
    perfStats,
    agentBridge,

    // WebSocket
    status,
    fpsRef,
    isGenerating,

    // WebSocket sends (used in render)
    sendSetDestination,
    sendClearDestination,
    sendSetCompositionConfig,
    sendStopGeneration,
    sendAudioPlay,
    sendAudioPause,
    sendAudioSeek,
    sendAudioTimeUpdate,

    // Handlers
    handleHeartClick,
    handlePlayerClose,
    handlePlayerMinimize,
    handlePlayAll,
    handleAudioReady,
    handleDestinationOrbClick,
    handleDestinationPanelClose,
    overallActivity,

    // Block config handlers
    handleBlockLinkTargetChange,
    handleBlockFeatureChange,
    handleBlockStrengthRangeChange,
    handleBlockAutoConfigChange,
    handleBlockSpatialModeChange,
    handleBlockSpatialMaskChange,
    handleBlockIntensitySourceChange,
    handleBlockIntensityCurveChange,
    handleBlockIntensityGammaChange,
    handleBlockSaeRankChange,
    handleToggleBlock,

    // Destination handlers
    handleSetPrompt,
    handleClearPromptDestination,
    handlePromptFreezeBlend,
    handlePromptSetBlendPosition,
    handlePromptSetMode,
    handlePromptSetReactiveConfig,
    handlePromptSetLinkTarget,
  };
}

export type AppCore = ReturnType<typeof useAppCore>;
