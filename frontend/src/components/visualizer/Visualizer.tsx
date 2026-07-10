/**
 * Visualizer - The render tree for the music visualizer.
 *
 * Receives only refs/physics/orchestration from useAppCore; everything else
 * comes from stores and lib/wire.ts right here, so nothing rides through
 * this component untouched. High-frequency audio state (activity,
 * prominence) is deliberately NOT subscribed at this level — the leaf
 * components that render it subscribe themselves, so this tree only
 * re-renders on user interaction and connection changes.
 *
 * NOTE: PerfOverlay, FxPanel, and HelpDialog are rendered by the app shell
 * (BrowserApp / DesktopApp), NOT here. This keeps Visualizer mode-agnostic.
 */

import { useMemo } from "react";
import { Canvas } from "../Canvas";
import { CrystalHeart } from "../CrystalHeart";
import { BellyScene } from "../BellyScene";
import { PlantStems } from "../PlantStems";
import { AudioPlayerWindow } from "../AudioPlayerWindow";
import { OrbSystem } from "../OrbSystem";
import { Notifications } from "../Notifications";
import { CompositionPanel, PromptDestinationPanel } from "../scenes";
import { useTrackInfo } from "../../stores/useSessionStore";
import { useConnectionStore } from "../../stores/useConnectionStore";
import { usePlayerWindowStore } from "../../stores/usePlayerWindowStore";
import { useDestinationStore } from "../../stores/useDestinationStore";
import { useEffectsStore } from "../../stores/useEffectsStore";
import { useShallow } from "zustand/shallow";
import {
  sendAudioPlay,
  sendAudioPause,
  sendAudioSeek,
  sendAudioTimeUpdate,
  sendStopGeneration,
  sendSetDestination,
  sendClearDestination,
  sendSetCompositionConfig,
} from "../../lib/wire";
import {
  handleSetPrompt,
  handleClearPromptDestination,
  handlePromptFreezeBlend,
  handlePromptSetBlendPosition,
  handlePromptSetMode,
  handlePromptSetReactiveConfig,
  handlePromptSetLinkTarget,
} from "../../lib/destinationControls";
import type { AppCore } from "../../hooks/useAppCore";
import type { DestinationSpace } from "../../types/destinations";

/** Stable empty object — avoids creating new reference every render */
const EMPTY_HOVER: Record<string, boolean> = {};

function handleDestinationOrbClick(space: DestinationSpace) {
  useDestinationStore.getState().setSelectedSpace(space);
}

function handleDestinationPanelClose() {
  useDestinationStore.getState().setSelectedSpace(null);
}

function handleHeartClick() {
  usePlayerWindowStore.getState().openFromHeart();
}

// ============================================================================
// Visualizer
// ============================================================================

export function Visualizer(props: AppCore) {
  const { physics, canvasRef, containerRef, dimensions, handlePlayAll, handleAudioReady } = props;

  const trackInfo = useTrackInfo();
  const status = useConnectionStore((s) => s.status);
  const isGenerating = useConnectionStore((s) => s.isGenerating);
  const isPlayerOpen = usePlayerWindowStore((s) => s.isOpen);
  const isPlayerMinimized = usePlayerWindowStore((s) => s.isMinimized);

  // Destination modulation store (shallow compare to avoid re-render on blendPosition updates)
  const latentDestinations = useDestinationStore(useShallow((s) => s.latent));
  const promptDestinations = useDestinationStore(useShallow((s) => s.prompt));
  const selectedSpace = useDestinationStore((s) => s.selectedSpace);

  // Visual effects from store (shallow selector: only re-render when these 6 fields change)
  const { showCrt, showDither, showChromatic, showBloom, showVhsTracking, showHeavyGrain } =
    useEffectsStore(useShallow((s) => ({
      showCrt: s.showCrt,
      showDither: s.showDither,
      showChromatic: s.showChromatic,
      showBloom: s.showBloom,
      showVhsTracking: s.showVhsTracking,
      showHeavyGrain: s.showHeavyGrain,
    })));

  // Memoize destination states to avoid new object every render
  const destinationStates = useMemo(() => ({
    latent: {
      mode: latentDestinations.mode,
      destinationA: latentDestinations.destinationA,
      destinationB: latentDestinations.destinationB,
    },
    prompt: {
      mode: promptDestinations.mode,
      destinationA: promptDestinations.destinationA,
      destinationB: promptDestinations.destinationB,
    },
  }), [
    latentDestinations.mode, latentDestinations.destinationA, latentDestinations.destinationB,
    promptDestinations.mode, promptDestinations.destinationA, promptDestinations.destinationB,
  ]);

  // Wait for physics to initialize
  if (!physics) {
    return (
      <div ref={containerRef} className="relative w-full h-full overflow-clip flex items-center justify-center">
        <div className="text-sm" style={{ color: "var(--color-text-muted)" }}>
          Initializing physics...
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-clip">
      {/* Canvas: fullscreen, the star of the show */}
      <Canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

      {/* Visual Effect Overlays (z-order: grain→VHS→bloom→chromatic→dither→noise→CRT) */}
      {showHeavyGrain && <div className="heavy-grain-overlay" />}
      {showVhsTracking && <div className="vhs-tracking-overlay" />}
      {showBloom && <div className="bloom-overlay" />}
      {showChromatic && <div className="chromatic-overlay" />}
      {showDither && <div className="dither-overlay" />}
      {showCrt && <div className="noise-overlay" />}
      {showCrt && <div className="crt-overlay" />}

      {/* Plant Stems - waveform visualization connecting flowers to heart */}
      <PlantStems
        stemOrbBodies={physics.bodies.stemOrbs}
        heartBody={physics.bodies.heart}
        destinationOrbBodies={physics.bodies.destinationOrbs}
        width={dimensions.width}
        height={dimensions.height}
      />

      {/* BellyScene - Single R3F Canvas for all 3D content (heart + orbs) */}
      <BellyScene
        width={dimensions.width}
        height={dimensions.height}
        heartBody={physics.bodies.heart}
        stemOrbBodies={physics.bodies.stemOrbs}
        destinationOrbBodies={physics.bodies.destinationOrbs}
        heartIsDragging={physics.isDragging(physics.bodies.heart)}
        heartIsHovered={false}
        heartIsPlayerOpen={isPlayerOpen && !isPlayerMinimized}
        heartIsReadyToGenerate={
          (latentDestinations.destinationA !== null || latentDestinations.destinationB !== null) &&
          (promptDestinations.destinationA !== null || promptDestinations.destinationB !== null)
        }
        destinationStates={destinationStates}
        orbHoverStates={EMPTY_HOVER}
      />

      {/* CrystalHeart - HTML overlay (click/drag/glow) — 3D mesh in BellyScene */}
      <CrystalHeart
        body={physics.bodies.heart}
        isDragging={physics.isDragging(physics.bodies.heart)}
        onClick={handleHeartClick}
      />

      {/* Slot Orbs */}
      <OrbSystem
        stemBodies={physics.bodies.stemOrbs}
        destinationBodies={physics.bodies.destinationOrbs}
        isDragging={physics.isDragging}
        containerSize={dimensions}
        destinationStates={destinationStates}
        onDestinationClick={handleDestinationOrbClick}
      />

      {/* Composition Panel - opens when latent orb is clicked */}
      <CompositionPanel
        destinationA={latentDestinations.destinationA}
        destinationB={latentDestinations.destinationB}
        isOpen={selectedSpace === 'latent'}
        onClose={handleDestinationPanelClose}
        orbPosition={
          physics?.bodies.destinationOrbs?.[0]
            ? {
                x: physics.bodies.destinationOrbs[0].position.x,
                y: physics.bodies.destinationOrbs[0].position.y,
              }
            : { x: 50, y: dimensions.height / 2 }
        }
        containerSize={dimensions}
        onSetSeed={(slot, seed) => {
          sendSetDestination('latent', slot, 'seed', { seed });
          useDestinationStore.getState().setDestination('latent', slot, { type: 'seed', label: `Seed ${seed}`, seed });
        }}
        onClearDestination={(slot) => {
          sendClearDestination('latent', slot);
          useDestinationStore.getState().clearDestination('latent', slot);
        }}
        onSetCompositionConfig={sendSetCompositionConfig}
      />

      {/* Prompt Scenes Panel - opens when prompt orb is clicked */}
      <PromptDestinationPanel
        destinationA={promptDestinations.destinationA}
        destinationB={promptDestinations.destinationB}
        blendPosition={promptDestinations.blendPosition}
        mode={promptDestinations.mode}
        reactiveConfig={promptDestinations.reactiveConfig}
        isOpen={selectedSpace === 'prompt'}
        onClose={handleDestinationPanelClose}
        orbPosition={
          physics?.bodies.destinationOrbs?.[1]
            ? {
                x: physics.bodies.destinationOrbs[1].position.x,
                y: physics.bodies.destinationOrbs[1].position.y,
              }
            : { x: dimensions.width - 50, y: dimensions.height / 2 }
        }
        containerSize={dimensions}
        onSetPrompt={handleSetPrompt}
        onClearDestination={handleClearPromptDestination}
        onFreezeBlend={handlePromptFreezeBlend}
        onSetBlendPosition={handlePromptSetBlendPosition}
        onSetMode={handlePromptSetMode}
        onSetReactiveConfig={handlePromptSetReactiveConfig}
        linkTarget={promptDestinations.linkTarget}
        onSetLinkTarget={handlePromptSetLinkTarget}
      />

      {/* Audio Player Window */}
      <AudioPlayerWindow
        isOpen={isPlayerOpen}
        onClose={() => usePlayerWindowStore.getState().close()}
        onMinimize={() => usePlayerWindowStore.getState().minimize()}
        isMinimized={isPlayerMinimized}
        onPlay={sendAudioPlay}
        onPause={sendAudioPause}
        onSeek={sendAudioSeek}
        onGenerate={handlePlayAll}
        onStopGeneration={sendStopGeneration}
        onNewSong={sendStopGeneration}
        isGenerating={isGenerating}
        onAudioReady={handleAudioReady}
        wsStatus={status}
        onTimeSync={sendAudioTimeUpdate}
        bpm={trackInfo?.bpm}
      />

      {/* Toast Notifications */}
      <Notifications />
    </div>
  );
}
