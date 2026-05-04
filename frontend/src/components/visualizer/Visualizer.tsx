/**
 * Visualizer - The render tree for the music visualizer.
 *
 * Receives all state/handlers from useAppCore via props.
 * Positions everything absolutely within a relative container.
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
import { useDestinationStore } from "../../stores/useDestinationStore";
import { useEffectsStore } from "../../stores/useEffectsStore";
import { useShallow } from "zustand/shallow";
import type { AppCore } from "../../hooks/useAppCore";

/** Stable empty object — avoids creating new reference every render */
const EMPTY_HOVER: Record<string, boolean> = {};

// ============================================================================
// Visualizer
// ============================================================================

export function Visualizer(props: AppCore) {
  const {
    physics,
    canvasRef,
    containerRef,
    dimensions,
    // Store data
    stemActivity,
    blockMappings,
    trackInfo,
    stemProminence,
    latentDestinations,
    promptDestinations,
    selectedSpace,
    // WebSocket
    status,
    isGenerating,
    isPlayerOpen,
    isPlayerMinimized,
    // WebSocket sends
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
  } = props;

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
        blockMappings={blockMappings}
        stemOrbBodies={physics.bodies.stemOrbs}
        heartBody={physics.bodies.heart}
        destinationOrbBodies={physics.bodies.destinationOrbs}
        latentConfigured={latentDestinations.destinationA !== null && latentDestinations.destinationB !== null}
        promptConfigured={promptDestinations.destinationA !== null && promptDestinations.destinationB !== null}
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
        heartActivity={overallActivity}
        heartBpm={trackInfo?.bpm ?? 120}
        heartIsDragging={physics.isDragging(physics.bodies.heart)}
        heartIsHovered={false}
        heartIsPlayerOpen={isPlayerOpen && !isPlayerMinimized}
        heartIsReadyToGenerate={
          (latentDestinations.destinationA !== null || latentDestinations.destinationB !== null) &&
          (promptDestinations.destinationA !== null || promptDestinations.destinationB !== null)
        }
        blockMappings={blockMappings}
        stemActivity={stemActivity}
        destinationActivity={overallActivity}
        destinationStates={destinationStates}
        orbHoverStates={EMPTY_HOVER}
      />

      {/* CrystalHeart - HTML overlay (click/drag/glow) — 3D mesh in BellyScene */}
      <CrystalHeart
        body={physics.bodies.heart}
        isDragging={physics.isDragging(physics.bodies.heart)}
        onClick={handleHeartClick}
        activity={overallActivity}
      />

      {/* Block Orbs */}
      <OrbSystem
        stemBodies={physics.bodies.stemOrbs}
        destinationBodies={physics.bodies.destinationOrbs}
        isDragging={physics.isDragging}
        stemActivity={stemActivity}
        stemProminence={stemProminence}
        blockMappings={blockMappings}
        containerSize={dimensions}
        onLinkTargetChange={handleBlockLinkTargetChange}
        onFeatureChange={handleBlockFeatureChange}
        onStrengthRangeChange={handleBlockStrengthRangeChange}
        onAutoConfigChange={handleBlockAutoConfigChange}
        onSpatialModeChange={handleBlockSpatialModeChange}
        onSpatialMaskChange={handleBlockSpatialMaskChange}
        onIntensitySourceChange={handleBlockIntensitySourceChange}
        onIntensityCurveChange={handleBlockIntensityCurveChange}
        onIntensityGammaChange={handleBlockIntensityGammaChange}
        onSaeRankChange={handleBlockSaeRankChange}
        onToggleBlock={handleToggleBlock}
        destinationActivity={overallActivity}
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
        onClose={handlePlayerClose}
        onMinimize={handlePlayerMinimize}
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
