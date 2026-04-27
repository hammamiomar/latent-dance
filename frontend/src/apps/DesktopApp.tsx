/**
 * DesktopApp - Character-wrapped entry point (square layout).
 *
 * Zone model (matching skeuomorphic mockup):
 *   TopZone:      face screen (narrower) + speaker grills on sides
 *   LabelZone:    engraved model/serial text + LED indicator strip
 *   MiddleZone:   belly screen (square SDXL canvas) + appendages
 *   SectionLabels: "SYSTEM CONTROL" / "PRESETS" / "LINK"
 *   ButtonRow:    green backlit pushbuttons (ModeBar)
 *
 * Visualizer stays mounted during all mode switches.
 * Face is sacred — nothing overlays it.
 */

import { useState, useEffect, useCallback } from "react";
import { useAppCore } from "../hooks/useAppCore";
import { Visualizer } from "../components/visualizer/Visualizer";
import { PerfOverlay } from "../components/visualizer/PerfOverlay";
import { HelpDialog } from "../components/HelpDialog";
import { CharacterBody } from "../components/character/CharacterBody";
import { FaceScreen } from "../components/character/FaceScreen";
import { FaceRenderer } from "../components/character/FaceRenderer";
import { BellyScreen } from "../components/character/BellyScreen";
import { Appendages } from "../components/character/Appendages";
import { ModeBar, type BellyMode } from "../components/character/ModeBar";
import { SettingsOverlay } from "../components/character/SettingsOverlay";
import { DataOverlay } from "../components/character/DataOverlay";
import { EngravedLogo } from "../components/character/EngravedLogo";

export function DesktopApp() {
  const [bellySize, setBellySize] = useState({ width: 0, height: 0 });
  const [activeMode, setActiveMode] = useState<BellyMode>("visualizer");
  const [staticFlash, setStaticFlash] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  // Transparent background for Electrobun frameless window
  useEffect(() => {
    document.documentElement.style.background = "transparent";
    document.body.style.background = "transparent";
    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";

    const root = document.getElementById("root");
    if (root) {
      root.style.background = "transparent";
      root.style.borderRadius = "10px";
      root.style.overflow = "hidden";
    }
  }, []);

  const core = useAppCore(bellySize);

  const handleModeChange = useCallback(
    (mode: BellyMode) => {
      if (mode === activeMode) return;
      setStaticFlash(true);
      setTimeout(() => {
        setActiveMode(mode);
        setStaticFlash(false);
      }, 100);
    },
    [activeMode]
  );

  return (
    <CharacterBody>
      {/* === TopZone: face display flanked by speaker grills === */}
      <div className="shrink-0 flex items-stretch">
        {/* Left speaker */}
        <div className="speaker-grill flex-1 min-w-[40px]" />

        {/* Face screen — narrower, centered */}
        <FaceScreen>
          <FaceRenderer />
          <div className="face-glass" />
          <div className="face-scanlines" />
        </FaceScreen>

        {/* Right speaker */}
        <div className="speaker-grill flex-1 min-w-[40px]" />
      </div>

      {/* === LabelZone: engraved text + LED indicators === */}
      <div className="label-zone shrink-0">
        <div className="flex flex-col gap-px">
          <div className="label-engraved">MODEL: HAMBAJUBA</div>
          <div className="label-engraved">SERIAL NO. 27000</div>
        </div>
        <div className="led-cluster">
          <div className="led-indicator" />
          <div className="led-indicator led-indicator--amber" />
        </div>
      </div>

      {/* === MiddleZone: belly + appendages === */}
      <div className="relative flex-1 min-h-0 overflow-clip">
        <Appendages />

        {/* Engraved logo — manufacturer stamp on the body metal */}
        <div className="absolute -right-3 bottom-2 z-10 pointer-events-none">
          <EngravedLogo height={140} />
        </div>

        <BellyScreen onResize={setBellySize}>
          <Visualizer {...core} />

          {core.showPerfOverlay && (
            <PerfOverlay fpsRef={core.fpsRef} driftRef={core.driftRef} perfStats={core.perfStats} />
          )}

          {staticFlash && (
            <div className="absolute inset-0 z-[10200] mode-transition-static pointer-events-none" />
          )}

          {activeMode === "settings" && <SettingsOverlay />}
          {activeMode === "data" && <DataOverlay />}
        </BellyScreen>
      </div>

      {/* === ControlZone: vents + button row === */}
      <div className="shrink-0 flex items-stretch">
        {/* Left vent */}
        <div className="vent-slot" />

        {/* Button row fills remaining space */}
        <div className="flex-1 min-w-0">
          <ModeBar
            activeMode={activeMode}
            onModeChange={handleModeChange}
            showHelp={showHelp}
            onHelpToggle={() => setShowHelp(h => !h)}
          />
        </div>

        {/* Right vent — mirrored */}
        <div className="vent-slot" style={{ transform: "scaleX(-1)" }} />
      </div>

      <HelpDialog isOpen={showHelp} onClose={() => setShowHelp(false)} />
    </CharacterBody>
  );
}
