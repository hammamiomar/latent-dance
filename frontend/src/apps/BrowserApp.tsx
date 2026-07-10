/**
 * BrowserApp - Full-viewport visualizer with app chrome.
 *
 * In browser mode the visualizer IS the viewport.
 * PerfOverlay, FxPanel, and HelpDialog render alongside it.
 */

import { useState, useEffect } from "react";
import { useAppCore } from "../hooks/useAppCore";
import { useCapabilities } from "../stores/useSessionStore";
import { Visualizer } from "../components/visualizer/Visualizer";
import { BackendGate } from "../components/visualizer/BackendGate";
import { PerfOverlay } from "../components/visualizer/PerfOverlay";
import { FxPanel } from "../components/visualizer/FxPanel";
import { HelpDialog } from "../components/HelpDialog";

export function BrowserApp() {
  const [showHelp, setShowHelp] = useState(false);
  const [dimensions, setDimensions] = useState({
    width: typeof window !== "undefined" ? window.innerWidth : 1200,
    height: typeof window !== "undefined" ? window.innerHeight : 800,
  });

  useEffect(() => {
    const onResize = () =>
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const core = useAppCore(dimensions);
  const capabilities = useCapabilities();

  return (
    <div className="w-screen h-screen bg-void-abyss overflow-hidden relative">
      {capabilities ? <Visualizer {...core} /> : <BackendGate />}
      {core.showPerfOverlay && <PerfOverlay />}
      <FxPanel />
      <HelpDialog isOpen={showHelp} onClose={() => setShowHelp(false)} />
    </div>
  );
}
