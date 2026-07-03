/**
 * ModeBar - Green backlit pushbuttons at the bottom of the character.
 *
 * Section labels: "SYSTEM CONTROL" | "PRESETS" | "LINK"
 * Pushbuttons use Josh Comeau's 3D button technique: edge + face layers
 * with translateY for physical throw on press.
 */

import { useState, useEffect, useCallback } from "react";
import { SHELL_BRIDGE_PORT } from "../../constants";
import { useEffectsStore } from "../../stores/useEffectsStore";

export type BellyMode = "visualizer" | "data";

interface ModeBarProps {
  activeMode: BellyMode;
  onModeChange: (mode: BellyMode) => void;
  showHelp: boolean;
  onHelpToggle: () => void;
}

const MODES: { key: BellyMode; label: string; title: string }[] = [
  { key: "visualizer", label: "BELLY", title: "Belly" },
  { key: "data", label: "DATA", title: "Data" },
];

export function ModeBar({ activeMode, onModeChange, showHelp, onHelpToggle }: ModeBarProps) {
  const {
    showCrt,
    showDither,
    showEffectsPanel,
    showChromatic,
    showBloom,
    showVhsTracking,
    showHeavyGrain,
    toggle,
  } = useEffectsStore();
  const [pinned, setPinned] = useState(true);
  const [brainOpen, setBrainOpen] = useState(false);

  useEffect(() => {
    fetch(`http://localhost:${SHELL_BRIDGE_PORT}/pin`)
      .then((r) => r.json())
      .then((data) => setPinned(data.pinned))
      .catch(() => {});
  }, []);

  // Track brain window state — initial fetch + slow poll for external closures.
  // The bridge only exists inside the Electrobun shell; give up after a few
  // misses so ?desktop=true in a plain browser doesn't ping forever.
  useEffect(() => {
    let cancelled = false;
    let failures = 0;
    let interval: number | undefined;
    const fetchStatus = () => {
      fetch(`http://localhost:${SHELL_BRIDGE_PORT}/brain/status`)
        .then((r) => r.json())
        .then((data) => {
          failures = 0;
          if (!cancelled) setBrainOpen(Boolean(data.open));
        })
        .catch(() => {
          failures += 1;
          if (failures >= 3 && interval !== undefined) {
            window.clearInterval(interval);
            interval = undefined;
          }
        });
    };
    fetchStatus();
    interval = window.setInterval(fetchStatus, 2000);
    return () => {
      cancelled = true;
      if (interval !== undefined) window.clearInterval(interval);
    };
  }, []);

  const togglePin = useCallback(() => {
    fetch(`http://localhost:${SHELL_BRIDGE_PORT}/pin`, { method: "POST" })
      .then((r) => r.json())
      .then((data) => setPinned(data.pinned))
      .catch(() => setPinned((p) => !p));
  }, []);

  const toggleBrainWindow = useCallback(() => {
    // Optimistic: flip immediately, then sync from response
    setBrainOpen((prev) => !prev);
    fetch(`http://localhost:${SHELL_BRIDGE_PORT}/brain/spawn`, { method: "POST" })
      .then((r) => r.json())
      .then((data) => setBrainOpen(Boolean(data.open)))
      .catch(() => {});
  }, []);

  return (
    <div className="shrink-0 relative">
      {/* FX popover — hardware-styled panel above buttons */}
      {showEffectsPanel && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-[10500]">
          <div className="fx-popover p-3 flex flex-col gap-3" style={{ minWidth: "160px" }}>
            <div className="flex flex-col gap-1.5">
              <div className="fx-popover__label">Screen</div>
              <button
                className={`led-toggle w-full ${showCrt ? "led-toggle--active" : ""}`}
                onClick={() => toggle("showCrt")}
              >
                {showCrt ? "\u25C6" : "\u25C7"} CRT
              </button>
              <button
                className={`led-toggle w-full ${showBloom ? "led-toggle--active" : ""}`}
                onClick={() => toggle("showBloom")}
              >
                {showBloom ? "\u25C6" : "\u25C7"} Bloom
              </button>
              <button
                className={`led-toggle w-full ${showChromatic ? "led-toggle--active" : ""}`}
                onClick={() => toggle("showChromatic")}
              >
                {showChromatic ? "\u25C6" : "\u25C7"} Chromatic
              </button>
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="fx-popover__label">Texture</div>
              <button
                className={`led-toggle w-full ${showDither ? "led-toggle--active" : ""}`}
                onClick={() => toggle("showDither")}
              >
                {showDither ? "\u25C6" : "\u25C7"} Dither
              </button>
              <button
                className={`led-toggle w-full ${showHeavyGrain ? "led-toggle--active" : ""}`}
                onClick={() => toggle("showHeavyGrain")}
              >
                {showHeavyGrain ? "\u25C6" : "\u25C7"} Grain
              </button>
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="fx-popover__label">Glitch</div>
              <button
                className={`led-toggle w-full ${showVhsTracking ? "led-toggle--active" : ""}`}
                onClick={() => toggle("showVhsTracking")}
              >
                {showVhsTracking ? "\u25C6" : "\u25C7"} VHS
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Pushbutton row */}
      <div className="pushbutton-row">
        {MODES.map(({ key, label, title }) => {
          const isActive = activeMode === key;
          return (
            <button
              key={key}
              className={`pushbutton ${isActive ? "pushbutton--active" : ""}`}
              onClick={() => onModeChange(key)}
              title={title}
            >
              <span className="pushbutton__edge" />
              <span className="pushbutton__face">{label}</span>
            </button>
          );
        })}

        <button
          className={`pushbutton ${brainOpen ? "pushbutton--active" : ""}`}
          onClick={toggleBrainWindow}
          title={brainOpen ? "Close brain" : "Open brain"}
        >
          <span className="pushbutton__edge" />
          <span className="pushbutton__face">BRAIN</span>
        </button>

        {/* Vertical separator */}
        <div className="pushbutton-separator" />

        {/* FX toggle */}
        <button
          className={`pushbutton ${showEffectsPanel ? "pushbutton--active" : ""}`}
          onClick={() => toggle("showEffectsPanel")}
          title="Visual effects"
        >
          <span className="pushbutton__edge" />
          <span className="pushbutton__face">FX</span>
        </button>

        {/* Separator */}
        <div className="pushbutton-separator" />

        {/* Pin toggle */}
        <button
          className={`pushbutton pushbutton--small ${pinned ? "pushbutton--active" : ""}`}
          onClick={togglePin}
          title="Sticky window"
        >
          <span className="pushbutton__edge" />
          <span className="pushbutton__face">STICKY</span>
        </button>

        {/* Separator */}
        <div className="pushbutton-separator" />

        {/* Help / parameter guide */}
        <button
          className={`pushbutton pushbutton--small ${showHelp ? "pushbutton--active" : ""}`}
          onClick={onHelpToggle}
          title="Parameter guide"
        >
          <span className="pushbutton__edge" />
          <span className="pushbutton__face">[?]</span>
        </button>
      </div>
    </div>
  );
}
