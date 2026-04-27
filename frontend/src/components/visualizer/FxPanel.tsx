/**
 * FxPanel - Visual effects toggle (CRT scanlines, dither).
 *
 * Reads directly from useEffectsStore — no props needed.
 * Browser mode: bottom-right absolute. Desktop mode: inside ModeBar popover.
 */

import { useEffectsStore } from "../../stores/useEffectsStore";

export function FxPanel() {
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

  return (
    <div className="absolute bottom-6 right-6 z-50 flex flex-col items-end gap-2">
      {showEffectsPanel && (
        <div
          className="win95-panel p-3 flex flex-col gap-3"
          style={{ minWidth: '160px' }}
        >
          {/* Screen group */}
          <div className="flex flex-col gap-1.5">
            <div className="text-xxs uppercase tracking-wider" style={{ color: 'var(--color-text-dim)' }}>
              Screen
            </div>
            <button
              className={`win95-button text-xs w-full ${showCrt ? 'win95-button--primary' : ''}`}
              onClick={() => toggle("showCrt")}
            >
              {showCrt ? '\u25C6' : '\u25C7'} CRT
            </button>
            <button
              className={`win95-button text-xs w-full ${showBloom ? 'win95-button--primary' : ''}`}
              onClick={() => toggle("showBloom")}
            >
              {showBloom ? '\u25C6' : '\u25C7'} Bloom
            </button>
            <button
              className={`win95-button text-xs w-full ${showChromatic ? 'win95-button--primary' : ''}`}
              onClick={() => toggle("showChromatic")}
            >
              {showChromatic ? '\u25C6' : '\u25C7'} Chromatic
            </button>
          </div>
          {/* Texture group */}
          <div className="flex flex-col gap-1.5">
            <div className="text-xxs uppercase tracking-wider" style={{ color: 'var(--color-text-dim)' }}>
              Texture
            </div>
            <button
              className={`win95-button text-xs w-full ${showDither ? 'win95-button--primary' : ''}`}
              onClick={() => toggle("showDither")}
            >
              {showDither ? '\u25C6' : '\u25C7'} Dither
            </button>
            <button
              className={`win95-button text-xs w-full ${showHeavyGrain ? 'win95-button--primary' : ''}`}
              onClick={() => toggle("showHeavyGrain")}
            >
              {showHeavyGrain ? '\u25C6' : '\u25C7'} Grain
            </button>
          </div>
          {/* Glitch group */}
          <div className="flex flex-col gap-1.5">
            <div className="text-xxs uppercase tracking-wider" style={{ color: 'var(--color-text-dim)' }}>
              Glitch
            </div>
            <button
              className={`win95-button text-xs w-full ${showVhsTracking ? 'win95-button--primary' : ''}`}
              onClick={() => toggle("showVhsTracking")}
            >
              {showVhsTracking ? '\u25C6' : '\u25C7'} VHS
            </button>
          </div>
        </div>
      )}
      <button
        className="win95-button text-xs px-3 py-1.5"
        onClick={() => toggle("showEffectsPanel")}
        style={{ opacity: showEffectsPanel ? 1 : 0.6 }}
        title="Visual effects"
      >
        FX
      </button>
    </div>
  );
}
