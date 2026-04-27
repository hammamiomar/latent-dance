/**
 * SettingsOverlay - Placeholder for desktop settings mode.
 *
 * Opaque overlay above the Visualizer workspace.
 * Will contain connection config, audio routing, etc.
 */

export function SettingsOverlay() {
  return (
    <div className="desktop-overlay">
      <div className="text-lg" style={{ color: "var(--color-text-muted)" }}>
        SETTINGS
      </div>
      <div className="text-xs" style={{ color: "var(--color-text-dim)" }}>
        Coming Soon
      </div>
    </div>
  );
}
