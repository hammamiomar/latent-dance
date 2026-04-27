/**
 * DataOverlay - Placeholder for desktop data/telemetry mode.
 *
 * Opaque overlay above the Visualizer workspace.
 * Will contain SAE feature browser, audio analysis, etc.
 */

export function DataOverlay() {
  return (
    <div className="desktop-overlay">
      <div className="text-lg" style={{ color: "var(--color-text-muted)" }}>
        DATA
      </div>
      <div className="text-xs" style={{ color: "var(--color-text-dim)" }}>
        Coming Soon
      </div>
    </div>
  );
}
