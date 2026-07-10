/**
 * BackendGate - honest "waiting" state while the capability manifest loads.
 *
 * Shown by the generation shells instead of the Visualizer until
 * lib/bootstrap.ts has fetched /api/capabilities (the backend may still be
 * loading its pipeline). There is no fallback manifest by design, so this
 * is the only thing that may render before the backend declares itself.
 */

export function BackendGate() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div
        className="animate-pulse"
        style={{
          fontFamily: "monospace",
          fontSize: "11px",
          letterSpacing: "0.25em",
          color: "var(--color-text-muted)",
        }}
      >
        WAITING FOR BACKEND…
      </div>
    </div>
  );
}
