/**
 * clientPerf — render-free counters for the diagnostic overlay.
 *
 * Written by useWebSocket (frame rate once per second, drift per telemetry
 * message); read by PerfOverlay during its own re-renders. A mutable module
 * object instead of store state so 10Hz telemetry never forces a React
 * update just to keep a hidden overlay current.
 */

export const clientPerf = {
  /** Frames received per second over the WS, updated ~1Hz. */
  wsFps: 0,
  /** |frontend audio time − backend audio time| in seconds, updated per telemetry message. */
  driftSec: 0,
};
