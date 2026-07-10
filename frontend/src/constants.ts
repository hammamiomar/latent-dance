/*
Constants for App Configs... maybe I name this config
*/
// Determine WebSocket URL based on environment
function getBrowserLocation() {
  return typeof window === "undefined" ? null : window.location;
}

/**
 * Streaming endpoint for a backend mode (from the capabilities manifest).
 * The server rejects mode mismatches, so this is only called once the
 * manifest is known — there is no hardcoded default mode.
 */
export function getWsUrl(mode: string): string {
  const path = `/ws/stream/${mode}`;
  const location = getBrowserLocation();
  if (!location) return `ws://localhost${path}`;

  // In development, use relative URL (Vite proxy handles it)
  // In production, use the same host as the page
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}`;
}

export const WS_CONFIG = {
  RECONNECT_DELAY: 2000, // in ms
  MAX_RECONNECT_ATTEMPTS: 3, // Reduced to avoid spam
} as const;

export const CANVAS_CONFIG = {
  BACKGROUND_COLOR: "#000000",
  JPEG_QUALITY: 90,
} as const;

export const PERF_CONFIG = {
  FPS_UPDATE_INTERVAL: 1000, // update interval in ms AKA: how often to recalculate FPS
  LATENCY_SAMPLE_SIZE: 10, // number of latency samples to average : Future use averaging ping times
} as const;

export const IS_DESKTOP_MODE = new URLSearchParams(getBrowserLocation()?.search ?? "").has("desktop");

// Electrobun shell bridge — tiny HTTP server in the main process for
// features that need native API access (setAlwaysOnTop, etc.)
export const SHELL_BRIDGE_PORT = 14321;

export const AGENT_BRIDGE_WS_URL = `ws://127.0.0.1:${SHELL_BRIDGE_PORT}/agent/ws?role=frontend`;
