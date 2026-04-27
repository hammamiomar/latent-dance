/*
Constants for App Configs... maybe I name this config
*/
// Determine WebSocket URL based on environment
const getWsUrl = () => {
  // In development, use relative URL (Vite proxy handles it)
  // In production, use the same host as the page
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return `${protocol}//${host}/ws/stream/sae_steering`;
};

export const WS_CONFIG = {
  // SAE steering endpoint - uses proxy in dev, same host in prod
  URL: getWsUrl(),
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

export const IS_DESKTOP_MODE = new URLSearchParams(window.location.search).has('desktop');

// Electrobun shell bridge — tiny HTTP server in the main process for
// features that need native API access (setAlwaysOnTop, etc.)
export const SHELL_BRIDGE_PORT = 14321;
