/**
 * bootstrap — fetch the backend capability manifest before connecting.
 *
 * The manifest supplies the backend mode (which selects the WS endpoint
 * /ws/stream/{mode}) and the slot vocabulary. There is deliberately NO
 * fallback manifest: rendering must be driven by what the backend declares,
 * never by a hardcoded SAE shape. Until this resolves, the generation
 * shells show a "waiting for backend" gate.
 */

import { parseBackendCapabilities } from "../types/wire/capabilities";
import { useSessionStore } from "../stores/useSessionStore";

const CAPABILITIES_URL = "/api/capabilities";
const RETRY_DELAY_MS = 1500;

let inflight: Promise<void> | null = null;

async function fetchOnce(): Promise<boolean> {
  try {
    const response = await fetch(CAPABILITIES_URL);
    if (!response.ok) return false;
    const manifest = parseBackendCapabilities(await response.json());
    useSessionStore.getState().setCapabilities(manifest);
    return true;
  } catch (error) {
    console.warn("[bootstrap] /api/capabilities not reachable yet:", error);
    return false;
  }
}

/**
 * Resolve once the session store holds a capabilities manifest, retrying
 * until the backend answers (it may still be loading the pipeline).
 * Idempotent: concurrent callers share one loop.
 */
export function ensureCapabilities(): Promise<void> {
  if (useSessionStore.getState().capabilities) return Promise.resolve();
  inflight ??= (async () => {
    try {
      while (!useSessionStore.getState().capabilities) {
        if (await fetchOnce()) break;
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
      }
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}
