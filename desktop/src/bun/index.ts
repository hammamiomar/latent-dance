/**
 * Main process — Electrobun desktop shell for hambajuba2ba.
 *
 * Loads Vite dev server (localhost:3000), which proxies API/WS
 * to localhost:8000, where the API server is expected to run.
 *
 * titleBarStyle:"hiddenInset" keeps native traffic lights (close/min/zoom)
 * but lets us position them inside the body. Native effects (shadow, drag)
 * are loaded via Bun FFI from libMacWindowEffects.dylib.
 */

import { BrowserWindow, ApplicationMenu } from "electrobun/bun";
import { dlopen, FFIType } from "bun:ffi";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ServerWebSocket } from "bun";
import { hermesModelName, hermesResponsesUrl, submitDirectiveToHermes } from "./hermesApi";

// --- Constants ---
const MIN_WIDTH = 800;
const MIN_HEIGHT = 800;
const DEFAULT_WIDTH = 1100;
const DEFAULT_HEIGHT = 1100;
const BRAIN_WIDTH = 520;
const BRAIN_HEIGHT = 720;

const APP_URL = "http://localhost:3000?desktop=true";
const BRAIN_URL = "http://localhost:3000/brain?desktop=true";

// Traffic lights hidden — moved off-screen so screws can use the corners.
// Close: Cmd+Q. Minimize: Cmd+M. Zoom: not needed (1:1 ratio enforced).
const TRAFFIC_LIGHTS_X = -100;
const TRAFFIC_LIGHTS_Y = -100;
const DRAG_REGION_X = 0;   // full width drag region (no traffic lights to dodge)
const DRAG_REGION_HEIGHT = 50;

// Brain window — top "title strip" with engraved BRAIN branding is drag region
const BRAIN_DRAG_HEIGHT = 40;

// --- Native macOS effects loader (shared by main + brain windows) ---
function loadNativeEffects() {
  if (process.platform !== "darwin") return null;
  const dylibPath = join(import.meta.dir, "libMacWindowEffects.dylib");
  if (!existsSync(dylibPath)) return null;
  try {
    const lib = dlopen(dylibPath, {
      ensureWindowShadow: {
        args: [FFIType.ptr],
        returns: FFIType.bool,
      },
      setWindowTrafficLightsPosition: {
        args: [FFIType.ptr, FFIType.f64, FFIType.f64],
        returns: FFIType.bool,
      },
      setNativeWindowDragRegion: {
        args: [FFIType.ptr, FFIType.f64, FFIType.f64],
        returns: FFIType.bool,
      },
      setTrafficLightsSubdued: {
        args: [FFIType.ptr],
        returns: FFIType.bool,
      },
      setWindowAspectRatio: {
        args: [FFIType.ptr, FFIType.f64, FFIType.f64],
        returns: FFIType.bool,
      },
    });
    return lib.symbols;
  } catch (err) {
    console.warn("Native macOS effects failed:", err);
    return null;
  }
}

const nativeSymbols = loadNativeEffects();

// --- Application menu (required for Cmd+Q, Cmd+C/V/X in frameless window) ---
ApplicationMenu.setApplicationMenu([
  {
    submenu: [
      { role: "hide" },
      { role: "hideOthers" },
      { role: "showAll" },
      { type: "separator" },
      { role: "quit" },
    ],
  },
  {
    label: "Edit",
    submenu: [
      { role: "undo" },
      { role: "redo" },
      { type: "separator" },
      { role: "cut" },
      { role: "copy" },
      { role: "paste" },
      { role: "selectAll" },
    ],
  },
]);

// --- Main window ---
const win = new BrowserWindow({
  title: "hambajuba2ba",
  frame: {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    x: 200,
    y: 50,
  },
  titleBarStyle: "hiddenInset",
  transparent: true,
  url: APP_URL,
});

// Float above other windows by default
win.setAlwaysOnTop(true);

// --- Native macOS effects (shadow, traffic lights, drag) ---
if (nativeSymbols) {
  const symbols = nativeSymbols;
  symbols.ensureWindowShadow(win.ptr);
  symbols.setTrafficLightsSubdued(win.ptr);
  symbols.setWindowAspectRatio(win.ptr, MIN_WIDTH, MIN_HEIGHT);

  const alignControls = () => {
    symbols.setWindowTrafficLightsPosition(
      win.ptr,
      TRAFFIC_LIGHTS_X,
      TRAFFIC_LIGHTS_Y,
    );
    symbols.setNativeWindowDragRegion(
      win.ptr,
      DRAG_REGION_X,
      DRAG_REGION_HEIGHT,
    );
  };

  alignControls();
  // Reposition after initial layout settles
  setTimeout(alignControls, 120);
  // Reposition on resize (traffic lights can shift)
  win.on("resize", alignControls);
}

// --- Shell bridge ---
// Tiny HTTP server so the frontend can call native APIs (e.g. pin toggle).
// RPC requires bundled views; this works with localhost dev server.
const BRIDGE_PORT = 14321;
const MAX_AGENT_MESSAGE_BYTES = 256_000;
const FRONTEND_ROUTE_TIMEOUT_MS = 120_000;
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
};

type BridgeRole = "frontend" | "mcp" | "brain";

interface BridgeSocketData {
  role: BridgeRole;
}

interface BridgeEnvelope {
  id?: string;
  type?: string;
  payload?: unknown;
  error?: { code?: string; message?: string };
}

interface PendingRoute {
  source: ServerWebSocket<BridgeSocketData>;
  sourceId: string;
  timeout: ReturnType<typeof setTimeout>;
}

const frontendRequestTypes = new Set([
  "agent.get_state",
  "agent.get_control_surface",
  "agent.get_music_window",
  "agent.get_song_analysis",
  "agent.report_phase",
  "agent.apply_visual_plan",
  "agent.set_armed",
]);
const brainBroadcastTypes = new Set([
  "brain.agent_event",
  "brain.state_update",
]);

let frontendSocket: ServerWebSocket<BridgeSocketData> | null = null;
const mcpSockets = new Set<ServerWebSocket<BridgeSocketData>>();
const brainSockets = new Set<ServerWebSocket<BridgeSocketData>>();
const pendingRoutes = new Map<string, PendingRoute>();
let brainWindow: BrowserWindow | null = null;
let activeHermesRequest: {
  id: string;
  controller: AbortController;
  directiveContext: string;
} | null = null;
let hermesQueue: Promise<void> = Promise.resolve();
let hermesQueueGeneration = 0;
const MAX_STEERING_CONTEXT_CHARS = 6_000;

function sendBridgeJson(ws: ServerWebSocket<BridgeSocketData>, payload: unknown) {
  ws.send(JSON.stringify(payload));
}

function sendBridgeError(
  ws: ServerWebSocket<BridgeSocketData>,
  id: string | undefined,
  message: string,
  code = "bridge_error",
) {
  sendBridgeJson(ws, {
    id,
    type: "error",
    error: { code, message },
  });
}

function sendBridgeResult(
  ws: ServerWebSocket<BridgeSocketData>,
  id: string,
  payload: unknown,
) {
  sendBridgeJson(ws, { id, type: "result", payload });
}

function bridgeRequestId() {
  return crypto.randomUUID();
}

function isAbortError(error: unknown) {
  if (typeof error === "string") return /abort|cancel|interrupt/i.test(error);
  if (error instanceof DOMException && error.name === "AbortError") return true;
  if (!(error instanceof Error)) return false;
  return error.name === "AbortError" || /abort|cancel|interrupt/i.test(error.message);
}

function payloadDirectiveText(payload: unknown) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return "";
  const record = payload as Record<string, unknown>;
  return typeof record.directive === "string" ? record.directive.trim() : "";
}

function combineDirectiveContext(previous: string | null | undefined, latest: string) {
  const combined = previous
    ? `${previous}\n\nThen the user added/steered:\n${latest}`
    : latest;
  if (combined.length <= MAX_STEERING_CONTEXT_CHARS) return combined;
  return `Earlier steering context was truncated.\n${combined.slice(-MAX_STEERING_CONTEXT_CHARS)}`;
}

async function runQueuedHermesDirective(
  payload: unknown,
  generation: number,
  directiveContext: string,
) {
  if (generation !== hermesQueueGeneration) {
    throw new Error("Hermes directive canceled");
  }
  const request = {
    id: bridgeRequestId(),
    controller: new AbortController(),
    directiveContext,
  };
  activeHermesRequest = request;

  try {
    return await submitDirectiveToHermes(
      payload,
      process.env,
      fetch,
      { signal: request.controller.signal },
    );
  } finally {
    if (activeHermesRequest?.id === request.id) {
      activeHermesRequest = null;
    }
  }
}

function withSteeringContext(payload: unknown, previousDirectiveContext?: string | null) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
  const record = payload as Record<string, unknown>;
  if (typeof record.directive !== "string") return payload;
  const latestDirective = record.directive.trim();
  const lines = [
    "This is additive steering/correction for the current visual plan, not an unrelated reset unless the user explicitly asks for a reset.",
    "Do not ignore earlier pending user intent; merge it with the latest steering unless the latest line clearly contradicts it.",
    previousDirectiveContext ? `Earlier pending user intent:\n${previousDirectiveContext}` : "",
    "Re-read Hamba state before applying so the revision follows whatever actually landed.",
    `Latest user steering/add-on:\n${latestDirective}`,
  ].filter(Boolean);
  return { ...record, directive: lines.join("\n\n") };
}

async function submitDirectiveWithSteeringQueue(payload: unknown) {
  const isSteeringActiveRun = Boolean(activeHermesRequest);
  const previousDirectiveContext = activeHermesRequest?.directiveContext ?? null;
  const latestDirective = payloadDirectiveText(payload);
  const directiveContext = combineDirectiveContext(previousDirectiveContext, latestDirective);
  if (isSteeringActiveRun) {
    cancelActiveHermesRequest("Canceled by newer steering directive");
    hermesQueue = Promise.resolve();
  }
  const generation = hermesQueueGeneration;
  const queuedPayload = isSteeringActiveRun
    ? withSteeringContext(payload, previousDirectiveContext)
    : payload;
  const resultPromise = hermesQueue.then(
    () => runQueuedHermesDirective(queuedPayload, generation, directiveContext),
    () => runQueuedHermesDirective(queuedPayload, generation, directiveContext),
  );
  hermesQueue = resultPromise.then(
    () => undefined,
    () => undefined,
  );
  return resultPromise;
}

function cancelActiveHermesRequest(reason = "Canceled by user") {
  hermesQueueGeneration += 1;
  if (!activeHermesRequest) return false;
  activeHermesRequest.controller.abort(reason);
  activeHermesRequest = null;
  return true;
}

function bridgeStatus() {
  return {
    frontend_connected: Boolean(frontendSocket),
    mcp_connections: mcpSockets.size,
    brain_connections: brainSockets.size,
    pending_routes: pendingRoutes.size,
    hermes_model: hermesModelName(process.env),
    hermes_endpoint: hermesResponsesUrl(process.env),
  };
}

function spawnBrainWindow() {
  if (brainWindow) {
    brainWindow.show();
    return { opened: true, created: false };
  }

  brainWindow = new BrowserWindow({
    title: "hamba brain",
    frame: {
      width: BRAIN_WIDTH,
      height: BRAIN_HEIGHT,
      x: 1340,
      y: 80,
    },
    titleBarStyle: "hiddenInset",
    transparent: true,
    styleMask: { Resizable: true },
    url: BRAIN_URL,
  });
  brainWindow.setAlwaysOnTop(true);
  const openedWindow = brainWindow;

  // Native macOS effects — shadow, traffic lights off-screen, drag region
  if (nativeSymbols) {
    const symbols = nativeSymbols;
    symbols.ensureWindowShadow(openedWindow.ptr);
    symbols.setTrafficLightsSubdued(openedWindow.ptr);
    symbols.setWindowAspectRatio(openedWindow.ptr, BRAIN_WIDTH, BRAIN_HEIGHT);
    const alignBrain = () => {
      symbols.setWindowTrafficLightsPosition(
        openedWindow.ptr,
        TRAFFIC_LIGHTS_X,
        TRAFFIC_LIGHTS_Y,
      );
      symbols.setNativeWindowDragRegion(
        openedWindow.ptr,
        0,
        BRAIN_DRAG_HEIGHT,
      );
    };
    alignBrain();
    setTimeout(alignBrain, 120);
    openedWindow.on("resize", alignBrain);
  }

  openedWindow.on("close", () => {
    if (brainWindow === openedWindow) brainWindow = null;
  });
  return { opened: true, created: true };
}

function parseBridgeEnvelope(raw: string): BridgeEnvelope {
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Bridge message must be a JSON object");
  }
  return parsed as BridgeEnvelope;
}

function routeBridgeResponse(message: BridgeEnvelope) {
  if (!message.id) return;
  const route = pendingRoutes.get(message.id);
  if (!route) return;
  pendingRoutes.delete(message.id);
  clearTimeout(route.timeout);
  sendBridgeJson(route.source, {
    ...message,
    id: route.sourceId,
  });
}

function failPendingFrontendRoutes(message: string) {
  for (const route of pendingRoutes.values()) {
    clearTimeout(route.timeout);
    sendBridgeError(route.source, route.sourceId, message, "frontend_unavailable");
  }
  pendingRoutes.clear();
}

function broadcastToBrain(message: BridgeEnvelope) {
  if (!message.type || !brainBroadcastTypes.has(message.type)) return 0;
  const payload = {
    type: message.type,
    payload: message.payload,
  };
  let delivered = 0;
  for (const brainSocket of brainSockets) {
    sendBridgeJson(brainSocket, payload);
    delivered += 1;
  }
  return delivered;
}

function forwardToFrontend(
  source: ServerWebSocket<BridgeSocketData>,
  message: BridgeEnvelope,
) {
  if (!message.id || !message.type) {
    sendBridgeError(source, message.id, "Bridge request needs id and type", "bad_request");
    return;
  }
  if (!frontendSocket) {
    sendBridgeError(
      source,
      message.id,
      "Main Hamba frontend is not connected to the desktop bridge yet",
      "frontend_unavailable",
    );
    return;
  }
  if (!frontendRequestTypes.has(message.type)) {
    sendBridgeError(source, message.id, `Unknown frontend request: ${message.type}`, "unknown_type");
    return;
  }

  const forwardedId = bridgeRequestId();
  const timeout = setTimeout(() => {
    const route = pendingRoutes.get(forwardedId);
    if (!route) return;
    pendingRoutes.delete(forwardedId);
    sendBridgeError(
      route.source,
      route.sourceId,
      `Frontend request timed out: ${message.type}`,
      "route_timeout",
    );
  }, FRONTEND_ROUTE_TIMEOUT_MS);
  pendingRoutes.set(forwardedId, { source, sourceId: message.id, timeout });
  sendBridgeJson(frontendSocket, {
    id: forwardedId,
    type: message.type,
    payload: message.payload,
  });
}

async function handleFrontendRequest(
  ws: ServerWebSocket<BridgeSocketData>,
  message: BridgeEnvelope,
) {
  if (message.type && brainBroadcastTypes.has(message.type)) {
    const delivered = broadcastToBrain(message);
    if (message.id) {
      sendBridgeResult(ws, message.id, { accepted: true, delivered });
    }
    return;
  }

  if (!message.id || !message.type) {
    sendBridgeError(ws, message.id, "Bridge request needs id and type", "bad_request");
    return;
  }
  if (message.type !== "agent.submit_directive") {
    sendBridgeError(ws, message.id, `Unknown bridge request: ${message.type}`, "unknown_type");
    return;
  }

  try {
    const result = await submitDirectiveWithSteeringQueue(message.payload);
    sendBridgeResult(ws, message.id, result);
  } catch (err) {
    sendBridgeError(
      ws,
      message.id,
      isAbortError(err) ? "Hermes directive canceled" : err instanceof Error ? err.message : String(err),
      isAbortError(err) ? "hermes_canceled" : "hermes_error",
    );
  }
}

async function handleBrainRequest(
  ws: ServerWebSocket<BridgeSocketData>,
  message: BridgeEnvelope,
) {
  if (!message.id || !message.type) {
    sendBridgeError(ws, message.id, "Brain request needs id and type", "bad_request");
    return;
  }

  switch (message.type) {
    case "brain.get_bridge_status":
      sendBridgeResult(ws, message.id, bridgeStatus());
      return;
    case "brain.get_state":
      forwardToFrontend(ws, {
        ...message,
        type: "agent.get_state",
      });
      return;
    case "brain.set_armed":
      forwardToFrontend(ws, {
        ...message,
        type: "agent.set_armed",
      });
      return;
    case "brain.submit_directive":
      try {
        const result = await submitDirectiveWithSteeringQueue(message.payload);
        sendBridgeResult(ws, message.id, result);
      } catch (err) {
        sendBridgeError(
          ws,
          message.id,
          isAbortError(err) ? "Hermes directive canceled" : err instanceof Error ? err.message : String(err),
          isAbortError(err) ? "hermes_canceled" : "hermes_error",
        );
      }
      return;
    case "brain.cancel_directive": {
      const canceled = cancelActiveHermesRequest();
      sendBridgeResult(ws, message.id, { accepted: true, canceled });
      return;
    }
    default:
      sendBridgeError(ws, message.id, `Unknown brain request: ${message.type}`, "unknown_type");
  }
}

async function handleBridgeMessage(
  ws: ServerWebSocket<BridgeSocketData>,
  rawMessage: string | Buffer,
) {
  if (typeof rawMessage !== "string") {
    sendBridgeError(ws, undefined, "Bridge only accepts JSON text messages", "bad_request");
    return;
  }
  if (rawMessage.length > MAX_AGENT_MESSAGE_BYTES) {
    sendBridgeError(ws, undefined, "Bridge message is too large", "message_too_large");
    return;
  }

  let message: BridgeEnvelope;
  try {
    message = parseBridgeEnvelope(rawMessage);
  } catch (err) {
    sendBridgeError(
      ws,
      undefined,
      err instanceof Error ? err.message : String(err),
      "bad_json",
    );
    return;
  }

  if (message.type === "result" || message.type === "error") {
    routeBridgeResponse(message);
    return;
  }

  if (ws.data.role === "mcp") {
    forwardToFrontend(ws, message);
    return;
  }

  if (ws.data.role === "brain") {
    await handleBrainRequest(ws, message);
    return;
  }

  await handleFrontendRequest(ws, message);
}

Bun.serve<BridgeSocketData>({
  port: BRIDGE_PORT,
  fetch(req, server) {
    if (req.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(req.url);

    if (url.pathname === "/agent/ws") {
      const role = url.searchParams.get("role");
      if (role !== "frontend" && role !== "mcp" && role !== "brain") {
        return new Response("Invalid bridge role", { status: 400, headers: corsHeaders });
      }
      if (server.upgrade(req, { data: { role } })) {
        return;
      }
      return new Response("WebSocket upgrade failed", { status: 500, headers: corsHeaders });
    }

    if (url.pathname === "/pin" && req.method === "POST") {
      const pinned = !win.isAlwaysOnTop();
      win.setAlwaysOnTop(pinned);
      return Response.json({ pinned }, { headers: corsHeaders });
    }

    if (url.pathname === "/pin" && req.method === "GET") {
      return Response.json({ pinned: win.isAlwaysOnTop() }, { headers: corsHeaders });
    }

    if (url.pathname === "/brain/spawn" && req.method === "POST") {
      // Toggle behavior — open if closed, close if open. Single endpoint, single click.
      if (brainWindow) {
        const w = brainWindow;
        brainWindow = null;  // clear immediately so quick re-clicks don't double-fire
        w.close();
        return Response.json({ open: false, closed: true }, { headers: corsHeaders });
      }
      const result = spawnBrainWindow();
      return Response.json({ open: true, ...result }, { headers: corsHeaders });
    }

    if (url.pathname === "/brain/status" && req.method === "GET") {
      return Response.json({ open: brainWindow !== null }, { headers: corsHeaders });
    }

    return new Response("Not found", { status: 404, headers: corsHeaders });
  },
  websocket: {
    open(ws) {
      if (ws.data.role === "frontend") {
        if (frontendSocket && frontendSocket !== ws) {
          frontendSocket.close(1000, "Replaced by a newer frontend bridge");
        }
        frontendSocket = ws;
      } else if (ws.data.role === "mcp") {
        mcpSockets.add(ws);
      } else {
        brainSockets.add(ws);
      }
      console.info(`Agent bridge connected: ${ws.data.role}`, bridgeStatus());
    },
    message(ws, message) {
      void handleBridgeMessage(ws, message);
    },
    close(ws) {
      if (frontendSocket === ws) {
        frontendSocket = null;
        failPendingFrontendRoutes("Hamba frontend bridge disconnected");
      }
      mcpSockets.delete(ws);
      brainSockets.delete(ws);
      console.info(`Agent bridge disconnected: ${ws.data.role}`, bridgeStatus());
      for (const [id, route] of pendingRoutes) {
        if (route.source === ws) {
          clearTimeout(route.timeout);
          pendingRoutes.delete(id);
        }
      }
    },
  },
});

// Aspect ratio + minimum size enforced natively via setWindowAspectRatio
// (NSWindow.contentAspectRatio + minSize). No JS snap-back needed.
