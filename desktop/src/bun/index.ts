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

// --- Constants ---
const MIN_WIDTH = 800;
const MIN_HEIGHT = 800;
const DEFAULT_WIDTH = 1600;
const DEFAULT_HEIGHT = 1600;

const APP_URL = "http://localhost:3000?desktop=true";

// Traffic lights hidden — moved off-screen so screws can use the corners.
// Close: Cmd+Q. Minimize: Cmd+M. Zoom: not needed (1:1 ratio enforced).
const TRAFFIC_LIGHTS_X = -100;
const TRAFFIC_LIGHTS_Y = -100;
const DRAG_REGION_X = 0;   // full width drag region (no traffic lights to dodge)
const DRAG_REGION_HEIGHT = 50;

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
if (process.platform === "darwin") {
  const dylibPath = join(import.meta.dir, "libMacWindowEffects.dylib");

  if (existsSync(dylibPath)) {
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

      const { symbols } = lib;
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
    } catch (err) {
      console.warn("Native macOS effects failed:", err);
    }
  }
}

// --- Shell bridge ---
// Tiny HTTP server so the frontend can call native APIs (e.g. pin toggle).
// RPC requires bundled views; this works with localhost dev server.
const BRIDGE_PORT = 14321;
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
};

Bun.serve({
  port: BRIDGE_PORT,
  fetch(req) {
    if (req.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(req.url);

    if (url.pathname === "/pin" && req.method === "POST") {
      const pinned = !win.isAlwaysOnTop();
      win.setAlwaysOnTop(pinned);
      return Response.json({ pinned }, { headers: corsHeaders });
    }

    if (url.pathname === "/pin" && req.method === "GET") {
      return Response.json({ pinned: win.isAlwaysOnTop() }, { headers: corsHeaders });
    }

    return new Response("Not found", { status: 404, headers: corsHeaders });
  },
});

// Aspect ratio + minimum size enforced natively via setWindowAspectRatio
// (NSWindow.contentAspectRatio + minSize). No JS snap-back needed.
