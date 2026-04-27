/**
 * FaceRenderer — The canvas for hamba's face.
 *
 * Owns the textmode.js WebGL2 lifecycle. Creates a terminal-style ASCII
 * canvas, sets up a 30fps draw loop, and renders the face state produced
 * by useFaceAnimation each frame.
 *
 * Layered concentric ellipses for eyes (dim outline → sclera → iris → pupil).
 * Orange-slice ellipse mouth — abstract, not a literal mouth.
 * The face literally glows via the .face-glow CSS filter.
 */

import { useRef, useEffect, useCallback } from "react";
import { textmode } from "textmode.js";
import {
  useFaceAnimation,
  type FaceState,
  type EyeShape,
  type MouthShape,
} from "../../hooks/useFaceAnimation";

// =============================================================================
// CONSTANTS
// =============================================================================

const FONT_SIZE = 6;
const FRAME_RATE = 30;

// Phosphor green palette — 4 brightness tiers of hsl(147°, ~55%, ~%)
const PALETTE = {
  bright: [74, 222, 128] as const, // Pupil, highlights
  mid: [50, 160, 90] as const, // Sclera, iris, mouth fill
  dim: [30, 100, 60] as const, // Eye outline, contemplative
  subtle: [15, 50, 30] as const, // Faint ambient glow
};

// Glitch characters for disconnect effect
const GLITCH_CHARS = "░▒▓█╱╲─│┌┐└┘├┤┬┴┼";

// Sparkle characters for pet bliss — scattered bright twinkles
const SPARKLE_CHARS = "✦✧*·.+";
// Fixed sparkle positions (pre-computed so they don't jump every frame)
const SPARKLE_SLOTS = [
  { x: -18, y: -6 }, { x: 19, y: -5 }, { x: -8, y: -8 },
  { x: 10, y: -7 }, { x: -22, y: 1 }, { x: 23, y: 0 },
  { x: -15, y: 5 }, { x: 16, y: 6 }, { x: 0, y: -9 },
  { x: -5, y: 7 }, { x: 7, y: 8 }, { x: -20, y: -3 },
];

// Eyes — concentric ellipses (absolute cell sizes)
const EYE_W = 9; // normal eye width
const EYE_H = 7; // normal eye height
const EYE_WIDE_EXTRA = 2; // extra rows for surprised/wide
const EYE_SEP = 28; // center-to-center separation
const EYE_Y = -1; // eyes slightly above center

// Mouth — orange-slice ellipse
const MOUTH_W = 10; // mouth width
const MOUTH_Y = 4; // mouth below center

// =============================================================================
// DRAWING HELPERS
// =============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TM = any; // textmode instance

/** Apply a palette color scaled by brightness. */
function setColor(
  t: TM,
  color: readonly [number, number, number],
  brightness: number,
) {
  const b = Math.min(brightness, 1.2);
  t.charColor(
    Math.round(color[0] * b),
    Math.round(color[1] * b),
    Math.round(color[2] * b),
  );
}

// =============================================================================
// EYE RENDERING — Layered concentric ellipses
// =============================================================================

/**
 * Draw one eye at (cx, cy) using layered ellipses.
 *
 * 4 layers: dim ░ outline → mid ▒ sclera → ▓ iris → bright █ pupil.
 * Iris + pupil shift with lookX/lookY. Pupil scales with dilation.
 * Blink = thin line. Half = lower portion only, no pupil detail.
 */
function drawEye(
  t: TM,
  cx: number,
  cy: number,
  shape: EyeShape,
  lookX: number,
  lookY: number,
  dilation: number,
  brightness: number,
) {
  if (shape === "hidden") return;

  t.push();
  t.translate(cx, cy);

  // ---- Blink: thin line ----
  if (shape === "closed") {
    setColor(t, PALETTE.mid, brightness);
    t.char("▓");
    t.ellipse(EYE_W, 1);
    t.pop();
    return;
  }

  // ---- Happy squish: ^.^ angled chevron with layered texture ----
  if (shape === "happy") {
    const hw = Math.floor(EYE_W / 2); // half-width

    // Helper: draw a ^ chevron at given offset (for layering)
    const drawChevron = (
      char: string,
      color: readonly [number, number, number],
      b: number,
      offsetY: number,
    ) => {
      setColor(t, color, b);
      t.char(char);
      // Left leg
      for (let i = 0; i <= hw; i++) {
        t.push();
        t.translate(-hw + i, -i + offsetY);
        t.point();
        t.pop();
      }
      // Right leg
      for (let i = 0; i <= hw; i++) {
        t.push();
        t.translate(i, -hw + i + offsetY);
        t.point();
        t.pop();
      }
    };

    // Layer 1: Dim outer glow (░) — shifted down 1 for depth
    drawChevron("░", PALETTE.dim, brightness, 1);
    // Layer 2: Mid body (▒) — the main chevron
    drawChevron("▒", PALETTE.mid, brightness, 0);
    // Layer 3: Bright core (▓) — shifted up 1, the bright ridge
    drawChevron("▓", PALETTE.bright, brightness, -1);

    t.pop();
    return;
  }

  // ---- Determine eye height from shape ----
  let h = EYE_H;
  let yShift = 0;
  if (shape === "half") {
    h = Math.max(3, Math.floor(EYE_H * 0.5));
    yShift = Math.floor(EYE_H * 0.25);
  } else if (shape === "wide") {
    h = EYE_H + EYE_WIDE_EXTRA;
  }

  t.push();
  t.translate(0, yShift);

  // Layer 1: Outline (dim ░)
  setColor(t, PALETTE.dim, brightness);
  t.char("░");
  t.ellipse(EYE_W, h);

  // Layer 2: Sclera (mid ▒)
  const scleraB = shape === "wide" ? brightness * 1.15 : brightness;
  setColor(t, PALETTE.mid, scleraB);
  t.char("▒");
  t.ellipse(Math.max(1, EYE_W - 2), Math.max(1, h - 2));

  // Half-closed: no iris/pupil detail (too sleepy)
  if (shape === "half") {
    t.pop(); // yShift
    t.pop(); // cx, cy
    return;
  }

  // ---- Iris + Pupil (shift with look direction) ----
  const maxOffX = Math.max(0, Math.floor(EYE_W / 2 - 2));
  const maxOffY = Math.max(0, Math.floor(h / 2 - 1));
  const offX = Math.round(lookX * maxOffX);
  const offY = Math.round(lookY * maxOffY);

  // Pupil radius from dilation (0→pinpoint, 1→dilated)
  const pupilR = Math.max(1, Math.round(dilation * 2.5));
  const irisR = pupilR + 1;

  t.push();
  t.translate(offX, offY);

  // Layer 3: Iris (▓, mid-bright)
  setColor(t, PALETTE.mid, brightness * 1.2);
  t.char("▓");
  t.ellipse(irisR * 2, irisR * 2);

  // Layer 4: Pupil (█, bright)
  setColor(t, PALETTE.bright, brightness);
  t.char("█");
  t.ellipse(pupilR * 2, pupilR * 2);

  t.pop(); // pupil offset
  t.pop(); // yShift
  t.pop(); // cx, cy
}

// =============================================================================
// MOUTH RENDERING — Orange-slice ellipses
// =============================================================================

/**
 * Draw the mouth at (0, cy). Abstract orange-slice shape — a wide, short
 * ellipse that reads as a curved mouth without being literal.
 */
function drawMouth(
  t: TM,
  cy: number,
  shape: MouthShape,
  openness: number,
  brightness: number,
) {
  if (shape === "hidden") return;

  t.push();
  t.translate(0, cy);

  switch (shape) {
    case "neutral":
      // Subtle thin orange slice
      setColor(t, PALETTE.dim, brightness);
      t.char("▓");
      t.ellipse(MOUTH_W, 2);
      break;

    case "smile":
      // Wider, brighter orange slice
      setColor(t, PALETTE.mid, brightness);
      t.char("▓");
      t.ellipse(MOUTH_W, 3);
      break;

    case "open": {
      // Surprise — rounder O (outer shell + dark void)
      const h = Math.max(2, Math.round(openness * 5));
      setColor(t, PALETTE.mid, brightness);
      t.char("▓");
      t.ellipse(MOUTH_W, h + 2);
      t.charColor(0, 0, 0);
      t.char("█");
      t.ellipse(Math.max(1, MOUTH_W - 2), Math.max(1, h));
      break;
    }

    case "excited": {
      // Wide ellipse — the drop, the climax
      const h = Math.max(3, Math.round(openness * 6));
      setColor(t, PALETTE.mid, brightness);
      t.char("▓");
      t.ellipse(MOUTH_W + 2, h);
      setColor(t, PALETTE.bright, brightness);
      t.char("█");
      t.ellipse(Math.max(1, MOUTH_W), Math.max(1, h - 2));
      break;
    }

    case "contemplative":
      // Small dim crescent — pensive
      setColor(t, PALETTE.subtle, brightness);
      t.char("▓");
      t.ellipse(Math.max(4, MOUTH_W - 3), 2);
      break;
  }

  t.pop();
}

// =============================================================================
// FACE ORCHESTRATOR
// =============================================================================

/** Draw the complete face from a FaceState snapshot. */
let _debugged = false;
function drawFace(t: TM, state: FaceState) {
  const grid = t.grid;
  if (!grid) return;
  const { cols, rows } = grid;

  if (!_debugged) {
    _debugged = true;
    console.log(
      "[FaceRenderer] grid:",
      cols,
      "×",
      rows,
      "| EYE:",
      EYE_W,
      "×",
      EYE_H,
      "| sep:",
      EYE_SEP,
      "| MOUTH_W:",
      MOUTH_W,
    );
  }

  // ---- Glitch: scatter random chars across entire grid ----
  if (state.glitch) {
    const count = Math.floor(cols * rows * 0.4);
    for (let i = 0; i < count; i++) {
      const x = Math.floor(Math.random() * cols) - Math.floor(cols / 2);
      const y = Math.floor(Math.random() * rows) - Math.floor(rows / 2);
      t.push();
      t.translate(x, y);
      setColor(t, PALETTE.bright, Math.random() * state.brightness);
      t.char(GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)]);
      t.point();
      t.pop();
    }
    return;
  }

  // ---- Eyes: concentric ellipses at ±EYE_SEP/2 ----
  const eyeX = Math.round(EYE_SEP / 2);

  drawEye(
    t,
    -eyeX,
    EYE_Y,
    state.leftEye,
    state.lookX,
    state.lookY,
    state.pupilDilation,
    state.brightness,
  );
  drawEye(
    t,
    eyeX,
    EYE_Y,
    state.rightEye,
    state.lookX,
    state.lookY,
    state.pupilDilation,
    state.brightness,
  );

  // ---- Mouth: orange-slice ellipse centered below eyes ----
  drawMouth(t, MOUTH_Y, state.mouth, state.mouthOpenness, state.brightness);

  // ---- Sparkles: scattered twinkles during pet interaction ----
  if (state.sparkle && state.sparkle > 0) {
    const sparkleCount = Math.ceil(state.sparkle * SPARKLE_SLOTS.length);
    const time = performance.now() / 1000;
    for (let i = 0; i < sparkleCount; i++) {
      const slot = SPARKLE_SLOTS[i];
      // Each sparkle twinkles at its own phase (staggered by index)
      const twinkle = Math.sin(time * 3 + i * 2.1) * 0.5 + 0.5;
      if (twinkle < 0.3) continue; // off phase — sparkle is hidden
      const charIdx = (i + Math.floor(time * 2)) % SPARKLE_CHARS.length;
      t.push();
      t.translate(slot.x, slot.y);
      setColor(t, PALETTE.bright, state.brightness * twinkle);
      t.char(SPARKLE_CHARS[charIdx]);
      t.point();
      t.pop();
    }
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

export function FaceRenderer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { tick, petRef } = useFaceAnimation();
  const lastMousePos = useRef({ x: 0, y: 0 });

  // Pet interaction — write mouse data to shared ref (no re-renders)
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;

      // Movement speed (EMA-smoothed)
      const dx = e.clientX - lastMousePos.current.x;
      const dy = e.clientY - lastMousePos.current.y;
      const speed = Math.sqrt(dx * dx + dy * dy);
      lastMousePos.current = { x: e.clientX, y: e.clientY };

      // Pet zone: central 40% × 50% of face
      const inZone =
        nx >= 0.3 && nx <= 0.7 && ny >= 0.25 && ny <= 0.75;

      const p = petRef.current;
      p.normalizedX = nx;
      p.normalizedY = ny;
      p.inZone = inZone;
      p.moveSpeed = p.moveSpeed * 0.7 + speed * 0.3;
      p.lastMoveTime = performance.now();

      // Cursor hint for discoverability
      container.style.cursor = inZone ? "grab" : "";
    },
    [petRef],
  );

  const handleMouseLeave = useCallback(() => {
    petRef.current.inZone = false;
    petRef.current.moveSpeed = 0;
    if (containerRef.current) containerRef.current.style.cursor = "";
  }, [petRef]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Suppress textmode.js built-in filter shader errors (invert, grayscale,
    // sepia, threshold). These filters compile shaders we never use — the
    // geometry shaders for rect/ellipse/point are separate and work fine.
    const suppressFilterShaderError = (e: PromiseRejectionEvent) => {
      if (e.reason?.message?.includes("Shader compilation error")) {
        e.preventDefault();
      }
    };
    window.addEventListener("unhandledrejection", suppressFilterShaderError);

    // Clean stale canvases (React Strict Mode double-mount race)
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Create textmode instance sized to container
    const rect = container.getBoundingClientRect();
    const t = textmode.create({
      width: rect.width || 300,
      height: rect.height || 120,
      fontSize: FONT_SIZE,
      frameRate: FRAME_RATE,
    });
    container.appendChild(t.canvas);

    // Style the canvas to fill the container
    t.canvas.style.width = "100%";
    t.canvas.style.height = "100%";
    t.canvas.style.display = "block";

    // Resize when container changes
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        t.resizeCanvas(width, height);
      }
    });
    observer.observe(container);

    // 30fps draw loop — tick the state machine, clear, render
    t.draw(() => {
      const state = tick(t.secs);
      t.background(0);
      drawFace(t, state);
    });

    return () => {
      window.removeEventListener("unhandledrejection", suppressFilterShaderError);
      observer.disconnect();
      t.destroy();
      // Remove canvas from DOM to free WebGL context
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    };
  }, [tick]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full face-glow"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    />
  );
}
