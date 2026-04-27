/**
 * PlantStems - Block-Centric Waveform Stem Visualization
 *
 * Each BLOCK ORB gets its own stem/tendril that reacts to
 * whatever audio stem is assigned to that block.
 *
 * Example: If block "down.2.1" has "drums" assigned, the tendril
 * from that orb reacts to drums audio data.
 */

import { useRef, useEffect, useCallback } from 'react';
import { createNoise3D } from 'simplex-noise';
import { useAudioActivityStore } from '../stores/useAudioActivityStore';
import { useDestinationStore } from '../stores/useDestinationStore';
import type { BlockCode, BlockMapping, StemChannelData, LinkTarget } from '../types/sae';
import type Matter from 'matter-js';
import { BLOCK_COLORS, STEM_COLORS as STEM_COLORS_HEX } from '../data/features';

const noise3D = createNoise3D();

// ============================================================================
// Types
// ============================================================================

interface Position {
  x: number;
  y: number;
}

interface PlantStemsProps {
  /** Block mappings - tells us which stem each block uses */
  blockMappings: Record<BlockCode, BlockMapping>;
  /** Matter.js bodies for block orbs */
  stemOrbBodies: Matter.Body[];
  /** Matter.js bodies for destination orbs (latent/prompt) */
  destinationOrbBodies: Matter.Body[];
  /** Matter.js body for heart (read position live in animation loop) */
  heartBody: Matter.Body;
  /** Whether latent destinations are configured */
  latentConfigured: boolean;
  /** Whether prompt destinations are configured */
  promptConfigured: boolean;
  /** Canvas width */
  width: number;
  /** Canvas height */
  height: number;
}

// ============================================================================
// Constants
// ============================================================================

/** Block order matches the orb body array */
const BLOCK_ORDER: BlockCode[] = ['down.2.1', 'mid.0', 'up.0.0', 'up.0.1'];

/** Convert hex color string to RGB components */
function hexToRgbObj(hex: string): { r: number; g: number; b: number } {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 0xff, g: (n >> 8) & 0xff, b: n & 0xff };
}

/** Colors by physical stem — derived from hex source of truth in features.ts */
const STEM_COLORS: Record<string, { r: number; g: number; b: number }> = {
  ...Object.fromEntries(
    Object.entries(STEM_COLORS_HEX).map(([k, v]) => [k, hexToRgbObj(v)])
  ),
  // Destination colors (not in features.ts)
  latent: { r: 138, g: 106, b: 170 },
  prompt: { r: 170, g: 138, b: 106 },
};

const SEGMENTS = 32;

// Pulse tracking
interface Pulse {
  progress: number;
  intensity: number;
}

// ============================================================================
// Helpers
// ============================================================================

/** Get physical stem from link target (drums_harmonic → drums) */
function getPhysicalStem(linkTarget: LinkTarget | undefined): string {
  if (!linkTarget) return 'other';
  // Derived targets
  if (linkTarget === 'tension' || linkTarget === 'global') return 'other';
  // Extract base stem from compound names
  for (const base of ['drums', 'vocals', 'bass', 'other']) {
    if (linkTarget.startsWith(base)) return base;
  }
  return 'other';
}

/** Cubic bezier stem path: drops straight down from flower, curves to heart */
function interpolateStemPath(flower: Position, heart: Position, segments: number): Position[] {
  // cp1: straight down from flower (vertical stem exit from calyx)
  const dropDistance = 60;
  const cp1: Position = { x: flower.x, y: flower.y + dropDistance };
  // cp2: approach heart from above
  const cp2: Position = { x: heart.x, y: heart.y - 30 };

  const points: Position[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    const t2 = t * t;
    const t3 = t2 * t;
    points.push({
      x: mt3 * flower.x + 3 * mt2 * t * cp1.x + 3 * mt * t2 * cp2.x + t3 * heart.x,
      y: mt3 * flower.y + 3 * mt2 * t * cp1.y + 3 * mt * t2 * cp2.y + t3 * heart.y,
    });
  }
  return points;
}

/** Calculate perpendicular normals at each point */
function calculateNormals(points: Position[]): { nx: number; ny: number }[] {
  const normals: { nx: number; ny: number }[] = [];
  for (let i = 0; i < points.length; i++) {
    let dx: number, dy: number;
    if (i === 0) {
      dx = points[1].x - points[0].x;
      dy = points[1].y - points[0].y;
    } else if (i === points.length - 1) {
      dx = points[i].x - points[i - 1].x;
      dy = points[i].y - points[i - 1].y;
    } else {
      dx = points[i + 1].x - points[i - 1].x;
      dy = points[i + 1].y - points[i - 1].y;
    }
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    normals.push({ nx: -dy / len, ny: dx / len });
  }
  return normals;
}

/** Draw smooth path through points */
function drawSmoothPath(ctx: CanvasRenderingContext2D, points: Position[]): void {
  if (points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length - 1; i++) {
    const xc = (points[i].x + points[i + 1].x) / 2;
    const yc = (points[i].y + points[i + 1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  const last = points[points.length - 1];
  ctx.lineTo(last.x, last.y);
  ctx.stroke();
}


// ============================================================================
// Main Component
// ============================================================================

export function PlantStems({
  blockMappings,
  stemOrbBodies,
  destinationOrbBodies,
  heartBody,
  latentConfigured,
  promptConfigured,
  width,
  height,
}: PlantStemsProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const animationRef = useRef<number>(0);

  // Refs for props that change frequently — read imperatively in rAF loop
  const blockMappingsRef = useRef(blockMappings);
  const latentConfiguredRef = useRef(latentConfigured);
  const promptConfiguredRef = useRef(promptConfigured);
  useEffect(() => { blockMappingsRef.current = blockMappings; }, [blockMappings]);
  useEffect(() => { latentConfiguredRef.current = latentConfigured; }, [latentConfigured]);
  useEffect(() => { promptConfiguredRef.current = promptConfigured; }, [promptConfigured]);

  // Track pulses per block + destinations
  const pulsesRef = useRef<Record<string, Pulse[]>>({
    'down.2.1': [],
    'mid.0': [],
    'up.0.0': [],
    'up.0.1': [],
    'latent': [],
    'prompt': [],
  });

  // Track opacity per block for smooth fade in/out on toggle
  const opacityRef = useRef<Record<string, number>>({});

  // Animation loop — reads props from refs to avoid callback recreation
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      animationRef.current = requestAnimationFrame(animate);
      return;
    }

    // Cache 2D context on first use
    if (!ctxRef.current) {
      ctxRef.current = canvas.getContext('2d');
    }
    const ctx = ctxRef.current;
    if (!ctx) {
      animationRef.current = requestAnimationFrame(animate);
      return;
    }

    // Get current audio data
    const audioStems = useAudioActivityStore.getState().stems;
    const blockActivity = useAudioActivityStore.getState().blocks;
    const time = performance.now() / 1000;

    // Read heart position LIVE from Matter.js body (not from closure)
    const heartPosition = heartBody?.position || { x: width / 2, y: height / 2 };

    // Read props from refs (imperative, no dependency chain)
    const mappings = blockMappingsRef.current;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw stem for each block (with smooth opacity transitions on toggle)
    BLOCK_ORDER.forEach((blockCode, index) => {
      const mapping = mappings[blockCode];
      if (!mapping) return;

      const body = stemOrbBodies[index];
      if (!body) return;

      // Smooth opacity transition (fade in/out on enable/disable)
      const targetOpacity = mapping.enabled ? 1.0 : 0.0;
      const currentOpacity = opacityRef.current[blockCode] ?? 0.0;
      const fadeSpeed = targetOpacity > currentOpacity ? 0.08 : 0.04;
      const newOpacity = currentOpacity + (targetOpacity - currentOpacity) * fadeSpeed;
      opacityRef.current[blockCode] = newOpacity;

      // Skip drawing if fully invisible
      if (newOpacity < 0.01) return;

      // Attach stem to the bottom of the flower (not the center).
      // Offset downward by the orb radius so it emerges from the base.
      const flowerPos = { x: body.position.x, y: body.position.y + 28 };
      const linkTarget = mapping.linkTarget;
      const physicalStem = getPhysicalStem(linkTarget);

      // Get audio data for the link target
      const audioData = audioStems[linkTarget as keyof typeof audioStems]
        || audioStems[physicalStem as keyof typeof audioStems];

      const blockHex = BLOCK_COLORS[blockCode] || '#888888';
      const color = hexToRgbObj(blockHex);

      // Draw with opacity
      ctx.save();
      ctx.globalAlpha = newOpacity;

      drawPlantStem(
        ctx,
        flowerPos,
        heartPosition,
        color,
        audioData,
        blockActivity?.[blockCode]?.physics,
        time,
        pulsesRef.current[blockCode],
        index * 7.3  // unique noise seed per stem
      );

      ctx.restore();

      // Trigger pulses on transients (only when visible enough)
      if (newOpacity > 0.5 && audioData && audioData.flash > 0.5) {
        const pulses = pulsesRef.current[blockCode];
        if (pulses.length === 0 || pulses[pulses.length - 1].progress > 0.15) {
          pulses.push({ progress: 0, intensity: audioData.flash });
        }
      }

      // Update pulses
      const pulses = pulsesRef.current[blockCode];
      for (let i = pulses.length - 1; i >= 0; i--) {
        pulses[i].progress += 0.025;
        pulses[i].intensity *= 0.97;
        if (pulses[i].progress > 1 || pulses[i].intensity < 0.05) {
          pulses.splice(i, 1);
        }
      }
    });

    // Draw destination stems (read configured state from refs)
    if (latentConfiguredRef.current) {
      const latentBlend = useDestinationStore.getState().latent.blendPosition;
      const latentBody = destinationOrbBodies[0];
      if (latentBody) {
        const latentBase = { x: latentBody.position.x, y: latentBody.position.y + 28 };
        drawDestinationStem(ctx, latentBase, heartPosition, STEM_COLORS.latent, latentBlend, time);
      }
    }
    if (promptConfiguredRef.current) {
      const promptBlend = useDestinationStore.getState().prompt.blendPosition;
      const promptBody = destinationOrbBodies[1];
      if (promptBody) {
        const promptBase = { x: promptBody.position.x, y: promptBody.position.y + 28 };
        drawDestinationStem(ctx, promptBase, heartPosition, STEM_COLORS.prompt, promptBlend, time);
      }
    }

    animationRef.current = requestAnimationFrame(animate);
  }, [stemOrbBodies, destinationOrbBodies, heartBody, width, height]);

  // Start animation loop
  useEffect(() => {
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="plant-stems-canvas"
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 35,
      }}
    />
  );
}

// ============================================================================
// Stem Drawing - Audio Reactive
// ============================================================================

function drawPlantStem(
  ctx: CanvasRenderingContext2D,
  flower: Position,
  heart: Position,
  color: { r: number; g: number; b: number },
  audioData: StemChannelData | undefined,
  physicsValue: number | undefined,
  time: number,
  pulses: Pulse[],
  stemSeed: number = 0
): void {
  const basePath = interpolateStemPath(flower, heart, SEGMENTS);
  const normals = calculateNormals(basePath);

  const energy = audioData?.energy_smooth ?? 0.2;
  const physics = physicsValue ?? energy;
  const envelope = audioData?.envelope ?? 0.2;
  const brightness = audioData?.brightness ?? 0.5;
  const flux = audioData?.flux ?? 0;

  const noiseTime = time * (0.6 + flux * 0.8);

  const waveformPath: Position[] = basePath.map((point, i) => {
    const t = i / SEGMENTS;
    const rootFade = Math.min(1, Math.max(0, (t - 0.10) / 0.2));
    const positionFade = rootFade * (1 - t * 0.5);

    const waveformAmp = envelope * 45 * positionFade;

    // Simplex noise at two octaves
    const n1 = noise3D(t * 3, noiseTime, stemSeed) * 10;
    const n2 = noise3D(t * 7, noiseTime * 1.4, stemSeed + 10) * 5;
    const organicFlow = (n1 + n2) * positionFade;

    // Energy + brightness reactive waves
    const energyNoise = noise3D(t * 5, time * 2, stemSeed + 20);
    const energyWave = (Math.sin(t * 6 * Math.PI + time * 4) + energyNoise * 0.6) * 12 * physics;
    const brightnessWave = Math.sin(t * (4 + brightness * 8) * Math.PI + time * 3) * 5 * physics;

    const totalDisplacement = waveformAmp + organicFlow + energyWave + brightnessWave;

    return {
      x: point.x + normals[i].nx * totalDisplacement,
      y: point.y + normals[i].ny * totalDisplacement,
    };
  });
  waveformPath[0] = { ...flower };
  waveformPath[waveformPath.length - 1] = { ...heart };

  // Tapered half-widths: thick at orb, thin at heart
  const baseWidth = 3 + physics * 5;
  const halfWidths = waveformPath.map((_, i) => {
    const t = i / SEGMENTS;
    const taper = 1 - t * 0.75;
    return baseWidth * taper * 0.5;
  });

  // === LAYER 1: Glow ===
  ctx.save();
  ctx.filter = 'blur(6px)';
  const glowAlpha = 0.15 + physics * 0.2;
  ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${glowAlpha})`;
  drawTaperedShape(ctx, waveformPath, normals, halfWidths, 2.5);
  ctx.restore();

  // === LAYER 2: Main stem ===
  const mainAlpha = 0.6 + physics * 0.3;
  ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${mainAlpha})`;
  drawTaperedShape(ctx, waveformPath, normals, halfWidths, 1.0);

  // === LAYER 3: Bright core ===
  const coreR = Math.min(255, color.r + 60);
  const coreG = Math.min(255, color.g + 60);
  const coreB = Math.min(255, color.b + 60);
  ctx.strokeStyle = `rgba(${coreR}, ${coreG}, ${coreB}, ${0.4 + physics * 0.5})`;
  ctx.lineWidth = Math.max(1, 1 + physics * 2);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  drawSmoothPath(ctx, waveformPath);

  // === LAYER 4: Pulses ===
  for (const pulse of pulses) {
    drawPulse(ctx, waveformPath, pulse, color);
  }
}

function drawTaperedShape(
  ctx: CanvasRenderingContext2D,
  path: Position[],
  normals: { nx: number; ny: number }[],
  halfWidths: number[],
  widthScale: number
): void {
  if (path.length < 2) return;
  ctx.beginPath();

  // Left edge (forward)
  const first = path[0];
  const hw0 = halfWidths[0] * widthScale;
  ctx.moveTo(first.x + normals[0].nx * hw0, first.y + normals[0].ny * hw0);

  for (let i = 1; i < path.length; i++) {
    const hw = halfWidths[i] * widthScale;
    const lx = path[i].x + normals[i].nx * hw;
    const ly = path[i].y + normals[i].ny * hw;
    if (i < path.length - 1) {
      const nextHw = halfWidths[i + 1] * widthScale;
      const nlx = path[i + 1].x + normals[i + 1].nx * nextHw;
      const nly = path[i + 1].y + normals[i + 1].ny * nextHw;
      ctx.quadraticCurveTo(lx, ly, (lx + nlx) / 2, (ly + nly) / 2);
    } else {
      ctx.lineTo(lx, ly);
    }
  }

  // Right edge (backward)
  for (let i = path.length - 1; i >= 0; i--) {
    const hw = halfWidths[i] * widthScale;
    const rx = path[i].x - normals[i].nx * hw;
    const ry = path[i].y - normals[i].ny * hw;
    if (i > 0) {
      const prevHw = halfWidths[i - 1] * widthScale;
      const prx = path[i - 1].x - normals[i - 1].nx * prevHw;
      const pry = path[i - 1].y - normals[i - 1].ny * prevHw;
      ctx.quadraticCurveTo(rx, ry, (rx + prx) / 2, (ry + pry) / 2);
    } else {
      ctx.lineTo(rx, ry);
    }
  }

  ctx.closePath();
  ctx.fill();
}

function drawDestinationStem(
  ctx: CanvasRenderingContext2D,
  flower: Position,
  heart: Position,
  color: { r: number; g: number; b: number },
  blendPosition: number,
  time: number
): void {
  const basePath = interpolateStemPath(flower, heart, SEGMENTS);
  const normals = calculateNormals(basePath);

  // Destination stems animate based on blend position
  const waveformPath: Position[] = basePath.map((point, i) => {
    const t = i / SEGMENTS;
    const rootFade = Math.min(1, Math.max(0, (t - 0.10) / 0.2));
    const positionFade = rootFade * (1 - t * 0.5);

    // Blend position creates traveling wave
    const blendWave = Math.sin(t * 4 * Math.PI - blendPosition * 6.28 + time * 2) * 10;

    // Organic flow
    const flow = Math.sin(t * 5 * Math.PI + time * 2) * 6 * positionFade;

    const totalDisplacement = (blendWave + flow) * positionFade;

    return {
      x: point.x + normals[i].nx * totalDisplacement,
      y: point.y + normals[i].ny * totalDisplacement,
    };
  });
  waveformPath[0] = { ...flower };
  waveformPath[waveformPath.length - 1] = { ...heart };

  // Softer rendering for destination stems
  ctx.save();
  ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.25)`;
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.filter = 'blur(4px)';
  drawSmoothPath(ctx, waveformPath);
  ctx.restore();

  ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  drawSmoothPath(ctx, waveformPath);
}

function drawPulse(
  ctx: CanvasRenderingContext2D,
  path: Position[],
  pulse: Pulse,
  color: { r: number; g: number; b: number }
): void {
  const index = Math.floor(pulse.progress * (path.length - 1));
  if (index < 0 || index >= path.length) return;

  const point = path[index];
  const radius = 6 + pulse.intensity * 10;

  const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
  const alpha = pulse.intensity * 0.6;
  gradient.addColorStop(0, `rgba(${color.r + 60}, ${color.g + 60}, ${color.b + 60}, ${alpha})`);
  gradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 0.5})`);
  gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
}
