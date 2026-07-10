/**
 * OrbSystem - Slot-centric orb management (like CrystalHeart).
 *
 * One orb per manifest steering slot; orbs float freely with physics —
 * same behavior as CrystalHeart. Orb i renders useOrbRenderData()[i] and
 * reads stemBodies[i]: the slot store's `order` is the only index↔slot
 * mapping (see lib/orbRenderData.ts).
 *
 * Design: Neural nodes peering into the model's consciousness.
 * - Glassy, reflective orbs (control nodes)
 * - Win95 tactile config panels (drag from title bar)
 * - Free physics movement, playable bounce
 */

import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { SlotConfigPanel } from "./steering/SlotConfigPanel";
import { springs } from "../hooks/animations";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import { useSlotStore } from "../stores/useSlotStore";
import { useOrbRenderData, physicalStemOf, type OrbRenderData } from "../lib/orbRenderData";
import type {
  Destination,
  DestinationMode,
  DestinationSpace,
} from "../types/destinations";
import { STEM_COLORS } from "../data/features";
import type Matter from "matter-js";

// =============================================================================
// GRID CONFIGURATION
// =============================================================================

const GRID_COLS = 6;
const GRID_ROWS = 6;
const GRID_PADDING = 80; // Padding from viewport edges

// =============================================================================
// TYPES
// =============================================================================

interface OrbSystemProps {
  /** Matter.js bodies for physics simulation */
  stemBodies: Matter.Body[];
  destinationBodies: Matter.Body[];
  /** Check if a body is currently being dragged */
  isDragging: (body: Matter.Body) => boolean;
  /** Container dimensions for grid calculation */
  containerSize?: { width: number; height: number };
  // === Destination Orbs ===
  destinationStates: {
    latent: {
      mode: DestinationMode;
      destinationA: Destination | null;
      destinationB: Destination | null;
    };
    prompt: {
      mode: DestinationMode;
      destinationA: Destination | null;
      destinationB: Destination | null;
    };
  };
  onDestinationClick: (space: DestinationSpace) => void;
}

const ORB_SIZE = 100;
const DESTINATION_COLORS = {
  latent: "#8a6aaa",
  prompt: "#aa8a6a",
} as const;

// =============================================================================
// GRID UTILITIES
// =============================================================================

function calculateGridPoints(
  width: number,
  height: number,
  cols: number = GRID_COLS,
  rows: number = GRID_ROWS
): { x: number; y: number }[][] {
  const grid: { x: number; y: number }[][] = [];
  const cellWidth = (width - GRID_PADDING * 2) / (cols - 1);
  const cellHeight = (height - GRID_PADDING * 2) / (rows - 1);

  for (let row = 0; row < rows; row++) {
    grid[row] = [];
    for (let col = 0; col < cols; col++) {
      grid[row][col] = {
        x: GRID_PADDING + col * cellWidth,
        y: GRID_PADDING + row * cellHeight,
      };
    }
  }
  return grid;
}

// NOTE: Snap-to-grid logic removed - orbs now behave exactly like CrystalHeart
// (free physics movement). Grid overlay kept for visual interest only.

// =============================================================================
// ORB SYSTEM COMPONENT
// =============================================================================

export function OrbSystem({
  stemBodies,
  destinationBodies,
  isDragging,
  containerSize,
  destinationStates,
  onDestinationClick,
}: OrbSystemProps) {
  // Slot identity + display metadata, index-aligned with stemBodies.
  const orbs = useOrbRenderData();
  const slots = useSlotStore((s) => s.slots);
  // Narrow audio subscriptions: prominence drives the per-orb glow and is a
  // fresh object per telemetry message (~10Hz during playback), so THIS
  // component re-renders with the music — the tree above it does not.
  const stemProminence = useAudioActivityStore((s) => s.prominence);
  const destinationActivity = useAudioActivityStore((s) => s.overallActivity);
  // Imperative position updates — no React re-renders for physics movement.
  // Refs keyed by body index (slot orbs) or destination space — index keys
  // stay stable across slot-store changes, so the rAF loop never re-binds.
  const orbDivRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const rafRef = useRef(0);

  const setOrbRef = useCallback((key: string) => (el: HTMLDivElement | null) => {
    if (el) orbDivRefs.current.set(key, el);
    else orbDivRefs.current.delete(key);
  }, []);

  // rAF loop: write left/top directly to DOM (like useArmAnimation pattern)
  useEffect(() => {
    const tick = () => {
      // Slot orbs
      for (let i = 0; i < stemBodies.length; i++) {
        const el = orbDivRefs.current.get(`slot-${i}`);
        if (!el) continue;
        el.style.left = `${stemBodies[i].position.x - ORB_SIZE / 2}px`;
        el.style.top = `${stemBodies[i].position.y - ORB_SIZE / 2}px`;
      }
      // Destination orbs
      for (let i = 0; i < destinationBodies.length; i++) {
        const key = i === 0 ? "dest-latent" : "dest-prompt";
        const el = orbDivRefs.current.get(key);
        if (!el) continue;
        el.style.left = `${destinationBodies[i].position.x - ORB_SIZE / 2}px`;
        el.style.top = `${destinationBodies[i].position.y - ORB_SIZE / 2}px`;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [stemBodies, destinationBodies]);

  // Selected orb for config panel
  const [selectedSlot, setSelectedSlot] = useState<string | null>(null);

  // Click vs drag detection - track mouse down position (like CrystalHeart)
  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);

  // Calculate grid based on container/viewport size
  const viewportSize = containerSize || { width: 0, height: 0 };

  const grid = useMemo(
    () => calculateGridPoints(viewportSize.width, viewportSize.height),
    [viewportSize.width, viewportSize.height]
  );

  // Handle orb click
  const handleOrbClick = useCallback((slot: string) => {
    setSelectedSlot((prev) => (prev === slot ? null : slot));
  }, []);

  // Close config panel
  const handleClosePanel = useCallback(() => {
    setSelectedSlot(null);
  }, []);

  // Get prominence for an orb (from backend telemetry or derive from rank)
  const getOrbProminence = useCallback(
    (orb: OrbRenderData): { prominence: number; surprise: boolean } => {
      // Try to get from backend telemetry
      const baseStem = physicalStemOf(orb.linkTarget);
      if (stemProminence && stemProminence[baseStem]) {
        return {
          prominence: stemProminence[baseStem].prominence,
          surprise: stemProminence[baseStem].surprise_active,
        };
      }

      // Fall back to rank-based prominence (compressed 4x range)
      const rankToProminence: Record<number | 'null', number> = {
        1: 1.0,
        2: 0.65,
        3: 0.40,
        4: 0.25,
        null: 0.05,
      };
      const rankKey = orb.saeRank === null ? 'null' : orb.saeRank;
      return {
        prominence: rankToProminence[rankKey] ?? 0.5,
        surprise: false,
      };
    },
    [stemProminence]
  );

  // Read selected orb's body ref directly — position is live (mutated by Matter.js).
  // No memo needed: SlotConfigPanel reads this once on mount to compute initial placement.
  const selectedIndex = selectedSlot !== null
    ? orbs.findIndex((orb) => orb.slot === selectedSlot)
    : -1;
  const selectedOrb = selectedIndex >= 0 ? orbs[selectedIndex] : undefined;
  const selectedOrbBody = selectedIndex >= 0 ? stemBodies[selectedIndex] : undefined;
  const selectedOrbPosition = selectedOrbBody
    ? { x: selectedOrbBody.position.x, y: selectedOrbBody.position.y }
    : undefined;

  // Empty set for grid overlay (snap-to-grid disabled)
  const emptySet = useMemo(() => new Set<number>(), []);

  return (
    <>
      {/* Subtle grid overlay - hidden since snap-to-grid is disabled */}
      <GridOverlay grid={grid} activeOrbs={emptySet} />

      {/* Orbs */}
      {stemBodies.map((body, index) => {
        const orb = orbs[index];
        if (!orb) return null;

        const isBeingDragged = isDragging(body);
        const isSelected = selectedSlot === orb.slot;
        const stemColor = STEM_COLORS[physicalStemOf(orb.linkTarget)] || "#888";

        // Get prominence for glow effect
        const { prominence, surprise } = getOrbProminence(orb);

        return (
          <div
            key={orb.slot}
            ref={setOrbRef(`slot-${index}`)}
            className="block-orb-wrapper"
            style={{
              position: "absolute",
              // left/top set imperatively by rAF loop
              width: ORB_SIZE,
              height: ORB_SIZE,

              zIndex: isBeingDragged || isSelected ? 60 : 50,
              cursor: isBeingDragged ? "grabbing" : "grab",
              transform: `scale(${isBeingDragged ? 1.1 : 1 + prominence * 0.05})`,
              // Prominence-based glow: more prominent = stronger glow
              filter: isBeingDragged
                ? "drop-shadow(0 8px 24px rgba(0,0,0,0.4))"
                : orb.enabled
                  ? `drop-shadow(0 0 ${8 + prominence * 20}px ${orb.color}${Math.round(prominence * 80 + 20).toString(16).padStart(2, '0')}) drop-shadow(0 4px 12px rgba(0,0,0,0.2))`
                  : "drop-shadow(0 4px 12px rgba(0,0,0,0.2))",
              transition: "transform 0.15s ease, filter 0.3s ease",
            }}
            data-slot={orb.slot}
            data-prominence={prominence.toFixed(2)}
            data-surprise={surprise}
            // Click detection like CrystalHeart - simple mouseDown/mouseUp
            onMouseDown={(e) => {
              mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
            }}
            onMouseUp={(e) => {
              if (!mouseDownPosRef.current) return;

              // Check if this was a click (minimal movement) vs a drag
              const dx = e.clientX - mouseDownPosRef.current.x;
              const dy = e.clientY - mouseDownPosRef.current.y;
              const distance = Math.sqrt(dx * dx + dy * dy);

              // If moved less than 5 pixels, treat as click
              if (distance < 5) {
                handleOrbClick(orb.slot);
              }

              mouseDownPosRef.current = null;
            }}
          >
            {/* 3D mesh now rendered in shared BellyScene Canvas */}

            {/* Slot Label (below orb) */}
            <div
              className="orb-label"
              style={{
                position: "absolute",
                bottom: -28,
                left: "50%",
                transform: "translateX(-50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 1,
                pointerEvents: "none",
              }}
            >
              {/* Semantic role label */}
              <span
                style={{
                  fontSize: "9px",
                  fontWeight: 700,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: orb.color,
                  textShadow: `0 0 10px ${orb.color}50, 0 1px 2px rgba(0,0,0,0.9)`,
                }}
              >
                {orb.displayName}
              </span>
              {/* Slot name (subtle) */}
              <span
                style={{
                  fontSize: "7px",
                  fontWeight: 500,
                  fontFamily: "monospace",
                  letterSpacing: "0.05em",
                  color: orb.color,
                  opacity: 0.5,
                  textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                }}
              >
                {orb.slot}
              </span>
              {/* Link target indicator */}
              <span
                style={{
                  fontSize: "7px",
                  fontWeight: 500,
                  letterSpacing: "0.08em",
                  color: stemColor,
                  opacity: orb.enabled ? 0.8 : 0.4,
                  textTransform: "uppercase",
                  textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                }}
              >
                {orb.linkTarget.replace("_", " ")}
              </span>
            </div>

          </div>
        );
      })}

      {/* Destination Orbs */}
      {destinationBodies.map((body, index) => {
        const space: DestinationSpace = index === 0 ? "latent" : "prompt";
        const state = destinationStates[space];
        const isBeingDragged = isDragging(body);
        const accentColor = DESTINATION_COLORS[space];
        const isConfigured = state.destinationA !== null && state.destinationB !== null;

        return (
          <div
            key={`destination-${space}`}
            ref={setOrbRef(`dest-${space}`)}
            className="block-orb-wrapper"
            style={{
              position: "absolute",
              // left/top set imperatively by rAF loop
              width: ORB_SIZE,
              height: ORB_SIZE,

              zIndex: isBeingDragged ? 60 : 50,
              cursor: isBeingDragged ? "grabbing" : "grab",
              transform: `scale(${isBeingDragged ? 1.08 : 1 + destinationActivity * 0.04})`,
              filter: isBeingDragged
                ? "drop-shadow(0 8px 24px rgba(0,0,0,0.4))"
                : `drop-shadow(0 0 ${8 + destinationActivity * 20}px ${accentColor}66) drop-shadow(0 4px 12px rgba(0,0,0,0.2))`,
              transition: "transform 0.15s ease, filter 0.3s ease",
            }}
            data-space={space}
            onMouseDown={(e) => {
              mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
            }}
            onMouseUp={(e) => {
              if (!mouseDownPosRef.current) return;

              const dx = e.clientX - mouseDownPosRef.current.x;
              const dy = e.clientY - mouseDownPosRef.current.y;
              const distance = Math.sqrt(dx * dx + dy * dy);

              if (distance < 5) {
                onDestinationClick(space);
              }

              mouseDownPosRef.current = null;
            }}
          >
            {/* 3D mesh now rendered in shared BellyScene Canvas */}

            <div
              className="orb-label"
              style={{
                position: "absolute",
                bottom: -24,
                left: "50%",
                transform: "translateX(-50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
                pointerEvents: "none",
              }}
            >
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 600,
                  fontFamily: "monospace",
                  letterSpacing: "0.05em",
                  color: accentColor,
                  textShadow: `0 0 8px ${accentColor}40, 0 1px 2px rgba(0,0,0,0.8)`,
                  textTransform: "uppercase",
                }}
              >
                {space}
              </span>
              <span
                style={{
                  fontSize: "8px",
                  fontWeight: 500,
                  letterSpacing: "0.08em",
                  color: accentColor,
                  opacity: isConfigured ? 1 : 0.5,
                  textTransform: "uppercase",
                  textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                }}
              >
                {state.mode}
              </span>
            </div>
          </div>
        );
      })}

      {/* Slot Config Panel */}
      <AnimatePresence>
        {selectedSlot && selectedOrb && slots[selectedSlot] && (
          <SlotConfigPanel
            slot={selectedSlot}
            mapping={slots[selectedSlot]}
            displayName={selectedOrb.displayName}
            description={selectedOrb.description}
            accentColor={selectedOrb.color}
            isOpen={true}
            onClose={handleClosePanel}
            orbPosition={selectedOrbPosition}
            containerSize={containerSize}
          />
        )}
      </AnimatePresence>
    </>
  );
}

// =============================================================================
// GRID OVERLAY COMPONENT
// =============================================================================

interface GridOverlayProps {
  grid: { x: number; y: number }[][];
  activeOrbs: Set<number>;
}

function GridOverlay({ grid, activeOrbs }: GridOverlayProps) {
  const isActive = activeOrbs.size > 0;

  return (
    <motion.div
      className="grid-overlay"
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        zIndex: 30,
      }}
      animate={{ opacity: isActive ? 1 : 0 }}
      transition={{ duration: 0.2 }}
    >
      <svg width="100%" height="100%" style={{ position: "absolute" }}>
        <defs>
          {/* Subtle glow filter for grid points */}
          <filter id="gridGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Grid intersection points */}
        {grid.map((row, rowIdx) =>
          row.map((point, colIdx) => (
            <motion.circle
              key={`${rowIdx}-${colIdx}`}
              cx={point.x}
              cy={point.y}
              r={3}
              fill="var(--color-text-dim)"
              opacity={0.3}
              filter="url(#gridGlow)"
              initial={{ scale: 0 }}
              animate={{ scale: isActive ? 1 : 0 }}
              transition={{
                delay: (rowIdx + colIdx) * 0.01,
                ...springs.bouncy,
              }}
            />
          ))
        )}

        {/* Faint grid lines */}
        {isActive && (
          <g opacity={0.1} stroke="var(--color-text-dim)" strokeWidth={1}>
            {/* Horizontal lines */}
            {grid.map((row, rowIdx) => (
              <motion.line
                key={`h-${rowIdx}`}
                x1={row[0].x}
                y1={row[0].y}
                x2={row[row.length - 1].x}
                y2={row[row.length - 1].y}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.3, delay: rowIdx * 0.02 }}
              />
            ))}
            {/* Vertical lines */}
            {grid[0].map((_, colIdx) => (
              <motion.line
                key={`v-${colIdx}`}
                x1={grid[0][colIdx].x}
                y1={grid[0][colIdx].y}
                x2={grid[grid.length - 1][colIdx].x}
                y2={grid[grid.length - 1][colIdx].y}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.3, delay: colIdx * 0.02 }}
              />
            ))}
          </g>
        )}
      </svg>
    </motion.div>
  );
}
