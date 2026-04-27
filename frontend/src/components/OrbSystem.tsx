/**
 * OrbSystem - Block-centric orb management (like CrystalHeart).
 *
 * Phase 1-2: Each orb represents a UNet block (not a stem).
 * Orbs float freely with physics - same behavior as CrystalHeart.
 *
 * Design: Neural nodes peering into the model's consciousness.
 * - Glassy, reflective orbs (control nodes)
 * - Win95 tactile config panels (drag from title bar)
 * - Free physics movement, playable bounce
 */

import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { FlowerStemType } from "./FlowerOrb";
import { BlockConfigPanel } from "./steering/BlockConfigPanel";
import { springs } from "../hooks/animations";
import type {
  BlockCode,
  BlockMapping,
  LinkTarget,
  Rank,
  SpatialMode,
  StemActivity,
  StrengthRange,
  IntensitySource,
  IntensityCurve,
} from "../types/sae";
import type {
  Destination,
  DestinationMode,
  DestinationSpace,
} from "../types/destinations";
import { BLOCKS, BLOCK_COLORS, STEM_COLORS } from "../data/features";
import type Matter from "matter-js";

/** Map link targets to physical stems for flower type */
function linkTargetToFlowerType(linkTarget: LinkTarget | undefined): FlowerStemType {
  if (!linkTarget) return 'other';
  if (linkTarget.startsWith('drums')) return 'drums';
  if (linkTarget.startsWith('other')) return 'other';
  if (linkTarget.startsWith('bass')) return 'bass';
  if (linkTarget.startsWith('vocals')) return 'vocals';
  return 'other';
}

// =============================================================================
// GRID CONFIGURATION
// =============================================================================

const GRID_COLS = 6;
const GRID_ROWS = 6;
const GRID_PADDING = 80; // Padding from viewport edges

/** Block codes in orb order */
const BLOCK_ORDER: BlockCode[] = ["down.2.1", "mid.0", "up.0.0", "up.0.1"];

/** Default grid positions for each block (col, row) - corners, crystal heart in center */
const DEFAULT_GRID_POSITIONS: Record<BlockCode, { col: number; row: number }> = {
  "down.2.1": { col: 0, row: 0 }, // Composition - top left
  "mid.0": { col: 5, row: 0 },    // Abstract - top right
  "up.0.0": { col: 0, row: 5 },   // Details - bottom left
  "up.0.1": { col: 5, row: 5 },   // Style - bottom right
};

// =============================================================================
// TYPES
// =============================================================================

interface OrbSystemProps {
  /** Matter.js bodies for physics simulation */
  stemBodies: Matter.Body[];
  destinationBodies: Matter.Body[];
  /** Check if a body is currently being dragged */
  isDragging: (body: Matter.Body) => boolean;
  /** Real-time stem activity levels */
  stemActivity: StemActivity;
  /** Block-centric mappings */
  blockMappings: Record<BlockCode, BlockMapping>;
  /** Container dimensions for grid calculation */
  containerSize?: { width: number; height: number };
  /** Computed prominence per stem (from backend telemetry) */
  stemProminence?: Record<string, { prominence: number; surprise_active: boolean }>;
  // === Block Config Callbacks ===
  onLinkTargetChange: (block: BlockCode, linkTarget: LinkTarget) => void;
  onFeatureChange: (block: BlockCode, featureId: number, featureLabel: string) => void;
  onStrengthRangeChange: (block: BlockCode, range: StrengthRange) => void;
  onAutoConfigChange: (block: BlockCode, autoConfig: boolean) => void;
  onSpatialModeChange: (block: BlockCode, spatialMode: SpatialMode) => void;
  onSpatialMaskChange: (block: BlockCode, mask: number[]) => void;
  onIntensitySourceChange: (block: BlockCode, source: IntensitySource) => void;
  onIntensityCurveChange: (block: BlockCode, curve: IntensityCurve) => void;
  onIntensityGammaChange: (block: BlockCode, gamma: number) => void;
  onSaeRankChange: (block: BlockCode, rank: Rank) => void;
  onToggleBlock: (block: BlockCode) => void;
  // === Destination Orbs ===
  destinationActivity: number;
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
  stemActivity: _stemActivity,
  blockMappings,
  containerSize,
  stemProminence,
  onLinkTargetChange,
  onFeatureChange,
  onStrengthRangeChange,
  onAutoConfigChange,
  onSpatialModeChange,
  onSpatialMaskChange,
  onIntensitySourceChange,
  onIntensityCurveChange,
  onIntensityGammaChange,
  onSaeRankChange,
  onToggleBlock,
  destinationActivity,
  destinationStates,
  onDestinationClick,
}: OrbSystemProps) {
  // Imperative position updates — no React re-renders for physics movement.
  // Refs for each orb div, keyed by block code or destination space.
  const orbDivRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const rafRef = useRef(0);

  const setOrbRef = useCallback((key: string) => (el: HTMLDivElement | null) => {
    if (el) orbDivRefs.current.set(key, el);
    else orbDivRefs.current.delete(key);
  }, []);

  // rAF loop: write left/top directly to DOM (like useArmAnimation pattern)
  useEffect(() => {
    const tick = () => {
      // Stem orbs
      for (let i = 0; i < stemBodies.length; i++) {
        const block = BLOCK_ORDER[i];
        if (!block) continue;
        const el = orbDivRefs.current.get(block);
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

  // Orb state - phase offsets for animation variety
  const orbStates = useMemo(
    () =>
      BLOCK_ORDER.map((block, i) => ({
        block,
        gridPos: DEFAULT_GRID_POSITIONS[block],
        phase: i * 0.25 + Math.random() * 0.1,
      })),
    []
  );

  // Selected orb for config panel
  const [selectedBlock, setSelectedBlock] = useState<BlockCode | null>(null);

  // Click vs drag detection - track mouse down position (like CrystalHeart)
  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);

  // Calculate grid based on container/viewport size
  const viewportSize = containerSize || { width: 0, height: 0 };

  const grid = useMemo(
    () => calculateGridPoints(viewportSize.width, viewportSize.height),
    [viewportSize.width, viewportSize.height]
  );

  // Handle orb click
  const handleOrbClick = useCallback((block: BlockCode) => {
    setSelectedBlock((prev) => (prev === block ? null : block));
  }, []);

  // Close config panel
  const handleClosePanel = useCallback(() => {
    setSelectedBlock(null);
  }, []);

  // Get prominence for a block (from backend telemetry or derive from rank)
  const getBlockProminence = useCallback(
    (block: BlockCode): { prominence: number; surprise: boolean } => {
      const mapping = blockMappings[block];
      if (!mapping) return { prominence: 0, surprise: false };

      // Try to get from backend telemetry
      const baseStem = linkTargetToFlowerType(mapping.linkTarget);
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
      const rankKey = mapping.saeRank === null ? 'null' : mapping.saeRank;
      return {
        prominence: rankToProminence[rankKey] ?? 0.5,
        surprise: false,
      };
    },
    [blockMappings, stemProminence]
  );

  // Read selected orb's body ref directly — position is live (mutated by Matter.js).
  // No memo needed: BlockConfigPanel reads this once on mount to compute initial placement.
  const selectedOrbBody = selectedBlock !== null
    ? stemBodies[BLOCK_ORDER.indexOf(selectedBlock)]
    : undefined;
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
        const orbState = orbStates[index];
        if (!orbState) return null;

        const block = orbState.block;
        const mapping = blockMappings[block];
        const isBeingDragged = isDragging(body);
        const isSelected = selectedBlock === block;

        const blockColor = BLOCK_COLORS[block];
        const baseStem = mapping ? linkTargetToFlowerType(mapping.linkTarget) : 'other';
        const stemColor = STEM_COLORS[baseStem] || "#888";

        // Get prominence for glow effect
        const { prominence, surprise } = getBlockProminence(block);

        return (
          <div
            key={block}
            ref={setOrbRef(block)}
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
                : mapping?.enabled
                  ? `drop-shadow(0 0 ${8 + prominence * 20}px ${blockColor}${Math.round(prominence * 80 + 20).toString(16).padStart(2, '0')}) drop-shadow(0 4px 12px rgba(0,0,0,0.2))`
                  : "drop-shadow(0 4px 12px rgba(0,0,0,0.2))",
              transition: "transform 0.15s ease, filter 0.3s ease",
            }}
            data-block={block}
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
                handleOrbClick(block);
              }

              mouseDownPosRef.current = null;
            }}
          >
            {/* 3D mesh now rendered in shared BellyScene Canvas */}

            {/* Block Label (below orb) */}
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
                  color: blockColor,
                  textShadow: `0 0 10px ${blockColor}50, 0 1px 2px rgba(0,0,0,0.9)`,
                }}
              >
                {BLOCKS[block].name}
              </span>
              {/* Block code (subtle) */}
              <span
                style={{
                  fontSize: "7px",
                  fontWeight: 500,
                  fontFamily: "monospace",
                  letterSpacing: "0.05em",
                  color: blockColor,
                  opacity: 0.5,
                  textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                }}
              >
                {block}
              </span>
              {/* Link target indicator */}
              {mapping && (
                <span
                  style={{
                    fontSize: "7px",
                    fontWeight: 500,
                    letterSpacing: "0.08em",
                    color: stemColor,
                    opacity: mapping.enabled ? 0.8 : 0.4,
                    textTransform: "uppercase",
                    textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                  }}
                >
                  {mapping.linkTarget.replace("_", " ")}
                </span>
              )}
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

      {/* Block Config Panel */}
      <AnimatePresence>
        {selectedBlock && blockMappings[selectedBlock] && (
          <BlockConfigPanel
            block={selectedBlock}
            mapping={blockMappings[selectedBlock]}
            isOpen={true}
            onClose={handleClosePanel}
            onLinkTargetChange={onLinkTargetChange}
            onFeatureChange={onFeatureChange}
            onStrengthRangeChange={onStrengthRangeChange}
            onAutoConfigChange={onAutoConfigChange}
            onSpatialModeChange={onSpatialModeChange}
            onSpatialMaskChange={onSpatialMaskChange}
            onIntensitySourceChange={onIntensitySourceChange}
            onIntensityCurveChange={onIntensityCurveChange}
            onIntensityGammaChange={onIntensityGammaChange}
            onSaeRankChange={onSaeRankChange}
            onToggle={onToggleBlock}
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

// =============================================================================
// CSS STYLES (add to global CSS)
// =============================================================================

/*
Add these styles to your global CSS:

.block-orb-wrapper {
  transition: filter 0.2s ease;
}

.block-orb-wrapper:hover {
  filter: drop-shadow(0 6px 20px rgba(0,0,0,0.3)) !important;
}

.orb-label {
  user-select: none;
  text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}
*/
