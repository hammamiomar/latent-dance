/**
 * useMatterPhysics - Matter.js physics world with dynamic constraints.
 *
 * Supports dormant → active orb flow:
 * - Dormant orbs cluster in bottom-right
 * - Active orbs move to corners with tethers to CrystalHeart
 * - Progressive tethering: more connections = more anchored
 */

import { useEffect, useRef, useCallback, useState, useSyncExternalStore } from "react";
import Matter from "matter-js";

// ============================================================================
// Physics Tick External Store
//
// Module-level store so only components that explicitly subscribe via
// usePhysicsTick() re-render at 60fps — prevents cascading re-renders
// through App.tsx when physics positions update.
// ============================================================================

let _physicsTick = 0;
const _physicsListeners = new Set<() => void>();

function _advancePhysicsTick() {
  _physicsTick++;
  for (const l of _physicsListeners) l();
}

function _subscribePhysicsTick(listener: () => void) {
  _physicsListeners.add(listener);
  return () => { _physicsListeners.delete(listener); };
}

function _getPhysicsTick() {
  return _physicsTick;
}

/**
 * Subscribe to physics frame updates. Call in components that need
 * to re-render when body positions change (OrbSystem, CrystalHeart).
 */
export function usePhysicsTick(): number {
  return useSyncExternalStore(_subscribePhysicsTick, _getPhysicsTick);
}

// ============================================================================
// Types
// ============================================================================

export interface PhysicsConfig {
  width: number;
  height: number;
  containerRef?: React.RefObject<HTMLElement | null>;
}

export interface PhysicsBodies {
  heart: Matter.Body;
  stemOrbs: Matter.Body[];
  destinationOrbs: Matter.Body[];
}

export interface PhysicsSettings {
  frictionAir: number;      // 0-0.05
  stiffness: number;        // 0-0.01
  damping: number;          // 0-0.1
  gravityY: number;         // -0.001 to 0.001
  pinToCorners: boolean;    // Lock orbs to corners
}

export const DEFAULT_PHYSICS_SETTINGS: PhysicsSettings = {
  frictionAir: 0.004,
  stiffness: 0.0008,
  damping: 0.005,
  gravityY: 0,
  pinToCorners: false,
};

export interface PhysicsWorld {
  engine: Matter.Engine;
  bodies: PhysicsBodies;
  isDragging: (body: Matter.Body) => boolean;
  draggedBody: Matter.Body | null;
  activateOrb: (orbIndex: number) => void;
  deactivateOrb: (orbIndex: number) => void;
  /** Move an orb smoothly to target position (for snap-to-grid) */
  moveOrbTo: (orbIndex: number, x: number, y: number) => void;
  activeOrbs: boolean[];
  activeCount: number;
  settings: PhysicsSettings;
  updateSettings: (settings: Partial<PhysicsSettings>) => void;
}

// ============================================================================
// Physics Constants
// ============================================================================

/** Exported for use in PhysicsCanvas tendril rendering */
export const TENDRIL_REST_LENGTH = 220;

const PHYSICS = {
  gravity: { x: 0, y: 0, scale: 0 },

  heart: {
    radius: 60,
    friction: 0.001,
    frictionAir: 0.004,    // Very low - floats in liquid
    restitution: 0.5,
    density: 0.0008,
  },

  stemOrb: {
    radius: 40,
    friction: 0.001,       // Same as heart
    frictionAir: 0.004,    // Same as heart
    restitution: 0.5,      // Same as heart
    density: 0.0008,       // Same as heart
  },

  // Very loose tendril - like a thread in water
  heartToOrb: {
    stiffness: 0.0008,     // Extremely soft
    damping: 0.005,        // Minimal damping = more wobbly
    length: TENDRIL_REST_LENGTH,
  },

  // Gentle home pull
  orbToCorner: {
    stiffness: 0.004,      // Very gentle
    damping: 0.01,
    length: 0,
  },

  // Progressive center anchor
  heartAnchor: {
    baseStiffness: 0,
    perConnection: 0.001,
    maxStiffness: 0.005,
    damping: 0.02,
  },

  mouse: {
    stiffness: 0.03,       // Softer drag feel
    damping: 0.08,
  },
};

// ============================================================================
// Corner positions - Order matches OrbSystem: down.2.1 (TL), mid.0 (TR), up.0.0 (BL), up.0.1 (BR)
// ============================================================================

function getCornerPositions(width: number, height: number) {
  const padding = 120;
  return [
    { x: padding, y: padding },                           // 0: Top-left (down.2.1)
    { x: width - padding, y: padding },                   // 1: Top-right (mid.0)
    { x: padding, y: height - padding - 60 },             // 2: Bottom-left (up.0.0)
    { x: width - padding, y: height - padding - 60 },     // 3: Bottom-right (up.0.1)
  ];
}

// Initial positions for block-centric orbs (start at corners)
function getInitialOrbPositions(width: number, height: number) {
  // Use corner positions directly - orbs start at their corner positions
  return getCornerPositions(width, height);
}

function getDestinationPositions(width: number, height: number) {
  const padding = 110;
  return [
    { x: padding, y: height / 2 },
    { x: width - padding, y: height / 2 },
  ];
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useMatterPhysics(config: PhysicsConfig): PhysicsWorld | null {
  const engineRef = useRef<Matter.Engine | null>(null);
  const bodiesRef = useRef<PhysicsBodies | null>(null);
  const constraintsRef = useRef<{
    heartToOrb: (Matter.Constraint | null)[];
    orbToCorner: (Matter.Constraint | null)[];
    destToHeart: (Matter.Constraint | null)[];
    destToAnchor: (Matter.Constraint | null)[];
    heartAnchor: Matter.Constraint | null;
  }>({ heartToOrb: [], orbToCorner: [], destToHeart: [], destToAnchor: [], heartAnchor: null });

  const runnerRef = useRef<number | null>(null);
  const wallsRef = useRef<Matter.Body[]>([]);
  const lastDimsRef = useRef<{ width: number; height: number } | null>(null);
  const [draggedBody, setDraggedBody] = useState<Matter.Body | null>(null);
  const [activeOrbs, setActiveOrbs] = useState<boolean[]>([false, false, false, false]);
  const [settings, setSettings] = useState<PhysicsSettings>(DEFAULT_PHYSICS_SETTINGS);

  // Ref to access current settings in callbacks (avoids stale closure)
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [ready, setReady] = useState(false);
  const initializedRef = useRef(false);
  const configRef = useRef(config);
  configRef.current = config;
  const lastDestinationAnchorRef = useRef<{ width: number; height: number } | null>(null);

  const activeCount = activeOrbs.filter(Boolean).length;

  // Update settings and apply to physics
  const updateSettings = useCallback((newSettings: Partial<PhysicsSettings>) => {
    setSettings(prev => {
      const updated = { ...prev, ...newSettings };

      // Apply to engine
      if (engineRef.current && bodiesRef.current) {
        const engine = engineRef.current;
        const { heart, stemOrbs, destinationOrbs } = bodiesRef.current;

        // Update gravity
        engine.gravity.y = updated.gravityY;

        // Update friction
        heart.frictionAir = updated.frictionAir;
        stemOrbs.forEach(orb => {
          orb.frictionAir = updated.frictionAir * 1.5;
        });
        destinationOrbs.forEach(orb => {
          orb.frictionAir = updated.frictionAir * 1.5;
        });

        // Update constraint stiffness/damping
        constraintsRef.current.heartToOrb.forEach(c => {
          if (c) {
            c.stiffness = updated.stiffness;
            c.damping = updated.damping;
          }
        });
        constraintsRef.current.destToAnchor.forEach(c => {
          if (c) {
            c.stiffness = Math.min(0.01, updated.stiffness * 2.5);
            c.damping = updated.damping;
          }
        });

        // Pin to corners mode
        constraintsRef.current.orbToCorner.forEach(c => {
          if (c) {
            c.stiffness = updated.pinToCorners ? 0.1 : 0.004;
          }
        });
      }

      return updated;
    });
  }, []);

  // Initialize physics world
  useEffect(() => {
    if (initializedRef.current) return;
    if (config.width === 0 || config.height === 0) return;

    initializedRef.current = true;

    const engine = Matter.Engine.create({ gravity: PHYSICS.gravity });
    engineRef.current = engine;

    // Heart starts in center with initial velocity for floating feel
    const heart = Matter.Bodies.circle(
      config.width / 2,
      config.height / 2,
      PHYSICS.heart.radius,
      {
        label: "heart",
        friction: PHYSICS.heart.friction,
        frictionAir: PHYSICS.heart.frictionAir,
        restitution: PHYSICS.heart.restitution,
        density: PHYSICS.heart.density,
      }
    );

    // Give heart initial velocity for floating effect
    Matter.Body.setVelocity(heart, {
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2,
    });

    // Block-centric: Orbs start at their corner positions (TL, TR, BL, BR)
    const initialPositions = getInitialOrbPositions(config.width, config.height);
    const stemOrbs = initialPositions.map((pos, i) =>
      Matter.Bodies.circle(pos.x, pos.y, PHYSICS.stemOrb.radius, {
        label: `stem-${i}`,
        friction: PHYSICS.stemOrb.friction,
        frictionAir: PHYSICS.stemOrb.frictionAir,
        restitution: PHYSICS.stemOrb.restitution,
        density: PHYSICS.stemOrb.density,
      })
    );

    const destinationPositions = getDestinationPositions(config.width, config.height);
    const destinationOrbs = destinationPositions.map((pos, i) =>
      Matter.Bodies.circle(pos.x, pos.y, PHYSICS.stemOrb.radius, {
        label: `destination-${i}`,
        friction: PHYSICS.stemOrb.friction,
        frictionAir: PHYSICS.stemOrb.frictionAir * 1.5,
        restitution: PHYSICS.stemOrb.restitution * 0.2,
        density: PHYSICS.stemOrb.density,
      })
    );
    destinationOrbs.forEach((orb, i) => {
      const pos = destinationPositions[i];
      Matter.Body.setPosition(orb, pos);
      Matter.Body.setVelocity(orb, { x: 0, y: 0 });
    });

    constraintsRef.current = {
      heartToOrb: [null, null, null, null],
      orbToCorner: [null, null, null, null],
      destToHeart: [null, null],
      destToAnchor: [null, null],
      heartAnchor: null,
    };

    // Walls with some bounce
    const wallThickness = 50;
    const walls = [
      Matter.Bodies.rectangle(config.width / 2, -wallThickness / 2, config.width + 200, wallThickness, {
        isStatic: true,
        restitution: 0.9,
      }),
      Matter.Bodies.rectangle(config.width / 2, config.height + wallThickness / 2, config.width + 200, wallThickness, {
        isStatic: true,
        restitution: 0.9,
      }),
      Matter.Bodies.rectangle(-wallThickness / 2, config.height / 2, wallThickness, config.height + 200, {
        isStatic: true,
        restitution: 0.9,
      }),
      Matter.Bodies.rectangle(config.width + wallThickness / 2, config.height / 2, wallThickness, config.height + 200, {
        isStatic: true,
        restitution: 0.9,
      }),
    ];

    Matter.Composite.add(engine.world, [heart, ...stemOrbs, ...destinationOrbs, ...walls]);
    bodiesRef.current = { heart, stemOrbs, destinationOrbs };
    wallsRef.current = walls;
    lastDimsRef.current = { width: config.width, height: config.height };

    // Destination orbs: NOT tethered to heart (independent floaters)

    // Destination orbs: anchor to left/right positions
    destinationOrbs.forEach((orb, index) => {
      const anchor = destinationPositions[index];
      const anchorConstraint = Matter.Constraint.create({
        bodyA: orb,
        pointB: anchor,
        stiffness: Math.min(0.01, settingsRef.current.stiffness * 2.5),
        damping: settingsRef.current.damping,
        length: 0,
        render: { visible: false },
      });
      Matter.Composite.add(engine.world, anchorConstraint);
      constraintsRef.current.destToAnchor[index] = anchorConstraint;
      // Snap to anchor after constraints to avoid initial bounce
      Matter.Body.setPosition(orb, anchor);
      Matter.Body.setVelocity(orb, { x: 0, y: 0 });
    });

    // Mouse constraint for dragging bodies
    const container = config.containerRef?.current || document.body;
    const mouse = Matter.Mouse.create(container);
    const mouseConstraint = Matter.MouseConstraint.create(engine, {
      mouse,
      constraint: {
        stiffness: PHYSICS.mouse.stiffness,
        damping: PHYSICS.mouse.damping,
        render: { visible: false },
      },
    });
    Matter.Composite.add(engine.world, mouseConstraint);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Matter.Events.on(mouseConstraint, "startdrag", (event: any) => {
      if (event.body) setDraggedBody(event.body as Matter.Body);
    });
    Matter.Events.on(mouseConstraint, "enddrag", () => setDraggedBody(null));

    // Idle drift counter
    let driftCounter = 0;

    // Soft bounds enforcement — only rescues bodies that escape past walls.
    // Preserves all momentum/physics feel. Just reflects velocity on escape
    // (like bouncing off an invisible outer wall) instead of killing motion.
    const rescueEscaped = () => {
      const allBodies = [heart, ...stemOrbs, ...destinationOrbs];
      const w = configRef.current.width;
      const h = configRef.current.height;

      for (const body of allBodies) {
        const { x, y } = body.position;
        const r = (body.circleRadius || 40);
        let escaped = false;
        let nx = x, ny = y;
        let vx = body.velocity.x, vy = body.velocity.y;

        if (x < -r) { nx = r + 10; vx = Math.abs(vx) * 0.5; escaped = true; }
        else if (x > w + r) { nx = w - r - 10; vx = -Math.abs(vx) * 0.5; escaped = true; }
        if (y < -r) { ny = r + 10; vy = Math.abs(vy) * 0.5; escaped = true; }
        else if (y > h + r) { ny = h - r - 10; vy = -Math.abs(vy) * 0.5; escaped = true; }

        if (escaped) {
          Matter.Body.setPosition(body, { x: nx, y: ny });
          Matter.Body.setVelocity(body, { x: vx, y: vy });
        }
      }
    };

    // Physics loop with idle drift
    const runPhysics = () => {
      Matter.Engine.update(engine, 1000 / 60);
      rescueEscaped();

      // Apply tiny random forces for idle drift (every ~60 frames = 1 sec)
      driftCounter++;
      if (driftCounter > 60) {
        driftCounter = 0;

        // Drift the heart gently
        const driftForce = 0.00002;
        Matter.Body.applyForce(heart, heart.position, {
          x: (Math.random() - 0.5) * driftForce,
          y: (Math.random() - 0.5) * driftForce,
        });

        // Drift active stem orbs
        stemOrbs.forEach((orb, i) => {
          // Only drift if not dormant (check via constraints)
          if (constraintsRef.current.heartToOrb[i]) {
            Matter.Body.applyForce(orb, orb.position, {
              x: (Math.random() - 0.5) * driftForce * 0.5,
              y: (Math.random() - 0.5) * driftForce * 0.5,
            });
          }
        });

        // Drift destination orbs (always)
        destinationOrbs.forEach((orb) => {
          Matter.Body.applyForce(orb, orb.position, {
            x: (Math.random() - 0.5) * driftForce * 0.4,
            y: (Math.random() - 0.5) * driftForce * 0.4,
          });
        });
      }

      _advancePhysicsTick();
      runnerRef.current = requestAnimationFrame(runPhysics);
    };
    runnerRef.current = requestAnimationFrame(runPhysics);
    setReady(true);

    return () => {
      if (runnerRef.current) cancelAnimationFrame(runnerRef.current);
      Matter.Engine.clear(engine);
      initializedRef.current = false;
    };
  }, [config.width, config.height, config.containerRef]);

  // Re-anchor destination orbs when dimensions change (after initial layout settles).
  useEffect(() => {
    if (!engineRef.current || !bodiesRef.current) return;
    if (config.width === 0 || config.height === 0) return;

    const last = lastDestinationAnchorRef.current;
    if (last && last.width === config.width && last.height === config.height) return;

    const destinationPositions = getDestinationPositions(config.width, config.height);
    const { destinationOrbs } = bodiesRef.current;
    destinationOrbs.forEach((orb, index) => {
      const anchor = destinationPositions[index];
      // Skip if being dragged
      if (draggedBody && draggedBody.id === orb.id) return;
      Matter.Body.setPosition(orb, anchor);
      Matter.Body.setVelocity(orb, { x: 0, y: 0 });
    });

    lastDestinationAnchorRef.current = { width: config.width, height: config.height };
  }, [config.width, config.height, draggedBody]);

  // Reposition entire physics world on container resize
  // Skip if workspace is too small for bodies (radius 60 heart + walls + padding).
  // Below ~250px, walls overlap and the constraint solver diverges → freeze.
  const MIN_PHYSICS_DIM = 280;
  useEffect(() => {
    if (!engineRef.current || !bodiesRef.current) return;
    if (config.width < MIN_PHYSICS_DIM || config.height < MIN_PHYSICS_DIM) return;

    const last = lastDimsRef.current;
    if (!last || (last.width === config.width && last.height === config.height)) return;

    const engine = engineRef.current;
    const { heart, stemOrbs } = bodiesRef.current;
    const scaleX = config.width / last.width;
    const scaleY = config.height / last.height;

    // 1. Recreate walls at new dimensions
    wallsRef.current.forEach(wall => Matter.Composite.remove(engine.world, wall));
    const wt = 50;
    const newWalls = [
      Matter.Bodies.rectangle(config.width / 2, -wt / 2, config.width + 200, wt, { isStatic: true, restitution: 0.9 }),
      Matter.Bodies.rectangle(config.width / 2, config.height + wt / 2, config.width + 200, wt, { isStatic: true, restitution: 0.9 }),
      Matter.Bodies.rectangle(-wt / 2, config.height / 2, wt, config.height + 200, { isStatic: true, restitution: 0.9 }),
      Matter.Bodies.rectangle(config.width + wt / 2, config.height / 2, wt, config.height + 200, { isStatic: true, restitution: 0.9 }),
    ];
    Matter.Composite.add(engine.world, newWalls);
    wallsRef.current = newWalls;

    // 2. Scale body positions proportionally
    if (!draggedBody || draggedBody.id !== heart.id) {
      Matter.Body.setPosition(heart, {
        x: heart.position.x * scaleX,
        y: heart.position.y * scaleY,
      });
      Matter.Body.setVelocity(heart, { x: 0, y: 0 });
    }

    stemOrbs.forEach(orb => {
      if (draggedBody && draggedBody.id === orb.id) return;
      Matter.Body.setPosition(orb, {
        x: orb.position.x * scaleX,
        y: orb.position.y * scaleY,
      });
      Matter.Body.setVelocity(orb, { x: 0, y: 0 });
    });

    // 3. Update corner anchor positions
    const corners = getCornerPositions(config.width, config.height);
    constraintsRef.current.orbToCorner.forEach((c, i) => {
      if (c) c.pointB = corners[i];
    });

    // 4. Update heart anchor to new center
    if (constraintsRef.current.heartAnchor) {
      constraintsRef.current.heartAnchor.pointB = { x: config.width / 2, y: config.height / 2 };
    }

    lastDimsRef.current = { width: config.width, height: config.height };
  }, [config.width, config.height, draggedBody]);

  // Activate an orb
  const activateOrb = useCallback((orbIndex: number) => {
    if (!engineRef.current || !bodiesRef.current) return;
    if (activeOrbs[orbIndex]) return;

    const engine = engineRef.current;
    const { heart, stemOrbs } = bodiesRef.current;
    const orb = stemOrbs[orbIndex];
    const corners = getCornerPositions(configRef.current.width, configRef.current.height);
    const corner = corners[orbIndex];

    // Move orb to corner with impulse
    const dx = corner.x - orb.position.x;
    const dy = corner.y - orb.position.y;
    Matter.Body.setVelocity(orb, { x: dx * 0.03, y: dy * 0.03 });

    // Use current settings for constraint parameters
    const currentSettings = settingsRef.current;

    // Create heart ↔ orb constraint
    const heartConstraint = Matter.Constraint.create({
      bodyA: heart,
      bodyB: orb,
      stiffness: currentSettings.stiffness,
      damping: currentSettings.damping,
      length: PHYSICS.heartToOrb.length,
      render: { visible: false },
    });
    Matter.Composite.add(engine.world, heartConstraint);
    constraintsRef.current.heartToOrb[orbIndex] = heartConstraint;

    // Create orb ↔ corner constraint (respects pinToCorners setting)
    const cornerConstraint = Matter.Constraint.create({
      bodyA: orb,
      pointB: corner,
      stiffness: currentSettings.pinToCorners ? 0.1 : PHYSICS.orbToCorner.stiffness,
      damping: PHYSICS.orbToCorner.damping,
      length: PHYSICS.orbToCorner.length,
      render: { visible: false },
    });
    Matter.Composite.add(engine.world, cornerConstraint);
    constraintsRef.current.orbToCorner[orbIndex] = cornerConstraint;

    setActiveOrbs((prev) => {
      const next = [...prev];
      next[orbIndex] = true;
      return next;
    });
  }, [activeOrbs]);

  // Deactivate an orb (block-centric: return to corner position)
  const deactivateOrb = useCallback((orbIndex: number) => {
    if (!engineRef.current || !bodiesRef.current) return;
    if (!activeOrbs[orbIndex]) return;

    const engine = engineRef.current;
    const orb = bodiesRef.current.stemOrbs[orbIndex];
    // Block-centric: return to corner position instead of dormant cluster
    const corners = getCornerPositions(configRef.current.width, configRef.current.height);
    const homePos = corners[orbIndex];

    // Remove constraints
    const heartConstraint = constraintsRef.current.heartToOrb[orbIndex];
    const cornerConstraint = constraintsRef.current.orbToCorner[orbIndex];

    if (heartConstraint) {
      Matter.Composite.remove(engine.world, heartConstraint);
      constraintsRef.current.heartToOrb[orbIndex] = null;
    }

    if (cornerConstraint) {
      Matter.Composite.remove(engine.world, cornerConstraint);
      constraintsRef.current.orbToCorner[orbIndex] = null;
    }

    // Move orb back to its corner position
    const dx = homePos.x - orb.position.x;
    const dy = homePos.y - orb.position.y;
    Matter.Body.setVelocity(orb, { x: dx * 0.03, y: dy * 0.03 });

    setActiveOrbs((prev) => {
      const next = [...prev];
      next[orbIndex] = false;
      return next;
    });
  }, [activeOrbs]);

  // Move an orb to a specific position (for snap-to-grid)
  const moveOrbTo = useCallback((orbIndex: number, x: number, y: number) => {
    if (!bodiesRef.current) return;
    const orb = bodiesRef.current.stemOrbs[orbIndex];
    if (!orb) return;

    // Stop all motion
    Matter.Body.setVelocity(orb, { x: 0, y: 0 });
    Matter.Body.setAngularVelocity(orb, 0);

    // Teleport to position
    Matter.Body.setPosition(orb, { x, y });
  }, []);

  // Update heart anchor based on active count
  useEffect(() => {
    if (!engineRef.current || !bodiesRef.current) return;

    const engine = engineRef.current;
    const heart = bodiesRef.current.heart;
    const center = {
      x: configRef.current.width / 2,
      y: configRef.current.height / 2,
    };

    // Remove old anchor
    if (constraintsRef.current.heartAnchor) {
      Matter.Composite.remove(engine.world, constraintsRef.current.heartAnchor);
      constraintsRef.current.heartAnchor = null;
    }

    // Calculate stiffness
    const stiffness = Math.min(
      PHYSICS.heartAnchor.maxStiffness,
      PHYSICS.heartAnchor.baseStiffness + activeCount * PHYSICS.heartAnchor.perConnection
    );

    // Adjust friction based on connections
    const baseFriction = 0.008;
    const maxFriction = 0.02;
    heart.frictionAir = baseFriction + (activeCount / 4) * (maxFriction - baseFriction);

    // Create anchor if needed
    if (activeCount > 0 && stiffness > 0) {
      const anchor = Matter.Constraint.create({
        bodyA: heart,
        pointB: center,
        stiffness,
        damping: PHYSICS.heartAnchor.damping,
        length: 0,
        render: { visible: false },
      });
      Matter.Composite.add(engine.world, anchor);
      constraintsRef.current.heartAnchor = anchor;
    }
  }, [activeCount]);

  const isDragging = useCallback(
    (body: Matter.Body) => draggedBody?.id === body.id,
    [draggedBody]
  );

  if (!ready || !engineRef.current || !bodiesRef.current) return null;

  return {
    engine: engineRef.current,
    bodies: bodiesRef.current,
    isDragging,
    draggedBody,
    activateOrb,
    deactivateOrb,
    moveOrbTo,
    activeOrbs,
    activeCount,
    settings,
    updateSettings,
  };
}
