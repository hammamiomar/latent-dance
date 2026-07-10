/**
 * useMatterPhysics - Matter.js physics world for the belly creatures.
 *
 * Heart + stem orbs + destination orbs float in liquid; consumers read
 * body positions imperatively (rAF/useFrame) — physics never triggers
 * React renders.
 */

import { useEffect, useRef, useCallback, useState } from "react";
import Matter from "matter-js";
import { slotAnchorPositions } from "../lib/slotLayout";

// ============================================================================
// Types
// ============================================================================

export interface PhysicsConfig {
  width: number;
  height: number;
  /** Steering orb count from the capability manifest; 0 defers world creation. */
  slotCount: number;
  containerRef?: React.RefObject<HTMLElement | null>;
}

export interface PhysicsBodies {
  heart: Matter.Body;
  stemOrbs: Matter.Body[];
  destinationOrbs: Matter.Body[];
}

export interface PhysicsWorld {
  engine: Matter.Engine;
  bodies: PhysicsBodies;
  isDragging: (body: Matter.Body) => boolean;
  draggedBody: Matter.Body | null;
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
    // The retired anchor effect rested the heart at 0.008 — keep that
    // value at creation so drift behavior is unchanged.
    frictionAir: 0.008,
    restitution: 0.5,
    density: 0.0008,
  },

  stemOrb: {
    radius: 40,
    friction: 0.001,       // Same as heart
    frictionAir: 0.004,
    restitution: 0.5,      // Same as heart
    density: 0.0008,       // Same as heart
  },

  mouse: {
    stiffness: 0.03,       // Softer drag feel
    damping: 0.08,
  },
};

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
    destToAnchor: (Matter.Constraint | null)[];
  }>({ destToAnchor: [] });

  const runnerRef = useRef<number | null>(null);
  const wallsRef = useRef<Matter.Body[]>([]);
  const lastDimsRef = useRef<{ width: number; height: number } | null>(null);
  const [draggedBody, setDraggedBody] = useState<Matter.Body | null>(null);

  const [ready, setReady] = useState(false);
  const initializedRef = useRef(false);
  const configRef = useRef(config);
  configRef.current = config;
  const lastDestinationAnchorRef = useRef<{ width: number; height: number } | null>(null);

  // Initialize physics world (waits for both real dimensions and a manifest —
  // slotCount arrives with the capability bootstrap)
  useEffect(() => {
    if (initializedRef.current) return;
    if (config.width === 0 || config.height === 0) return;
    if (config.slotCount <= 0) return;

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

    // One steering orb per manifest slot; body i is slot order[i]. Birth
    // positions come from the shared anchor table (lib/slotLayout).
    const initialPositions = slotAnchorPositions(config.slotCount, config.width, config.height);
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
      destToAnchor: [null, null],
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
        // Values the retired settings system always resolved to at init
        stiffness: 0.002,
        damping: 0.005,
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

        // Drift destination orbs (always)
        destinationOrbs.forEach((orb) => {
          Matter.Body.applyForce(orb, orb.position, {
            x: (Math.random() - 0.5) * driftForce * 0.4,
            y: (Math.random() - 0.5) * driftForce * 0.4,
          });
        });
      }

      runnerRef.current = requestAnimationFrame(runPhysics);
    };
    runnerRef.current = requestAnimationFrame(runPhysics);
    setReady(true);

    return () => {
      if (runnerRef.current) cancelAnimationFrame(runnerRef.current);
      Matter.Engine.clear(engine);
      initializedRef.current = false;
    };
  }, [config.width, config.height, config.slotCount, config.containerRef]);

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

    lastDimsRef.current = { width: config.width, height: config.height };
  }, [config.width, config.height, draggedBody]);

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
  };
}
