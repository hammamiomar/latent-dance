/**
 * BellyScene — Single shared R3F Canvas for all belly 3D content.
 *
 * Consolidates 7 separate WebGL contexts (1 heart + 6 orbs) into ONE.
 * Uses an orthographic camera where 1 unit ≈ 1 pixel.
 * Meshes read Matter.js body positions directly in useFrame (no React re-renders).
 * All interaction (click, drag, hover) stays on HTML overlay divs.
 */

import { useRef, useCallback, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type Matter from "matter-js";

import { CrystalHeartMesh } from "./CrystalHeart";
import { FlowerOrbMesh } from "./FlowerOrb";
import { useCanvasLightingStore } from "../stores/useCanvasLightingStore";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import type { BlockCode, BlockMapping, LinkTarget, StemActivity } from "../types/sae";
import type { Destination, DestinationMode, DestinationSpace } from "../types/destinations";
import { FLOWER_COLORS } from "../data/flowerColors";

// =============================================================================
// Constants
// =============================================================================

/** Scale factors for orthographic projection (derived from original perspective cameras) */
const HEART_ORTHO_SCALE = 48;
const ORB_ORTHO_SCALE = 30;

const BLOCK_ORDER: BlockCode[] = ["down.2.1", "mid.0", "up.0.0", "up.0.1"];

const BLOCK_COLORS: Record<string, string> = {
  "down.2.1": "#c45a2a",
  "mid.0": "#a84070",
  "up.0.0": "#4a9eb0",
  "up.0.1": "#5a8a4a",
};

/** Map link target to physical stem for flower coloring */
function linkTargetToFlowerStem(linkTarget: LinkTarget | undefined): "bass" | "drums" | "vocals" | "other" {
  if (!linkTarget) return "other";
  if (linkTarget.startsWith("drums")) return "drums";
  if (linkTarget.startsWith("other")) return "other";
  if (linkTarget.startsWith("bass")) return "bass";
  if (linkTarget.startsWith("vocals")) return "vocals";
  return "other";
}

// =============================================================================
// Camera Controller — syncs orthographic camera to container size
// =============================================================================

function CameraController({ width, height }: { width: number; height: number }) {
  const { camera, size: canvasSize } = useThree();

  useEffect(() => {
    const cam = camera as THREE.OrthographicCamera;
    // Use the actual Canvas CSS pixel size for the projection, NOT the
    // bellySize prop. If there's any mismatch (timing, rounding, layout),
    // the prop-based camera would shift meshes vs HTML overlays.
    // The physics world still uses width/height for body positions,
    // so we scale: physics coords → canvas pixels.
    const cw = canvasSize.width || width;
    const ch = canvasSize.height || height;

    cam.left = 0;
    cam.right = cw;
    cam.top = 0;
    cam.bottom = -ch;
    cam.near = 0.1;
    cam.far = 200;
    cam.position.set(0, 0, 100);
    cam.lookAt(0, 0, 0);
    cam.updateProjectionMatrix();
  }, [camera, width, height, canvasSize.width, canvasSize.height]);

  return null;
}

// =============================================================================
// Heart Wrapper — positions CrystalHeartMesh at physics body coords
// =============================================================================

interface HeartInSceneProps {
  heartBody: Matter.Body;
  activity: number;
  bpm: number;
  isDragging: boolean;
  isHovered: boolean;
  isPlayerOpen: boolean;
  isReadyToGenerate: boolean;
}

function HeartInScene({
  heartBody,
  activity,
  bpm,
  isDragging,
  isHovered,
  isPlayerOpen,
  isReadyToGenerate,
}: HeartInSceneProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Canvas lighting for video sync
  const canvasColor = useCanvasLightingStore((s) => s.dominantColor);
  const canvasBrightness = useCanvasLightingStore((s) => s.brightness);

  // Position from physics body (mutated in-place by Matter.js, read every frame)
  useFrame(() => {
    if (!groupRef.current) return;
    groupRef.current.position.set(
      heartBody.position.x,
      -heartBody.position.y,
      2 // in front of orbs
    );
  });

  const velocity = Math.sqrt(heartBody.velocity.x ** 2 + heartBody.velocity.y ** 2);

  return (
    <group ref={groupRef} scale={HEART_ORTHO_SCALE}>
      <CrystalHeartMesh
        isPlayerOpen={isPlayerOpen}
        isDragging={isDragging}
        isHovered={isHovered}
        velocity={velocity}
        activity={activity}
        bpm={bpm}
        canvasColor={canvasColor}
        canvasBrightness={canvasBrightness}
        isReadyToGenerate={isReadyToGenerate}
      />
    </group>
  );
}

// =============================================================================
// Orb Wrapper — positions FlowerOrbMesh at physics body coords
// =============================================================================

interface OrbInSceneProps {
  body: Matter.Body;
  color: { r: number; g: number; b: number };
  isBloomed: boolean;
  activity: number;
  transient: number;
  isHovered: boolean;
  /** Physical stem key for live audio reads in useFrame */
  stemKey?: "bass" | "drums" | "vocals" | "other";
}

function OrbInScene({
  body,
  color,
  isBloomed,
  activity,
  transient,
  isHovered,
  stemKey,
}: OrbInSceneProps) {
  const groupRef = useRef<THREE.Group>(null);
  const rotGroupRef = useRef<THREE.Group>(null);

  const canvasDominantColor = useCanvasLightingStore((s) => s.dominantColor);
  const canvasBrightness = useCanvasLightingStore((s) => s.brightness);

  useFrame((state) => {
    if (!groupRef.current || !rotGroupRef.current) return;
    groupRef.current.position.set(
      body.position.x,
      -body.position.y,
      1
    );

    // Shared rotation for flower + calyx (keeps them attached)
    const t = state.clock.elapsedTime;
    rotGroupRef.current.rotation.x = -0.7 + Math.sin(t * 0.2) * 0.05;
    rotGroupRef.current.rotation.y = t * 0.1;
  });

  return (
    <group ref={groupRef} scale={ORB_ORTHO_SCALE}>
      <group ref={rotGroupRef}>
        <FlowerOrbMesh
          color={color}
          isBloomedTarget={isBloomed}
          activity={activity}
          transient={transient}
          isHovered={isHovered}
          canvasDominantColor={canvasDominantColor}
          canvasBrightness={canvasBrightness}
          videoInfluence={0.2}
          skipRotation
          stemKey={stemKey}
        />
      </group>
    </group>
  );
}

// =============================================================================
// Scene Contents — all meshes, rendered inside the R3F Canvas
// =============================================================================

interface SceneContentsProps {
  width: number;
  height: number;
  heartBody: Matter.Body;
  stemOrbBodies: Matter.Body[];
  destinationOrbBodies: Matter.Body[];
  heartActivity: number;
  heartBpm: number;
  heartIsDragging: boolean;
  heartIsHovered: boolean;
  heartIsPlayerOpen: boolean;
  heartIsReadyToGenerate: boolean;
  blockMappings: Record<BlockCode, BlockMapping>;
  stemActivity: StemActivity;
  destinationActivity: number;
  destinationStates: {
    latent: { mode: DestinationMode; destinationA: Destination | null; destinationB: Destination | null };
    prompt: { mode: DestinationMode; destinationA: Destination | null; destinationB: Destination | null };
  };
  orbHoverStates: Record<string, boolean>;
}

function SceneContents({
  width,
  height,
  heartBody,
  stemOrbBodies,
  destinationOrbBodies,
  heartActivity,
  heartBpm,
  heartIsDragging,
  heartIsHovered,
  heartIsPlayerOpen,
  heartIsReadyToGenerate,
  blockMappings,
  stemActivity,
  destinationActivity,
  destinationStates,
  orbHoverStates,
}: SceneContentsProps) {
  // Compute per-orb data
  const getOrbActivity = useCallback(
    (block: BlockCode): number => {
      const mapping = blockMappings[block];
      if (!mapping) return 0;
      const baseStem = linkTargetToFlowerStem(mapping.linkTarget);
      const audioStems = useAudioActivityStore.getState().stems;
      const extended = audioStems[baseStem]?.energy_smooth ?? 0;
      const legacy = stemActivity[baseStem as keyof StemActivity] ?? 0;
      return Math.max(extended, legacy as number);
    },
    [blockMappings, stemActivity]
  );

  const getOrbTransient = useCallback(
    (block: BlockCode): number => {
      const mapping = blockMappings[block];
      if (!mapping) return 0;
      const audioStems = useAudioActivityStore.getState().stems;
      const physicalStem = linkTargetToFlowerStem(mapping.linkTarget);
      return audioStems[physicalStem]?.flash ?? 0;
    },
    [blockMappings]
  );

  const getOverallTransient = useCallback((): number => {
    const audioStems = useAudioActivityStore.getState().stems;
    return Math.max(
      audioStems.bass?.flash ?? 0,
      audioStems.drums?.flash ?? 0,
      audioStems.vocals?.flash ?? 0,
      audioStems.other?.flash ?? 0
    );
  }, []);

  const getOrbColor = useCallback(
    (block: BlockCode): { r: number; g: number; b: number } => {
      const hex = BLOCK_COLORS[block] || "#888888";
      return {
        r: parseInt(hex.slice(1, 3), 16) / 255,
        g: parseInt(hex.slice(3, 5), 16) / 255,
        b: parseInt(hex.slice(5, 7), 16) / 255,
      };
    },
    []
  );

  return (
    <>
      <CameraController width={width} height={height} />

      {/* Crystal Heart */}
      <HeartInScene
        heartBody={heartBody}
        activity={heartActivity}
        bpm={heartBpm}
        isDragging={heartIsDragging}
        isHovered={heartIsHovered}
        isPlayerOpen={heartIsPlayerOpen}
        isReadyToGenerate={heartIsReadyToGenerate}
      />

      {/* Stem Orbs */}
      {stemOrbBodies.map((body, index) => {
        const block = BLOCK_ORDER[index];
        if (!block) return null;
        const mapping = blockMappings[block];

        return (
          <OrbInScene
            key={block}
            body={body}
            color={getOrbColor(block)}
            isBloomed={mapping?.enabled ?? false}
            activity={getOrbActivity(block)}
            transient={getOrbTransient(block)}
            isHovered={orbHoverStates[block] ?? false}
            stemKey={linkTargetToFlowerStem(mapping?.linkTarget)}
          />
        );
      })}

      {/* Destination Orbs */}
      {destinationOrbBodies.map((body, index) => {
        const space: DestinationSpace = index === 0 ? "latent" : "prompt";
        const state = destinationStates[space];
        const isConfigured = state.destinationA !== null && state.destinationB !== null;
        const colorKey = space as keyof typeof FLOWER_COLORS;
        const color = FLOWER_COLORS[colorKey];

        return (
          <OrbInScene
            key={`dest-${space}`}
            body={body}
            color={color}
            isBloomed={isConfigured}
            activity={destinationActivity}
            transient={getOverallTransient()}
            isHovered={orbHoverStates[space] ?? false}
          />
        );
      })}
    </>
  );
}

// =============================================================================
// BellyScene — The single R3F Canvas
// =============================================================================

export type BellySceneProps = SceneContentsProps;

export function BellyScene(props: BellySceneProps) {
  return (
    <Canvas
      orthographic
      camera={{
        position: [0, 0, 100],
        left: 0,
        right: props.width || 1,
        top: 0,
        bottom: -(props.height || 1),
        near: 0.1,
        far: 200,
      }}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 40,
        pointerEvents: "none",
        background: "transparent",
      }}
      gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
    >
      <SceneContents {...props} />
    </Canvas>
  );
}
