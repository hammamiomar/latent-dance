/**
 * CrystalHeart - Geometric emerald gem heart at the center.
 *
 * The heart is the central processor in the visual pipeline:
 * Orbs (detection) → Tendrils (data flow) → Heart (processing) → Canvas (output)
 *
 * Design: Faceted emerald gemstone with internal "jardin" (inclusions),
 * slow pulsing veins synced to audio, BPM-driven heartbeat.
 */

import { useRef, useMemo, useState, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import type Matter from "matter-js";
import {
  NOISE_SIMPLEX_3D,
  FRESNEL_SCHLICK,
} from "../shaders/shaderUtils";
import { useAudioStore } from "../stores/useAudioStore";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";

// ============================================================================
// Types
// ============================================================================

interface CrystalHeartProps {
  body: Matter.Body;
  isDragging: boolean;
  onClick?: () => void;
  /** Overall audio activity level (0-1), used for glow intensity */
  activity?: number;
}

// ============================================================================
// Constants
// ============================================================================

const SIZE = 140; // Slightly larger for the gem heart
const HEART_MODEL_PATH = "/glb/crystalGemHeart.glb";

// Preload the model for faster initial render
useGLTF.preload(HEART_MODEL_PATH);

// ============================================================================
// Emerald Heart Shader
// ============================================================================

const emeraldVertexShader = /* glsl */ `
varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vUv;
varying vec3 vWorldPosition;
varying vec3 vViewDir;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  vUv = uv;
  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
  vViewDir = normalize(cameraPosition - vWorldPosition);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const emeraldFragmentShader = /* glsl */ `
uniform float uTime;
uniform float uActivity;
uniform float uBeatPhase;
uniform float uDragging;
uniform float uHover;
uniform vec3 uCanvasColor;
uniform float uCanvasBrightness;
uniform float uVideoInfluence;
uniform float uReadyToGenerate;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vUv;
varying vec3 vWorldPosition;
varying vec3 vViewDir;

// === LIGHTWEIGHT NOISE (single snoise only) ===
${NOISE_SIMPLEX_3D}

// === LIGHTING FUNCTIONS ===
${FRESNEL_SCHLICK}

// === MAIN (OPTIMIZED) ===
void main() {
  // Flip normal for backfaces (fixes inverted shading on some GLB models)
  vec3 normal = normalize(vNormal);
  if (!gl_FrontFacing) {
    normal = -normal;
  }
  vec3 viewDir = normalize(vViewDir);

  // === BASE EMERALD COLOR ===
  vec3 emeraldDeep = vec3(0.03, 0.12, 0.08);
  vec3 emeraldMid = vec3(0.06, 0.20, 0.12);
  vec3 emeraldBright = vec3(0.12, 0.35, 0.20);
  vec3 jadeGlow = vec3(0.25, 0.55, 0.35);

  // Gradient based on position (lighter at top)
  float heightGradient = vPosition.y * 0.5 + 0.5;
  vec3 baseColor = mix(emeraldDeep, emeraldMid, heightGradient);

  // === SIMPLE INTERNAL VARIATION (1 snoise instead of 2 worley + fbm) ===
  float variation = snoise(vPosition * 3.0) * 0.5 + 0.5;
  baseColor = mix(baseColor, emeraldMid * 0.7, variation * 0.3);

  // === VEIN PULSE (simplified - 1 snoise) ===
  float veinNoise = snoise(vPosition * 4.0 + uTime * 0.05);
  float veinPulse = sin(uTime * 2.0 + veinNoise * 6.28) * 0.5 + 0.5;
  float veinIntensity = smoothstep(0.3, 0.7, veinNoise) * (0.1 + uActivity * 0.4 + uBeatPhase * 0.2);
  baseColor += jadeGlow * veinIntensity * veinPulse;

  // === FRESNEL (Edge glow) ===
  float cosTheta = max(0.0, dot(viewDir, normal));
  float fresnel = fresnelSchlick(cosTheta, 0.04);
  baseColor += jadeGlow * fresnel * (0.4 + uActivity * 0.3);

  // === SIMPLE IRIDESCENCE (no noise lookup, use position directly) ===
  float filmPhase = dot(vPosition, vec3(1.0)) * 10.0;
  vec3 iridescence = vec3(
    0.5 + 0.5 * sin(filmPhase + cosTheta * 6.28),
    0.5 + 0.5 * sin(filmPhase + cosTheta * 6.28 + 2.09),
    0.5 + 0.5 * sin(filmPhase + cosTheta * 6.28 + 4.18)
  );
  baseColor += iridescence * fresnel * 0.1;

  // === SIMPLE SSS (no function call, inline) ===
  vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
  float sss = pow(max(0.0, dot(viewDir, -lightDir)), 3.0) * 0.3;
  baseColor += jadeGlow * sss * (0.2 + uActivity * 0.3);

  // === FACET SPECULAR ===
  vec3 halfDir = normalize(lightDir + viewDir);
  float specular = pow(max(0.0, dot(normal, halfDir)), 32.0);
  baseColor += vec3(0.8, 1.0, 0.9) * specular * 0.5;

  // === VIDEO TINT ===
  vec3 videoTint = uCanvasColor * uCanvasBrightness;
  baseColor = mix(baseColor, baseColor + videoTint * 0.3, uVideoInfluence);

  // === INTERACTION + HEARTBEAT ===
  baseColor += jadeGlow * (uDragging * 0.15 + uHover * 0.1);
  float heartbeat = sin(uBeatPhase * 6.28318) * 0.5 + 0.5;
  baseColor += emeraldBright * heartbeat * uActivity * 0.1;

  // === READY TO GENERATE GLOW ===
  if (uReadyToGenerate > 0.5) {
    vec3 readyGlow = vec3(0.9, 0.75, 0.3);  // Golden amber
    float readyPulse = 0.8 + sin(uTime * 3.0) * 0.2;
    baseColor += readyGlow * 0.15 * readyPulse * fresnel;
  }

  // === ALPHA ===
  float alpha = 0.75 + (1.0 - fresnel) * 0.15;
  gl_FragColor = vec4(baseColor, alpha);
}
`;

// ============================================================================
// Crystal Heart Mesh Component (R3F)
// ============================================================================

export interface CrystalHeartMeshProps {
  isPlayerOpen: boolean;
  isDragging: boolean;
  isHovered: boolean;
  velocity: number;
  activity: number;
  bpm: number;
  canvasColor: [number, number, number];
  canvasBrightness: number;
  isReadyToGenerate: boolean;
}

export function CrystalHeartMesh({
  isPlayerOpen: _isPlayerOpen, // Reserved for future use
  isDragging,
  isHovered,
  velocity,
  activity,
  bpm,
  canvasColor,
  canvasBrightness,
  isReadyToGenerate,
}: CrystalHeartMeshProps) {
  void _isPlayerOpen; // Suppress unused warning
  const meshRef = useRef<THREE.Mesh>(null);

  // Load GLB model and extract geometry
  const { scene } = useGLTF(HEART_MODEL_PATH);
  const geometry = useMemo(() => {
    // Find the first mesh in the loaded scene
    const meshes: THREE.Mesh[] = [];
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        meshes.push(child);
      }
    });

    if (meshes.length === 0) {
      return new THREE.SphereGeometry(0.5, 32, 32); // Fallback
    }

    const geo = meshes[0].geometry.clone();

    // Ensure normals exist for shader
    if (!geo.attributes.normal) {
      geo.computeVertexNormals();
    }

    // Center and normalize the geometry
    geo.center();
    geo.computeBoundingBox();
    const box = geo.boundingBox!;
    const maxDim = Math.max(
      box.max.x - box.min.x,
      box.max.y - box.min.y,
      box.max.z - box.min.z
    );
    // Scale to fit nicely (target ~1.5 units for the heart)
    const scale = 1.5 / maxDim;
    geo.scale(scale, scale, scale);

    return geo;
  }, [scene]);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uActivity: { value: 0 },
      uBeatPhase: { value: 0 },
      uDragging: { value: 0 },
      uHover: { value: 0 },
      uCanvasColor: { value: new THREE.Vector3(0.1, 0.2, 0.1) },
      uCanvasBrightness: { value: 0.5 },
      uVideoInfluence: { value: 0.25 }, // 25% default influence
      uReadyToGenerate: { value: 0 },
    }),
    []
  );

  useFrame((state) => {
    if (!meshRef.current) return;

    const t = state.clock.elapsedTime;

    // Read live audio directly from store (bypasses stale React props)
    const audioStems = useAudioActivityStore.getState().stems;
    const liveActivity = Math.max(
      activity,
      audioStems.bass?.energy_smooth ?? 0,
      audioStems.drums?.energy_smooth ?? 0,
      audioStems.vocals?.energy_smooth ?? 0,
      audioStems.other?.energy_smooth ?? 0,
    );

    // Compute beat phase from audio store (no React re-renders)
    const currentTime = useAudioStore.getState().currentTime;
    const beatPhase = bpm > 0 ? (currentTime * bpm / 60) % 1 : 0;

    // Update uniforms with smooth lerping
    uniforms.uTime.value = t;
    uniforms.uActivity.value = THREE.MathUtils.lerp(uniforms.uActivity.value, liveActivity, 0.1);
    uniforms.uBeatPhase.value = beatPhase;
    uniforms.uDragging.value = THREE.MathUtils.lerp(uniforms.uDragging.value, isDragging ? 1 : 0, 0.15);
    uniforms.uHover.value = THREE.MathUtils.lerp(uniforms.uHover.value, isHovered ? 1 : 0, 0.15);
    uniforms.uCanvasColor.value.set(canvasColor[0], canvasColor[1], canvasColor[2]);
    uniforms.uCanvasBrightness.value = canvasBrightness;
    uniforms.uReadyToGenerate.value = THREE.MathUtils.lerp(
      uniforms.uReadyToGenerate.value,
      isReadyToGenerate ? 1 : 0,
      0.1
    );

    // === ROTATION ===
    // Slow continuous spin (0.1 rad/s)
    meshRef.current.rotation.y = t * 0.1;
    // Gentle wobble
    meshRef.current.rotation.x = Math.sin(t * 0.3) * 0.05;
    meshRef.current.rotation.z = Math.cos(t * 0.25) * 0.03;

    // === HEARTBEAT SCALE ===
    // Pulse on beat (±3% scale)
    const heartbeatScale = 1.0 + Math.sin(beatPhase * Math.PI * 2) * 0.03 * liveActivity;

    // Breathing animation
    const breatheSpeed = 1.5 + liveActivity * 1.0 + velocity * 0.3;
    const breatheAmp = 0.015 + (isDragging ? 0.01 : 0) + liveActivity * 0.01;
    const breathe = 1 + Math.sin(t * breatheSpeed) * breatheAmp;

    meshRef.current.scale.setScalar(heartbeatScale * breathe);
  });

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <shaderMaterial
        vertexShader={emeraldVertexShader}
        fragmentShader={emeraldFragmentShader}
        uniforms={uniforms}
        transparent
        side={THREE.FrontSide}
        depthWrite={true}
      />
    </mesh>
  );
}

// ============================================================================
// Main CrystalHeart Component
// ============================================================================

export function CrystalHeart({
  body,
  isDragging,
  onClick,
  activity = 0,
}: CrystalHeartProps) {

  // Imperative DOM refs — position updated by rAF, not React renders
  const rootRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);

  // rAF loop: write left/top/glow imperatively (zero re-renders for physics)
  useEffect(() => {
    const tick = () => {
      if (rootRef.current) {
        rootRef.current.style.left = `${body.position.x - SIZE / 2}px`;
        rootRef.current.style.top = `${body.position.y - SIZE / 2}px`;
      }
      if (glowRef.current) {
        const vel = Math.sqrt(body.velocity.x ** 2 + body.velocity.y ** 2);
        const alpha = (0.12 + activity * 0.1 + vel * 0.02).toFixed(3);
        glowRef.current.style.background =
          `radial-gradient(circle, rgba(74,158,94,${alpha}) 0%, rgba(74,138,74,0.04) 50%, transparent 70%)`;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [body, activity]);

  // Track mouse down position to differentiate click vs drag
  const mouseDownPos = useRef<{ x: number; y: number } | null>(null);
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseDown = (e: React.MouseEvent) => {
    mouseDownPos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!mouseDownPos.current) return;

    // Check if this was a click (minimal movement) vs a drag
    const dx = e.clientX - mouseDownPos.current.x;
    const dy = e.clientY - mouseDownPos.current.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // If moved less than 5 pixels, treat as click
    if (distance < 5) {
      onClick?.();
    }

    mouseDownPos.current = null;
  };

  return (
    <div
      ref={rootRef}
      className="crystal-heart"
      style={{
        position: "absolute",
        // left/top set imperatively by rAF loop
        width: SIZE,
        height: SIZE,
        zIndex: 100,
        cursor: isDragging ? "grabbing" : isHovered ? "pointer" : "grab",
      }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Ambient glow effect */}
      <div
        ref={glowRef}
        className="crystal-heart__glow"
        style={{
          position: "absolute",
          inset: -30,
          borderRadius: "50%",
          // Initial gradient; updated imperatively by rAF
          background: "radial-gradient(circle, rgba(74,158,94,0.12) 0%, rgba(74,138,74,0.04) 50%, transparent 70%)",
          pointerEvents: "none",
          opacity: isDragging || isHovered ? 1 : 0.7,
          transition: "opacity 0.3s ease",
        }}
      />

      {/* Click hint */}
      {isHovered && !isDragging && (
        <div
          style={{
            position: "absolute",
            bottom: -28,
            left: "50%",
            transform: "translateX(-50%)",
            fontSize: "9px",
            color: "var(--color-text-muted)",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            textShadow: "0 1px 2px rgba(0,0,0,0.8)",
          }}
        >
          click to open player
        </div>
      )}
    </div>
  );
}
