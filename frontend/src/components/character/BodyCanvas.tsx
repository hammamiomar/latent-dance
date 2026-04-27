/**
 * BodyCanvas — R3F background canvas rendering the brushed metal body surface.
 *
 * A single fullscreen plane with a custom shader, positioned behind all HTML.
 * Reads audio + canvas lighting stores via .getState() in useFrame — no
 * React re-renders for uniform updates.
 *
 * Follows the same pattern as BellyScene: orthographic, pointer-events none,
 * alpha-enabled, store reads in useFrame.
 */

import { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

import { bodyVertexShader, bodyFragmentShader } from "../../shaders/bodyShader";
import { useCanvasLightingStore } from "../../stores/useCanvasLightingStore";
import { useAudioActivityStore } from "../../stores/useAudioActivityStore";
import { useAudioStore } from "../../stores/useAudioStore";
import { useLayoutStore } from "../../stores/useLayoutStore";

// =============================================================================
// BODY PLANE — the actual mesh inside the R3F scene
// =============================================================================

function BodyPlane() {
  const meshRef = useRef<THREE.Mesh>(null);
  const { size } = useThree();

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uActivity: { value: 0 },
      uBeatPhase: { value: 0 },
      uBassEnergy: { value: 0 },
      uBellyGlowColor: { value: new THREE.Vector3(0.3, 0.28, 0.22) },
      uBellyGlowBright: { value: 0.4 },
      uFaceGlowColor: { value: new THREE.Vector3(0.29, 0.87, 0.50) },
      uFaceGlowBright: { value: 0.6 },
      uFaceRect: { value: new THREE.Vector4(0.2, 0.0, 0.8, 0.15) },
      uBellyRect: { value: new THREE.Vector4(0.05, 0.2, 0.95, 0.85) },
      uResolution: { value: new THREE.Vector2(size.width, size.height) },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  useFrame((state) => {
    if (!meshRef.current) return;

    const t = state.clock.elapsedTime;
    uniforms.uTime.value = t;

    // Resolution (may change on resize)
    uniforms.uResolution.value.set(state.size.width, state.size.height);

    // Resize plane to match viewport
    meshRef.current.scale.set(state.size.width, state.size.height, 1);

    // --- Canvas lighting store (belly glow) ---
    const lighting = useCanvasLightingStore.getState();
    const [r, g, b] = lighting.dominantColor;
    const target = uniforms.uBellyGlowColor.value;
    target.x = THREE.MathUtils.lerp(target.x, r, 0.08);
    target.y = THREE.MathUtils.lerp(target.y, g, 0.08);
    target.z = THREE.MathUtils.lerp(target.z, b, 0.08);
    uniforms.uBellyGlowBright.value = THREE.MathUtils.lerp(
      uniforms.uBellyGlowBright.value,
      lighting.brightness,
      0.08
    );

    // --- Audio activity store ---
    const activity = useAudioActivityStore.getState();
    const bassEnergy = activity.stems.bass?.energy_smooth ?? 0;
    const drumsEnergy = activity.stems.drums?.energy_smooth ?? 0;
    const vocalsEnergy = activity.stems.vocals?.energy_smooth ?? 0;
    const overallActivity = Math.min(
      1,
      (bassEnergy + drumsEnergy + vocalsEnergy) / 2
    );

    uniforms.uActivity.value = THREE.MathUtils.lerp(
      uniforms.uActivity.value,
      overallActivity,
      0.1
    );
    uniforms.uBassEnergy.value = THREE.MathUtils.lerp(
      uniforms.uBassEnergy.value,
      bassEnergy,
      0.1
    );

    // --- Beat phase from audio time ---
    const audioState = useAudioStore.getState();
    // BPM not directly available — use a moderate default.
    // Beat phase wraps 0-1 based on audio time, approx 120 BPM.
    const bpm = 120;
    const beatPhase = audioState.isPlaying
      ? (audioState.currentTime * bpm / 60) % 1
      : 0;
    uniforms.uBeatPhase.value = beatPhase;

    // --- Layout rects (zone positions) ---
    const layout = useLayoutStore.getState();
    uniforms.uFaceRect.value.set(...layout.faceRect);
    uniforms.uBellyRect.value.set(...layout.bellyRect);
  });

  return (
    <mesh ref={meshRef}>
      <planeGeometry args={[1, 1]} />
      <shaderMaterial
        vertexShader={bodyVertexShader}
        fragmentShader={bodyFragmentShader}
        uniforms={uniforms}
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  );
}

// =============================================================================
// CANVAS WRAPPER
// =============================================================================

export function BodyCanvas() {
  return (
    <Canvas
      orthographic
      camera={{ position: [0, 0, 1], near: 0.1, far: 10, zoom: 1 }}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: -1,
        pointerEvents: "none",
        background: "transparent",
      }}
      gl={{
        alpha: true,
        antialias: false,
        powerPreference: "low-power",
        preserveDrawingBuffer: false,
      }}
      dpr={[1, 1.5]}
      frameloop="always"
    >
      <BodyPlane />
    </Canvas>
  );
}
