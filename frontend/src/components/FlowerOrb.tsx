/**
 * FlowerOrb - Blooming Flower Visualization
 *
 * Design Vision:
 * Organic flowers that bloom when enabled/configured. Plant-based visual
 * language where music "stems" are literal stems connecting to flowers.
 *
 * Bloom States:
 * - Unbloomed (bud): Closed petals, dim core - stem disabled or destinations not set
 * - Bloomed: Open petals, bright core - stem enabled or both destinations configured
 * - Active: Flutter + glow response to audio transients (within bloomed state)
 *
 * Shader Features:
 * - Parametric petal geometry via vertex displacement
 * - Bloom uniform controls petal opening (0 = bud, 1 = full bloom)
 * - Fresnel for petal edge glow
 * - Activity-driven flutter and core pulse
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

import { FRESNEL_SCHLICK, SSS_APPROXIMATION } from '../shaders/shaderUtils';
import { useAudioActivityStore } from '../stores/useAudioActivityStore';
import type { AllStems } from '../types/sae';

// =============================================================================
// Stem Colors
// =============================================================================

// =============================================================================
// Vertex Shader - Petal Displacement
// =============================================================================

const flowerVertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uBloom;        // 0 = closed bud, 1 = full bloom
  uniform float uActivity;     // Audio activity level
  uniform float uFlutter;      // Transient-triggered flutter

  varying vec3 vNormal;
  varying vec3 vPosition;
  varying vec2 vUv;
  varying vec3 vViewDir;
  varying float vPetalFactor;   // 0 = valley/base, 1 = petal center at tip
  varying float vHeightFactor;  // 0 = base, 1 = petal tip
  varying float vSepalFactor;   // 0 = not sepal, 1 = sepal center

  #define PETAL_COUNT 8.0
  #define SEPAL_COUNT 5.0
  #define TAU 6.28318530718

  void main() {
    vPosition = position;
    vUv = uv;

    // === 1. PETAL DEFINITION ===
    float angle = atan(position.x, position.z);
    float petalPhase = fract(angle / TAU * PETAL_COUNT + 0.5);
    // 0/1 = edge between petals, 0.5 = center of petal
    float petalCenter = 1.0 - 2.0 * abs(petalPhase - 0.5);
    // smoothstep for soft rounded lobes instead of sharp star points
    float petalShape = smoothstep(0.0, 0.8, petalCenter);

    // Height: 0 at rounded base, 1 at petal rim.
    // Starts at y=0.15 so the lower cup stays solid and round.
    float heightFactor = smoothstep(0.15, 0.95, position.y);
    vHeightFactor = heightFactor;
    vPetalFactor = petalShape * heightFactor;

    // === SEPAL DEFINITION (bottom hemisphere) ===
    // 5 pointed leaves offset from petal alignment, only below equator
    float sepalPhase = fract(angle / TAU * SEPAL_COUNT + 0.3);
    float sepalCenter = 1.0 - 2.0 * abs(sepalPhase - 0.5);
    // Sharper/more pointed than petals
    float sepalShape = smoothstep(0.0, 0.55, sepalCenter);
    // Fade from bottom pole up to just above equator
    float sepalRegion = smoothstep(0.2, -0.4, position.y);
    vSepalFactor = sepalShape * sepalRegion;

    // === 2. START DISPLACEMENT ===
    vec3 displaced = position;
    vec2 radialDir = normalize(position.xz + vec2(0.001));

    // Base bulb: round the bottom
    float bottomSquash = smoothstep(0.1, -0.5, position.y);
    displaced.xz *= 1.0 - bottomSquash * 0.25;
    displaced.y -= bottomSquash * 0.1;

    // === 3. BUD STATE — gentle teardrop ===
    float budClose = 1.0 - uBloom;
    displaced.xz *= 1.0 - budClose * heightFactor * 0.35;
    displaced.y += budClose * heightFactor * 0.2;

    // === 4. BLOOM — open cup with wider mouth ===
    // Linear heightFactor (not quadratic) so the opening spreads evenly.
    float openAngle = uBloom * heightFactor * 1.3;  // ~75° at rim
    displaced.xz += radialDir * sin(openAngle) * 0.55;
    displaced.y -= (1.0 - cos(openAngle)) * 0.18 * heightFactor;

    // Petal centers lean out more than valleys
    float petalBoost = petalShape * uBloom * heightFactor * 0.12;
    displaced.xz += radialDir * petalBoost;

    // === 5. SHUTTER TWIST — petals cascade over each other ===
    // Tangential push (perpendicular to radial in XZ plane).
    // All petals twist the same rotational direction → iris/shutter overlap.
    vec2 tangentDir = vec2(-radialDir.y, radialDir.x);
    float twist = uBloom * heightFactor * petalShape * 0.35;
    displaced.xz += tangentDir * twist;

    // === 6. VALLEY CONTRACTION (bloom-dependent) ===
    float valleyPull = (1.0 - petalShape) * heightFactor * uBloom * 0.5;
    displaced.xz *= 1.0 - valleyPull;

    // === 7. PETAL TIP TAPER ===
    float taper = 1.0 - heightFactor * heightFactor * 0.15;
    displaced.xz *= taper;

    // === 8. SEPALS (pointed green leaves at the base) ===
    // Extend downward and slightly outward when bloomed
    float sepalPull = vSepalFactor * uBloom;
    displaced.y -= sepalPull * 0.35;
    displaced.xz += radialDir * sepalPull * 0.1;
    // Non-sepal base tucks inward when bloomed (narrows the stem attachment)
    float bottomTuck = (1.0 - sepalShape) * sepalRegion * uBloom * 0.12;
    displaced.xz *= 1.0 - bottomTuck;

    // === PETAL TREMBLE (per-petal audio oscillation) ===
    float petalIndex = floor(angle / TAU * PETAL_COUNT);
    float tremble = sin(uTime * 3.0 + petalIndex * 1.3) * uActivity * 0.03;
    displaced.xz *= 1.0 + tremble * petalShape * heightFactor;

    // === FLUTTER (transient response — scales with bloom) ===
    float flutter = uFlutter * uBloom * sin(uTime * 15.0 + position.x * 10.0) * 0.05;
    displaced.xz *= 1.0 + flutter * petalShape * heightFactor;

    // === BREATHING (activity response) ===
    float breathe = sin(uTime * 2.0) * 0.02 * uActivity;
    displaced *= 1.0 + breathe;

    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize(cameraPosition - (modelMatrix * vec4(displaced, 1.0)).xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
  }
`;

// =============================================================================
// Fragment Shader - Flower Coloring
// =============================================================================

const flowerFragmentShader = /* glsl */ `
  uniform float uTime;
  uniform float uBloom;
  uniform float uActivity;
  uniform float uFlutter;
  uniform vec3 uColor;         // Stem-specific color
  uniform float uHover;

  // Video sync
  uniform vec3 uCanvasDominantColor;
  uniform float uCanvasBrightness;
  uniform float uVideoInfluence;

  varying vec3 vNormal;
  varying vec3 vPosition;
  varying vec2 vUv;
  varying vec3 vViewDir;
  varying float vPetalFactor;
  varying float vHeightFactor;
  varying float vSepalFactor;

  ${FRESNEL_SCHLICK}

  ${SSS_APPROXIMATION}

  void main() {
    vec3 viewDir = normalize(vViewDir);
    vec3 normal = normalize(vNormal);
    float NdotV = max(0.0, dot(normal, viewDir));

    // === FRESNEL ===
    float fresnel = fresnelSchlick(NdotV, 0.04);

    // === COLOR ZONES ===
    // Center (pistil) — exposed more as petals open
    float centerFactor = smoothstep(0.3, 0.0, length(vPosition.xz));
    vec3 centerColor = uColor * 1.5 + vec3(0.2, 0.1, 0.0);

    // Petal color gradient: warmer/lighter toward tips
    vec3 tipColor = uColor * 1.3 + vec3(0.1, 0.05, 0.0);
    vec3 petalColor = mix(uColor * 0.85, tipColor, vHeightFactor * vPetalFactor);

    // === BASE COLOR ===
    vec3 baseColor = mix(petalColor, centerColor, centerFactor);

    // Valley shadows — dark interior between petals (AO approximation)
    float valleyShadow = (1.0 - vPetalFactor) * vHeightFactor * 0.4;
    baseColor *= 1.0 - valleyShadow;

    // === BLOOM FADE ===
    // In bud state, everything is darker/greener
    vec3 budColor = vec3(0.12, 0.18, 0.08);
    baseColor = mix(budColor, baseColor, uBloom * 0.8 + 0.2);

    // === SEPAL COLORING ===
    // Dark green leaves at the base, tinted with stem color for cohesion
    vec3 sepalGreen = vec3(0.10, 0.20, 0.06);
    sepalGreen = mix(sepalGreen, uColor * 0.3, 0.25);
    baseColor = mix(baseColor, sepalGreen, vSepalFactor * uBloom * 0.85);

    // === VIDEO TINT ===
    vec3 videoTint = uCanvasDominantColor * uCanvasBrightness;
    baseColor = mix(baseColor, videoTint, uVideoInfluence * uBloom * 0.2);

    // === SSS (petal translucency) ===
    // Light filtering through thin petal surfaces
    vec3 lightDir = normalize(vec3(0.5, 0.8, 0.3));
    float thickness = 1.0 - vPetalFactor * 0.6; // thinner at petal centers
    vec3 sss = subsurfaceScattering(lightDir, viewDir, normal, thickness, uColor * 1.5, 3.0);

    // === GLOW ===
    // Center glow pulses with activity
    float pulseSpeed = 1.5 + uActivity * 3.0;
    float pulse = sin(uTime * pulseSpeed) * 0.5 + 0.5;
    float centerGlow = centerFactor * (0.3 + pulse * 0.4 * uActivity) * uBloom;

    // Edge glow (fresnel rim light)
    float edgeGlow = fresnel * (0.2 + uActivity * 0.3) * uBloom;

    // === COMPOSE ===
    vec3 color = baseColor * (0.6 + uBloom * 0.4);
    color += centerColor * centerGlow;
    color += uColor * edgeGlow;
    color += sss * uBloom * 0.3; // SSS only visible when bloomed

    // Hover highlight
    color += uColor * uHover * 0.15;

    // === ALPHA ===
    // Petal translucency: edges and tips more transparent when unfurled
    float baseAlpha = 0.9 - uBloom * 0.15;
    float petalTranslucency = vPetalFactor * vHeightFactor * uBloom * 0.3;
    float alpha = baseAlpha - petalTranslucency;
    alpha = mix(alpha, 0.95, centerFactor); // Center always solid

    // Valley transparency: when bloomed, gaps between petals become see-through
    float valleyAlpha = smoothstep(0.0, 0.25, vPetalFactor);
    alpha *= mix(1.0, valleyAlpha, uBloom * vHeightFactor);

    // Dormant (unbloomed) is dimmer
    float dormantFade = 0.4 + uBloom * 0.6;
    color *= dormantFade;
    alpha *= dormantFade * 0.8 + 0.2;

    gl_FragColor = vec4(color, alpha);
  }
`;

// =============================================================================
// FlowerOrb Mesh Component
// =============================================================================

export interface FlowerOrbMeshProps {
  color: { r: number; g: number; b: number };
  isBloomedTarget: boolean;
  activity: number;
  transient: number;
  isHovered: boolean;
  canvasDominantColor: [number, number, number];
  canvasBrightness: number;
  videoInfluence: number;
  onClick?: () => void;
  /** Skip internal rotation (parent group handles it) */
  skipRotation?: boolean;
  /** Physical stem key — if provided, reads audio store directly in useFrame (bypasses stale React props) */
  stemKey?: AllStems;
}

export function FlowerOrbMesh({
  color,
  isBloomedTarget,
  activity,
  transient,
  isHovered,
  canvasDominantColor,
  canvasBrightness,
  videoInfluence,
  onClick,
  skipRotation,
  stemKey,
}: FlowerOrbMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uBloom: { value: 0 },
      uActivity: { value: 0 },
      uFlutter: { value: 0 },
      uColor: { value: new THREE.Vector3(color.r, color.g, color.b) },
      uHover: { value: 0 },
      uCanvasDominantColor: { value: new THREE.Vector3(...canvasDominantColor) },
      uCanvasBrightness: { value: canvasBrightness },
      uVideoInfluence: { value: videoInfluence },
    }),
    [color.r, color.g, color.b, canvasDominantColor, canvasBrightness, videoInfluence]
  );

  useFrame((state) => {
    if (!meshRef.current) return;

    const t = state.clock.elapsedTime;
    const mat = meshRef.current.material as THREE.ShaderMaterial;

    // Read live audio directly from store if stemKey provided (bypasses stale React props)
    let liveActivity = activity;
    let liveTransient = transient;
    if (stemKey) {
      const audioStems = useAudioActivityStore.getState().stems;
      liveActivity = audioStems[stemKey]?.energy_smooth ?? 0;
      liveTransient = audioStems[stemKey]?.flash ?? 0;
    }

    mat.uniforms.uTime.value = t;

    // Smooth bloom transition (slower for dramatic effect)
    mat.uniforms.uBloom.value = THREE.MathUtils.lerp(
      mat.uniforms.uBloom.value,
      isBloomedTarget ? 1 : 0,
      0.04
    );

    // Activity lerp
    mat.uniforms.uActivity.value = THREE.MathUtils.lerp(
      mat.uniforms.uActivity.value,
      liveActivity,
      0.1
    );

    // Flutter decay
    const currentFlutter = mat.uniforms.uFlutter.value;
    const targetFlutter = liveTransient > 0.5 ? 1 : 0;
    mat.uniforms.uFlutter.value = THREE.MathUtils.lerp(
      currentFlutter,
      targetFlutter,
      liveTransient > 0.5 ? 0.3 : 0.08 // Fast attack, slow decay
    );

    // Hover
    mat.uniforms.uHover.value = THREE.MathUtils.lerp(
      mat.uniforms.uHover.value,
      isHovered ? 1 : 0,
      0.15
    );

    // Video uniforms
    mat.uniforms.uCanvasDominantColor.value.set(...canvasDominantColor);
    mat.uniforms.uCanvasBrightness.value = canvasBrightness;
    mat.uniforms.uVideoInfluence.value = videoInfluence;

    // Tilt forward ~40° so we look into the bloom, not at the side profile.
    // Gentle nod oscillates around the tilt.
    // When skipRotation is set, the parent group handles rotation (for calyx sync).
    if (!skipRotation) {
      meshRef.current.rotation.x = -0.7 + Math.sin(t * 0.2) * 0.05;
      meshRef.current.rotation.y = t * 0.1;
    }
  });

  return (
    <mesh ref={meshRef} onClick={onClick}>
      <sphereGeometry args={[1, 48, 48]} />
      <shaderMaterial
        vertexShader={flowerVertexShader}
        fragmentShader={flowerFragmentShader}
        uniforms={uniforms}
        transparent
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// =============================================================================
// Stem type shared with OrbSystem labeling
// =============================================================================

export type FlowerStemType = 'bass' | 'drums' | 'vocals' | 'other';
