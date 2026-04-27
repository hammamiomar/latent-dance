/**
 * Body Shader — Brushed dark aluminum surface for the character body.
 *
 * Replaces the CSS Simurai gradient stack + SVG feDiffuseLighting with
 * real WebGL: Ward anisotropic BRDF, FBM bump mapping, procedural
 * machining grain, screen glow illumination, thin-film iridescence,
 * Fresnel rim darkening, audio-reactive breathing.
 *
 * Renders on a single fullscreen plane behind all HTML.
 */

import {
  NOISE_SIMPLEX_3D,
  NOISE_FBM,
  FRESNEL_SCHLICK,
  THIN_FILM_IRIDESCENCE,
  BRDF_WARD_ANISOTROPIC,
} from "./shaderUtils";

// =============================================================================
// VERTEX SHADER
// =============================================================================

export const bodyVertexShader = /* glsl */ `
varying vec2 vUv;
varying vec3 vWorldPos;

void main() {
  // Flip Y so (0,0) = top-left, matching screen / getBoundingClientRect coords
  vUv = vec2(uv.x, 1.0 - uv.y);
  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// =============================================================================
// FRAGMENT SHADER
// =============================================================================

export const bodyFragmentShader = /* glsl */ `
precision highp float;

// --- Uniforms ---
uniform float uTime;
uniform float uActivity;
uniform float uBeatPhase;
uniform float uBassEnergy;

uniform vec3  uBellyGlowColor;
uniform float uBellyGlowBright;
uniform vec3  uFaceGlowColor;
uniform float uFaceGlowBright;

uniform vec4  uFaceRect;   // left, top, right, bottom (0-1)
uniform vec4  uBellyRect;  // left, top, right, bottom (0-1)

uniform vec2  uResolution;

// --- Varyings ---
varying vec2 vUv;
varying vec3 vWorldPos;

// --- Library chunks ---
${NOISE_SIMPLEX_3D}
${NOISE_FBM}
${FRESNEL_SCHLICK}
${THIN_FILM_IRIDESCENCE}
${BRDF_WARD_ANISOTROPIC}

// ==========================================================================
// HELPERS
// ==========================================================================

// Smooth rectangular mask — 1 inside rect, 0 outside, smooth edges
float rectMask(vec2 uv, vec4 rect, float feather) {
  float l = smoothstep(rect.x - feather, rect.x + feather, uv.x);
  float r = 1.0 - smoothstep(rect.z - feather, rect.z + feather, uv.x);
  float t = smoothstep(rect.y - feather, rect.y + feather, uv.y);
  float b = 1.0 - smoothstep(rect.w - feather, rect.w + feather, uv.y);
  return l * r * t * b;
}

// Distance from point to rect center, normalized by rect size
float distToRect(vec2 uv, vec4 rect) {
  vec2 center = (rect.xy + rect.zw) * 0.5;
  vec2 halfSize = (rect.zw - rect.xy) * 0.5;
  vec2 d = (uv - center) / max(halfSize, vec2(0.001));
  return length(d);
}


// ==========================================================================
// MAIN
// ==========================================================================

void main() {
  vec2 uv = vUv;

  // Correct for non-square aspect ratio in noise sampling
  float aspect = uResolution.x / uResolution.y;
  vec2 uvAspect = vec2(uv.x * aspect, uv.y);

  // --- 1. Base color (warm dark gunmetal with vertical gradient) ---
  vec3 color = mix(
    vec3(0.149, 0.141, 0.129),  // #262422 top
    vec3(0.118, 0.110, 0.102),  // #1e1c1a bottom
    uv.y
  );
  // Slightly lighter midband
  color += vec3(0.012) * (1.0 - abs(uv.y - 0.45) * 2.5);

  // --- 2. FBM bump mapping → perturbed normals ---
  // Directional grain: high frequency along Y (machining lines), low along X
  float grainScale = 160.0;
  vec3 grainP = vec3(uvAspect.x * 4.0, uvAspect.y * grainScale, 0.0);
  float grainHeight = snoise(grainP) * 0.5
                    + snoise(grainP * 2.1 + 13.7) * 0.25
                    + snoise(grainP * 4.3 + 27.1) * 0.125;

  // Micro-noise (surface imperfections)
  float microNoise = fbm(vec3(uvAspect * 80.0, 0.0), 3);
  float totalHeight = grainHeight * 0.015 + microNoise * 0.008;

  // Perturb normal via screen-space derivatives
  float dhdx = dFdx(totalHeight);
  float dhdy = dFdy(totalHeight);
  vec3 N = normalize(vec3(-dhdx * 6.0, -dhdy * 6.0, 1.0));

  // --- 3. Ward anisotropic specular ---
  vec3 V = vec3(0.0, 0.0, 1.0);  // Orthographic: camera looks straight at plane
  vec3 T = normalize(vec3(1.0, 0.0, 0.0));  // Grain direction: horizontal machining

  // Overhead key light (slightly from top-left)
  vec3 L1 = normalize(vec3(-0.2, -0.4, 1.0));
  float spec1 = wardAnisotropic(N, V, L1, T, 0.12, 0.55);

  // Fill light (from bottom-right, dimmer)
  vec3 L2 = normalize(vec3(0.3, 0.3, 1.0));
  float spec2 = wardAnisotropic(N, V, L2, T, 0.15, 0.6) * 0.3;

  // Metallic F0 for aluminum: ~0.91-0.92
  vec3 metalColor = vec3(0.91, 0.92, 0.92);
  float fresnel = fresnelSchlick(max(dot(V, N), 0.0), 0.91);

  color += metalColor * (spec1 + spec2) * 0.07 * (0.8 + fresnel * 0.2);

  // Activity-driven specular boost
  color += metalColor * (spec1 + spec2) * 0.02 * uActivity;

  // --- 4. Subtle machining line modulation ---
  // Very fine horizontal lines at prime pixel periods (like Simurai CSS trick)
  float line1 = step(0.92, fract(uv.y * uResolution.y / 3.0));
  float line2 = step(0.94, fract(uv.y * uResolution.y / 5.0));
  float line3 = step(0.95, fract(uv.y * uResolution.y / 7.0));
  color += vec3(0.018) * line1;
  color -= vec3(0.012) * line2;
  color += vec3(0.010) * line3;

  // --- 5. Thin-film iridescence (very subtle) ---
  float cosTheta = max(dot(V, N), 0.0);
  vec3 iridescence = thinFilmIridescence(cosTheta, 0.35, 1.5);
  color += (iridescence - 0.5) * 0.015;

  // --- 6. Screen glow illumination ---
  // Belly glow: distance-attenuated colored light
  float bellyDist = distToRect(uv, uBellyRect);
  float bellyGlow = exp(-bellyDist * 1.8) * uBellyGlowBright;
  color += uBellyGlowColor * bellyGlow * 0.12;

  // Face glow: green phosphor
  float faceDist = distToRect(uv, uFaceRect);
  float faceGlow = exp(-faceDist * 2.2) * uFaceGlowBright;
  color += uFaceGlowColor * faceGlow * 0.08;

  // --- 7. Fresnel edge darkening + vignette ---
  vec2 centered = uv * 2.0 - 1.0;
  float edgeDist = length(centered * vec2(0.8, 1.0));
  float vignette = 1.0 - smoothstep(0.5, 1.2, edgeDist) * 0.2;

  // Top heavier (overhead rack lighting)
  vignette -= (1.0 - smoothstep(0.0, 0.08, uv.y)) * 0.15;
  // Bottom subtle
  vignette -= smoothstep(0.95, 1.0, uv.y) * 0.08;

  color *= vignette;

  // --- 8. Speakers + vents handled by CSS (beveled housings) ---

  // --- 9. Audio-reactive breathing ---
  float breathe = sin(uBeatPhase * 6.28318) * 0.5 + 0.5;
  color += vec3(0.015, 0.012, 0.008) * breathe * uActivity;

  // Bass warmth
  color += vec3(0.008, 0.004, 0.001) * uBassEnergy;

  // --- 10. 3D bevel at body edges ---
  float bevelWidth = 0.004;
  float bevelL = 1.0 - smoothstep(0.0, bevelWidth, uv.x);
  float bevelT = 1.0 - smoothstep(0.0, bevelWidth, uv.y);
  float bevelR = 1.0 - smoothstep(0.0, bevelWidth, 1.0 - uv.x);
  float bevelB = 1.0 - smoothstep(0.0, bevelWidth, 1.0 - uv.y);
  color += vec3(0.03) * (bevelL + bevelT);   // Light edges (top-left)
  color -= vec3(0.06) * (bevelR + bevelB);   // Dark edges (bottom-right)

  // --- 11. FBM surface variation (replaces SVG feTurbulence) ---
  color *= 1.0 + microNoise * 0.04;

  gl_FragColor = vec4(color, 1.0);
}
`;
