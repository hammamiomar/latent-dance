/**
 * Shared GLSL Shader Utilities
 *
 * Export GLSL code chunks as template literals for use in custom shaders.
 * These are the building blocks for CrystalHeart, FlowerOrb, and other effects.
 */

// =============================================================================
// NOISE FUNCTIONS
// =============================================================================

/**
 * Simplex 3D Noise
 * Credit: Ashima Arts (https://github.com/ashima/webgl-noise)
 */
export const NOISE_SIMPLEX_3D = /* glsl */ `
vec3 mod289_3(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289_4(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289_4(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289_3(i);
  vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
`;

/**
 * Fractal Brownian Motion (FBM)
 * Requires: NOISE_SIMPLEX_3D
 */
export const NOISE_FBM = /* glsl */ `
float fbm(vec3 p, int octaves) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;

  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    value += amplitude * snoise(p * frequency);
    frequency *= 2.0;
    amplitude *= 0.5;
  }
  return value;
}

float fbm4(vec3 p) {
  return fbm(p, 4);
}
`;

/**
 * Worley (Cellular) Noise
 * Creates cell-like patterns, good for crystal inclusions
 */
export const NOISE_WORLEY = /* glsl */ `
vec3 hash3(vec3 p) {
  p = vec3(
    dot(p, vec3(127.1, 311.7, 74.7)),
    dot(p, vec3(269.5, 183.3, 246.1)),
    dot(p, vec3(113.5, 271.9, 124.6))
  );
  return fract(sin(p) * 43758.5453123);
}

float worley(vec3 p) {
  vec3 i_st = floor(p);
  vec3 f_st = fract(p);

  float min_dist = 1.0;

  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        vec3 neighbor = vec3(float(x), float(y), float(z));
        vec3 point = hash3(i_st + neighbor);
        vec3 diff = neighbor + point - f_st;
        float dist = length(diff);
        min_dist = min(min_dist, dist);
      }
    }
  }

  return min_dist;
}
`;

/**
 * Flow Noise - Self-advecting noise for organic fluid motion
 * Requires: NOISE_SIMPLEX_3D
 */
export const NOISE_FLOW = /* glsl */ `
float flowNoise(vec3 p, float time) {
  // Calculate gradient of noise at this point
  float eps = 0.01;
  vec3 gradient = vec3(
    snoise(p + vec3(eps, 0.0, 0.0)) - snoise(p - vec3(eps, 0.0, 0.0)),
    snoise(p + vec3(0.0, eps, 0.0)) - snoise(p - vec3(0.0, eps, 0.0)),
    snoise(p + vec3(0.0, 0.0, eps)) - snoise(p - vec3(0.0, 0.0, eps))
  ) / (2.0 * eps);

  // Displace by gradient over time
  vec3 displaced = p + gradient * time * 0.3;
  return snoise(displaced);
}
`;

/**
 * Curl Noise - Divergence-free noise for fluid simulation
 * Requires: NOISE_SIMPLEX_3D
 */
export const NOISE_CURL = /* glsl */ `
vec3 curlNoise(vec3 p) {
  float eps = 0.01;

  // Partial derivatives
  float n1 = snoise(p + vec3(eps, 0.0, 0.0));
  float n2 = snoise(p - vec3(eps, 0.0, 0.0));
  float n3 = snoise(p + vec3(0.0, eps, 0.0));
  float n4 = snoise(p - vec3(0.0, eps, 0.0));
  float n5 = snoise(p + vec3(0.0, 0.0, eps));
  float n6 = snoise(p - vec3(0.0, 0.0, eps));

  float dx = (n3 - n4 - n5 + n6) / (2.0 * eps);
  float dy = (n5 - n6 - n1 + n2) / (2.0 * eps);
  float dz = (n1 - n2 - n3 + n4) / (2.0 * eps);

  return vec3(dx, dy, dz);
}
`;

// =============================================================================
// LIGHTING & MATERIAL FUNCTIONS
// =============================================================================

/**
 * Schlick's Fresnel Approximation
 * More accurate than simple pow() for edge glow
 */
export const FRESNEL_SCHLICK = /* glsl */ `
float fresnelSchlick(float cosTheta, float F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickVec3(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
`;

/**
 * Subsurface Scattering Approximation
 * Light penetrating translucent organic surface
 */
export const SSS_APPROXIMATION = /* glsl */ `
vec3 subsurfaceScattering(
  vec3 lightDir,
  vec3 viewDir,
  vec3 normal,
  float thickness,
  vec3 subsurfaceColor,
  float power
) {
  // Compute half-vector between light and back-projected normal
  vec3 H = normalize(lightDir + normal * 0.5);

  // View-dependent transmission
  float VdotH = pow(max(0.0, dot(viewDir, -H)), power);

  // Thickness attenuation (thicker = less transmission)
  float attenuation = 1.0 - thickness;

  return subsurfaceColor * VdotH * attenuation;
}
`;

/**
 * Thin-Film Iridescence (Physics-Based)
 * Real wavelength-dependent phase shifts like soap bubbles or beetle shells
 */
export const THIN_FILM_IRIDESCENCE = /* glsl */ `
vec3 thinFilmIridescence(float dotVN, float filmThickness, float IOR) {
  // Film thickness in nanometers (typical range: 100-500nm)
  float thickness = filmThickness * 400.0 + 100.0;

  // Optical path difference
  float cosThetaT = sqrt(1.0 - (1.0 - dotVN * dotVN) / (IOR * IOR));
  float pathDiff = 2.0 * thickness * IOR * cosThetaT;

  // Phase per wavelength (RGB approximation: 650nm, 550nm, 450nm)
  float phaseR = mod(pathDiff / 650.0, 1.0) * 6.28318;
  float phaseG = mod(pathDiff / 550.0, 1.0) * 6.28318;
  float phaseB = mod(pathDiff / 450.0, 1.0) * 6.28318;

  // Interference pattern
  return vec3(
    0.5 + 0.5 * cos(phaseR),
    0.5 + 0.5 * cos(phaseG),
    0.5 + 0.5 * cos(phaseB)
  );
}
`;

/**
 * Chromatic Dispersion / Aberration
 * Different refraction for R, G, B channels (diamond fire effect)
 */
export const CHROMATIC_DISPERSION = /* glsl */ `
vec3 chromaticDispersion(
  vec3 viewDir,
  vec3 normal,
  float aberration,
  vec3 envColor
) {
  // Different IOR offsets per channel
  float iorR = 1.0 - aberration * 0.02;
  float iorG = 1.0;
  float iorB = 1.0 + aberration * 0.02;

  // Refracted directions
  vec3 refractR = refract(viewDir, normal, iorR);
  vec3 refractG = refract(viewDir, normal, iorG);
  vec3 refractB = refract(viewDir, normal, iorB);

  // In a full implementation, you'd sample an environment map here
  // For now, we simulate with directional color shifts
  float r = dot(refractR, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
  float g = dot(refractG, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
  float b = dot(refractB, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;

  return envColor * vec3(r, g, b);
}
`;

/**
 * Ward Anisotropic Specular BRDF
 * Stretched highlights along the grain direction — brushed metal, hair, velvet.
 * alphaX = roughness along tangent (grain), alphaY = roughness across (bitangent).
 * Small alphaX + large alphaY = tight line highlight stretched perpendicular to grain.
 */
export const BRDF_WARD_ANISOTROPIC = /* glsl */ `
float wardAnisotropic(
  vec3 N, vec3 V, vec3 L, vec3 T, float alphaX, float alphaY
) {
  vec3 H = normalize(L + V);
  vec3 B = normalize(cross(N, T));

  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);
  float NdotH = max(dot(N, H), 0.001);
  float HdotT = dot(H, T);
  float HdotB = dot(H, B);

  float exponent = -2.0 * (
    (HdotT * HdotT) / (alphaX * alphaX) +
    (HdotB * HdotB) / (alphaY * alphaY)
  ) / (1.0 + NdotH);

  return (1.0 / (4.0 * 3.14159 * alphaX * alphaY * sqrt(NdotL * NdotV)))
    * exp(exponent);
}
`;

// =============================================================================
// COLOR & PALETTE FUNCTIONS
// =============================================================================

/**
 * Earthy Color Palette
 * Cycles through muted forest/amber/teal tones
 */
export const PALETTE_EARTHY = /* glsl */ `
vec3 earthyPalette(float t) {
  vec3 c0 = vec3(0.35, 0.28, 0.18);  // Burnt sienna
  vec3 c1 = vec3(0.25, 0.35, 0.22);  // Forest moss
  vec3 c2 = vec3(0.40, 0.32, 0.20);  // Amber
  vec3 c3 = vec3(0.22, 0.28, 0.25);  // Deep teal

  float segment = t * 4.0;
  int idx = int(floor(segment));
  float frac = fract(segment);

  if (idx == 0) return mix(c0, c1, frac);
  if (idx == 1) return mix(c1, c2, frac);
  if (idx == 2) return mix(c2, c3, frac);
  return mix(c3, c0, frac);
}
`;

/**
 * Emerald Color Palette
 * Deep jade greens with slight variation
 */
export const PALETTE_EMERALD = /* glsl */ `
vec3 emeraldPalette(float t) {
  vec3 deep = vec3(0.04, 0.15, 0.10);    // #0a2619
  vec3 mid = vec3(0.08, 0.25, 0.16);     // #14402a
  vec3 bright = vec3(0.15, 0.40, 0.25);  // #266640
  vec3 jade = vec3(0.20, 0.50, 0.35);    // #338059

  float segment = t * 3.0;
  int idx = int(floor(segment));
  float frac = fract(segment);

  if (idx == 0) return mix(deep, mid, frac);
  if (idx == 1) return mix(mid, bright, frac);
  return mix(bright, jade, frac);
}
`;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Smooth minimum - blends between values smoothly
 * Useful for organic shapes
 */
export const UTIL_SMIN = /* glsl */ `
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
`;

/**
 * Remap value from one range to another
 */
export const UTIL_REMAP = /* glsl */ `
float remap(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
}
`;

/**
 * Smooth pulse - creates a smooth bump at 't = center'
 */
export const UTIL_PULSE = /* glsl */ `
float pulse(float t, float center, float width) {
  float d = abs(t - center);
  return 1.0 - smoothstep(0.0, width, d);
}
`;

// =============================================================================
// COMPOSITE CHUNKS (Common combinations)
// =============================================================================

/**
 * All noise functions combined
 */
export const ALL_NOISE = `
${NOISE_SIMPLEX_3D}
${NOISE_FBM}
${NOISE_WORLEY}
${NOISE_FLOW}
${NOISE_CURL}
`;

/**
 * All lighting functions combined
 */
export const ALL_LIGHTING = `
${FRESNEL_SCHLICK}
${SSS_APPROXIMATION}
${THIN_FILM_IRIDESCENCE}
${CHROMATIC_DISPERSION}
`;

/**
 * Standard vertex shader for orbs/heart
 */
export const STANDARD_VERTEX_SHADER = /* glsl */ `
varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vUv;
varying vec3 vWorldPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  vUv = uv;
  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;
