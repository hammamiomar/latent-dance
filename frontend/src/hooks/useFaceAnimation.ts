/**
 * useFaceAnimation — The brain of hamba's face.
 *
 * Pure imperative animation hook. No React re-renders. All mutable state
 * lives in useRef. The textmode draw loop calls tick(secs) each frame and
 * receives a FaceState snapshot describing what to render.
 *
 * State machine: BOOT → IDLE ↔ PERK_UP → LISTENING → (DISCONNECT → IDLE)
 * Listening has 4 sub-states: BOBBING, TENSION, SWAYING, ECSTATIC.
 *
 * Audio data read imperatively from Zustand stores (same pattern as
 * OrbSystem and PlantStems).
 */

import { useRef, useCallback, type MutableRefObject } from "react";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import { useAgentStore } from "../stores/useAgentStore";
import { useSessionStore } from "../stores/useSessionStore";
import { petBridge } from "../shared/petBridge";

// =============================================================================
// TYPES
// =============================================================================

export type EyeShape = "hidden" | "closed" | "half" | "normal" | "wide" | "happy";
export type MouthShape =
  | "hidden"
  | "neutral"
  | "smile"
  | "open"
  | "excited"
  | "contemplative";
export type FacePhase =
  | "BOOT"
  | "IDLE"
  | "PERK_UP"
  | "LISTENING"
  | "DISCONNECT";
export type ListenSubState = "BOBBING" | "TENSION" | "SWAYING" | "ECSTATIC";

export interface FaceState {
  leftEye: EyeShape;
  rightEye: EyeShape;
  lookX: number; // -1 to +1, continuous
  lookY: number; // -1 to +1, subtle
  pupilDilation: number; // 0 to 1
  mouth: MouthShape;
  mouthOpenness: number; // 0–1 for 'open'/'excited' height
  brightness: number; // 0–1, global multiplier
  phase: FacePhase;
  subState?: ListenSubState;
  glitch?: boolean;
  sparkle?: number; // 0–1 sparkle intensity (scattered bright chars)
}

// =============================================================================
// PET INTERACTION — Mutable bridge written by FaceRenderer, read by tick().
// =============================================================================

export interface PetInput {
  /** Mouse position normalized to face container (0–1) */
  normalizedX: number;
  normalizedY: number;
  /** Whether mouse is currently inside the pet zone */
  inZone: boolean;
  /** Accumulated mouse movement speed (EMA-smoothed pixels/event) */
  moveSpeed: number;
  /** Timestamp of last mouse move (performance.now()) */
  lastMoveTime: number;
  /** Current pet intensity (0–1), written by tick for other hooks to read */
  intensity: number;
}

// =============================================================================
// CONSTANTS
// =============================================================================

// Phase durations
const BOOT_DURATION = 2.5;
const PERK_UP_DURATION = 0.5;
const DISCONNECT_DURATION = 1.0;
const SILENCE_TIMEOUT = 2.0;

// Audio smoothing (asymmetric EMA — same DNA as bridge/physics.py)
const ATTACK = 0.3; // Fast response to increases
const RELEASE = 0.05; // Slow decay

// Bob amplitude ramping (momentum — buddy doesn't stop grooving instantly)
const RAMP_UP = 0.02; // ~1.5s to reach target at 30fps
const RAMP_DOWN = 0.008; // ~3s to decay

// Sub-state detection thresholds
const PERK_UP_ENERGY = 0.15;
const BOBBING_ENERGY = 0.3;
const TENSION_ENERGY = 0.5;
const ECSTATIC_ENERGY = 0.8;
const SWAYING_MIN = 0.05;
const SWAYING_MAX = 0.3;
const TRANSIENT_FREQUENT = 0.3;
const TRANSIENT_RARE = 0.1;

// Hysteresis (prevents flickering between sub-states)
const NORMAL_HYSTERESIS = 1.0;
const SEEK_HYSTERESIS = 0.3;
const SEEK_THRESHOLD = 1.0; // seconds jump → seek detected

// Idle smoothing rate (per-frame at 30fps, ~250ms time constant)
const IDLE_SMOOTH = 0.12;

// Pet interaction
const PET_SPEED_THRESHOLD = 2.0; // px/event minimum to count as "petting"
const PET_RAMP_UP = 0.015; // ~2s to peak at 30fps
const PET_RAMP_DOWN = 0.006; // ~5.5s to fully decay
const PET_DECAY_DELAY = 0.3; // seconds before decay starts
const PET_BLISS_THRESHOLD = 0.85;
const PET_BLISS_DURATION = 1.5;

// =============================================================================
// IDLE BEAT POOL — Weighted random selection, no repeats
// =============================================================================

interface IdleBeat {
  name: string;
  duration: number;
  weight: number;
}

const IDLE_BEATS: IdleBeat[] = [
  { name: "lookLeft", duration: 0.8, weight: 12 },
  { name: "lookRight", duration: 0.8, weight: 12 },
  { name: "lookDown", duration: 0.6, weight: 8 },
  { name: "center", duration: 0.6, weight: 13 },
  { name: "blink", duration: 0.15, weight: 20 },
  { name: "doubleBlink", duration: 0.3, weight: 10 },
  { name: "wideEyes", duration: 0.4, weight: 5 },
  { name: "smile", duration: 1.0, weight: 10 },
  { name: "settle", duration: 0.5, weight: 5 },
  { name: "neutral", duration: 0.5, weight: 5 },
];

// =============================================================================
// HELPERS
// =============================================================================

/** Asymmetric EMA — fast attack, slow release. Bridges 10Hz audio → 30fps face. */
function smooth(current: number, target: number): number {
  const alpha = target > current ? ATTACK : RELEASE;
  return current * (1 - alpha) + target * alpha;
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Pick a random beat from the pool, excluding the given index. */
function pickWeightedBeat(exclude: number): number {
  let totalWeight = 0;
  for (let i = 0; i < IDLE_BEATS.length; i++) {
    if (i !== exclude) totalWeight += IDLE_BEATS[i].weight;
  }
  let r = Math.random() * totalWeight;
  for (let i = 0; i < IDLE_BEATS.length; i++) {
    if (i === exclude) continue;
    r -= IDLE_BEATS[i].weight;
    if (r <= 0) return i;
  }
  return 0;
}

/** Read isReceiving + stems imperatively from the audio store. */
function checkMusicDetection(): boolean {
  const store = useAudioActivityStore.getState();
  if (!store.isReceiving) return false;
  const { stems } = store;
  const maxEnergy = Math.max(
    stems.drums?.energy_smooth ?? 0,
    stems.bass?.energy_smooth ?? 0,
    stems.vocals?.energy_smooth ?? 0,
  );
  return maxEnergy > PERK_UP_ENERGY;
}

/** Classify global audio signal into a listening sub-state. */
function detectSubState(audio: SmoothedAudio): ListenSubState {
  const { overallEnergy, avgEnergy, transientRate } = audio;
  // Order matters — ECSTATIC first (highest energy), then narrow down
  if (overallEnergy > ECSTATIC_ENERGY && transientRate > TRANSIENT_FREQUENT)
    return "ECSTATIC";
  if (overallEnergy > TENSION_ENERGY && transientRate < TRANSIENT_RARE)
    return "TENSION";
  if (transientRate > TRANSIENT_FREQUENT && avgEnergy > BOBBING_ENERGY)
    return "BOBBING";
  if (avgEnergy > SWAYING_MIN && avgEnergy < SWAYING_MAX) return "SWAYING";
  return "BOBBING"; // default dance state
}

// =============================================================================
// INTERNAL STATE TYPES
// =============================================================================

interface SmoothedAudio {
  overallEnergy: number;
  avgEnergy: number;
  transientRate: number;
}

interface FaceRefs {
  phase: FacePhase;
  subState: ListenSubState;
  phaseStart: number;
  lastSecs: number;

  // Idle
  idle: {
    phase: "pause" | "beat";
    beatIndex: number;
    lastBeatIndex: number;
    timer: number;
    pauseDuration: number;
    // Smoothed continuous values (prevent snapping between beats)
    lookX: number;
    lookY: number;
    dilation: number;
  };

  // Audio smoothing
  audio: SmoothedAudio;

  // Bob amplitude (with momentum)
  bobAmpX: number;
  bobAmpY: number;

  // Hysteresis
  hysteresisAccum: number;
  hysteresisTarget: ListenSubState;
  hysteresisWindow: number;

  // Seek detection
  lastAudioTime: number;

  // Silence counter
  silenceTimer: number;

  // Swaying phase accumulator
  swayPhase: number;

  // Ecstatic rapid blink timer
  blinkTimer: number;
  blinkActive: boolean;

  // Pet interaction
  pet: {
    intensity: number; // 0–1 accumulated petting intensity
    blissTimer: number; // seconds in bliss payoff state
    decayDelay: number; // seconds until decay starts
  };
}

// =============================================================================
// PHASE HANDLERS
// =============================================================================

/** BOOT — Eyes-first awakening (0–2.5s). Linear interp, machine cold-starting. */
function tickBoot(elapsed: number): FaceState {
  const base: FaceState = {
    leftEye: "hidden",
    rightEye: "hidden",
    lookX: 0,
    lookY: 0,
    pupilDilation: 0,
    mouth: "hidden",
    mouthOpenness: 0,
    brightness: 0,
    phase: "BOOT",
  };

  // 0.0–0.3s: Black screen
  if (elapsed < 0.3) return base;

  // 0.3–0.6s: Right pupil appears — a single dot in the darkness
  if (elapsed < 0.6) {
    base.rightEye = "normal";
    base.pupilDilation = 0.3;
    base.brightness = 0.4;
    return base;
  }

  // 0.6–0.9s: Left pupil joins
  if (elapsed < 0.9) {
    base.leftEye = "normal";
    base.rightEye = "normal";
    base.pupilDilation = 0.3;
    base.brightness = 0.5;
    return base;
  }

  // 0.9–1.3s: Eye shapes form, brightness ramps 0.5→0.8
  if (elapsed < 1.3) {
    base.leftEye = "normal";
    base.rightEye = "normal";
    base.pupilDilation = 0.4;
    base.brightness = lerp(0.5, 0.8, (elapsed - 0.9) / 0.4);
    return base;
  }

  // 1.3–1.8s: Looks around cautiously. lookX sweeps 0→-0.6→+0.6→0
  if (elapsed < 1.8) {
    base.leftEye = "normal";
    base.rightEye = "normal";
    base.brightness = 0.8;
    base.pupilDilation = 0.5;
    const t = (elapsed - 1.3) / 0.5;
    if (t < 0.33) base.lookX = lerp(0, -0.6, t / 0.33);
    else if (t < 0.66) base.lookX = lerp(-0.6, 0.6, (t - 0.33) / 0.33);
    else base.lookX = lerp(0.6, 0, (t - 0.66) / 0.34);
    return base;
  }

  // 1.8–2.1s: Spots you — wonder. Eyes wide, pupils dilate.
  if (elapsed < 2.1) {
    base.leftEye = "wide";
    base.rightEye = "wide";
    base.lookX = 0;
    base.pupilDilation = 0.8;
    base.brightness = 0.9;
    return base;
  }

  // 2.1–2.5s: Eager to dance — head-bob + sparkle
  base.leftEye = "normal";
  base.rightEye = "normal";
  base.mouth = "smile";
  base.pupilDilation = 0.6;
  const t = (elapsed - 2.1) / 0.4;
  base.lookX = Math.sin(t * Math.PI * 4) * 0.3; // rapid oscillation ±0.3
  base.brightness = 0.9 + 0.1 * Math.sin(t * Math.PI * 6); // 2-3 flashes
  return base;
}

/** IDLE — Curious & restless. Random weighted beats, no repeat. */
function tickIdle(r: FaceRefs, dt: number): FaceState {
  const idle = r.idle;
  idle.timer += dt;

  // Determine targets for this frame
  let targetLookX = 0;
  let targetLookY = 0;
  let targetDilation = 0.5;
  let leftEye: EyeShape = "normal";
  let rightEye: EyeShape = "normal";
  let mouth: MouthShape = "neutral";

  if (idle.phase === "pause") {
    if (idle.timer >= idle.pauseDuration) {
      // Start new beat
      idle.beatIndex = pickWeightedBeat(idle.lastBeatIndex);
      idle.lastBeatIndex = idle.beatIndex;
      idle.phase = "beat";
      idle.timer = 0;
    }
    // During pause: targets are baseline (defaults above)
  } else {
    const beat = IDLE_BEATS[idle.beatIndex];
    const progress = Math.min(1, idle.timer / beat.duration);

    if (progress >= 1) {
      // Beat complete, start pause
      idle.phase = "pause";
      idle.timer = 0;
      idle.pauseDuration = 0.5 + Math.random() * 1.0;
    }

    // Apply beat-specific targets (discrete shapes snap, continuous smooth)
    switch (beat.name) {
      case "lookLeft":
        targetLookX = -0.5;
        break;
      case "lookRight":
        targetLookX = 0.5;
        break;
      case "lookDown":
        targetLookY = 0.3;
        mouth = "contemplative";
        break;
      case "center":
        // lookX targets 0 (default)
        break;
      case "blink":
        leftEye = progress < 0.5 ? "closed" : "normal";
        rightEye = progress < 0.5 ? "closed" : "normal";
        break;
      case "doubleBlink": {
        const p = Math.floor(progress * 4);
        const closed = p === 0 || p === 2;
        leftEye = closed ? "closed" : "normal";
        rightEye = closed ? "closed" : "normal";
        break;
      }
      case "wideEyes":
        leftEye = "wide";
        rightEye = "wide";
        targetDilation = 0.7;
        break;
      case "smile":
        mouth = "smile";
        break;
      // 'settle' and 'neutral' use baseline defaults
    }
  }

  // Smooth continuous values (prevents snapping between beats)
  idle.lookX += (targetLookX - idle.lookX) * IDLE_SMOOTH;
  idle.lookY += (targetLookY - idle.lookY) * IDLE_SMOOTH;
  idle.dilation += (targetDilation - idle.dilation) * IDLE_SMOOTH;

  return {
    leftEye,
    rightEye,
    lookX: idle.lookX,
    lookY: idle.lookY,
    pupilDilation: idle.dilation,
    mouth,
    mouthOpenness: 0,
    brightness: 1.0,
    phase: "IDLE",
  };
}

/** PERK_UP — Music discovery (0.5s one-shot). Wonder → eagerness. */
function tickPerkUp(elapsed: number): FaceState {
  // 0–0.3s: Eyes snap wide, mouth open — savoring
  if (elapsed < 0.3) {
    return {
      leftEye: "wide",
      rightEye: "wide",
      lookX: 0,
      lookY: 0,
      pupilDilation: 0.9,
      mouth: "open",
      mouthOpenness: 0.8,
      brightness: 1.0,
      phase: "PERK_UP",
    };
  }

  // 0.3–0.5s: Ease to normal + smile
  const t = Math.min(1, (elapsed - 0.3) / 0.2);
  const e = easeInOutCubic(t);

  return {
    leftEye: e > 0.5 ? "normal" : "wide",
    rightEye: e > 0.5 ? "normal" : "wide",
    lookX: 0,
    lookY: 0,
    pupilDilation: lerp(0.9, 0.6, e),
    mouth: e > 0.5 ? "smile" : "open",
    mouthOpenness: lerp(0.8, 0, e),
    brightness: 1.0,
    phase: "PERK_UP",
  };
}

/** LISTENING — sub-state dispatch. Audio already smoothed in tick(). */
function tickListening(r: FaceRefs, dt: number): FaceState {
  switch (r.subState) {
    case "BOBBING":
      return tickBobbing(r);
    case "TENSION":
      return tickTension(r);
    case "SWAYING":
      return tickSwaying(r, dt);
    case "ECSTATIC":
      return tickEcstatic(r, dt);
  }
}

/** BOBBING — BPM-driven figure-8 head bob. The primary dance motion. */
function tickBobbing(r: FaceRefs): FaceState {
  const bpm = useSessionStore.getState().trackInfo?.bpm ?? 120;
  const audioTime = useAudioActivityStore.getState().audioTime;
  const phase = audioTime * (bpm / 60) * Math.PI * 2;

  // Target amplitude scales with energy
  const targetAmpX = 0.15 + r.audio.overallEnergy * 0.25; // 0.15→0.4
  const targetAmpY = 0.08 + r.audio.overallEnergy * 0.12; // 0.08→0.2

  // Ramp with momentum — buddy has inertia
  const aX = targetAmpX > r.bobAmpX ? RAMP_UP : RAMP_DOWN;
  r.bobAmpX = r.bobAmpX * (1 - aX) + targetAmpX * aX;
  const aY = targetAmpY > r.bobAmpY ? RAMP_UP : RAMP_DOWN;
  r.bobAmpY = r.bobAmpY * (1 - aY) + targetAmpY * aY;

  return {
    leftEye: "normal",
    rightEye: "normal",
    lookX: Math.sin(phase) * r.bobAmpX,
    lookY: Math.cos(phase * 0.5) * r.bobAmpY,
    pupilDilation: 0.4 + r.audio.overallEnergy * 0.3,
    mouth: "smile",
    mouthOpenness: 0,
    brightness: 1.0,
    phase: "LISTENING",
    subState: "BOBBING",
  };
}

/** TENSION — Focused stillness. Wide eyes locked in, feeling sustained intensity. */
function tickTension(r: FaceRefs): FaceState {
  return {
    leftEye: "wide",
    rightEye: "wide",
    lookX: 0,
    lookY: 0,
    pupilDilation: 0.6 + r.audio.overallEnergy * 0.3,
    mouth: r.audio.overallEnergy > 0.6 ? "smile" : "neutral",
    mouthOpenness: 0,
    brightness: 1.0 + (Math.random() - 0.5) * 0.04, // subtle CRT flicker ±0.02
    phase: "LISTENING",
    subState: "TENSION",
  };
}

/** SWAYING — Quiet listening. Half-closed eyes, slow drift, breathing dilation. */
function tickSwaying(r: FaceRefs, dt: number): FaceState {
  r.swayPhase += dt;

  // Slow sinusoidal eye drift (pseudo-perlin)
  const lookX = Math.sin(r.swayPhase * 0.7) * 0.15;
  const lookY = Math.sin(r.swayPhase * 0.5 + 1.0) * 0.08;

  // Breathing dilation cycle (2.5s period)
  const dilation = 0.4 + 0.1 * Math.sin((r.swayPhase * Math.PI * 2) / 2.5);

  return {
    leftEye: "half",
    rightEye: "half",
    lookX,
    lookY,
    pupilDilation: dilation,
    mouth: "smile",
    mouthOpenness: 0,
    brightness: 1.0 + (Math.random() - 0.5) * 0.04,
    phase: "LISTENING",
    subState: "SWAYING",
  };
}

/** ECSTATIC — Go wild. The drop. Everything maxed. */
function tickEcstatic(r: FaceRefs, dt: number): FaceState {
  const bpm = useSessionStore.getState().trackInfo?.bpm ?? 120;
  const audioTime = useAudioActivityStore.getState().audioTime;
  const phase = audioTime * (bpm / 60) * Math.PI * 2;

  // Max amplitude figure-8
  const lookX = Math.sin(phase) * 0.5;
  const lookY = Math.cos(phase * 0.5) * 0.25;

  // Occasional rapid blinks
  r.blinkTimer -= dt;
  if (r.blinkTimer <= 0) {
    r.blinkActive = !r.blinkActive;
    r.blinkTimer = r.blinkActive ? 0.05 : 0.5 + Math.random() * 0.5;
  }
  const eye: EyeShape = r.blinkActive ? "closed" : "wide";

  return {
    leftEye: eye,
    rightEye: eye,
    lookX,
    lookY,
    pupilDilation: 0.9 + Math.random() * 0.1,
    mouth: "excited",
    mouthOpenness: 0.8 + r.audio.overallEnergy * 0.2,
    brightness: 1.0 + r.audio.overallEnergy * 0.05,
    phase: "LISTENING",
    subState: "ECSTATIC",
  };
}

/** DISCONNECT — Glitch + confusion (1s), then → IDLE. */
function tickDisconnect(elapsed: number): FaceState {
  // 0–0.1s: Glitch — flag for renderer to scatter random chars
  if (elapsed < 0.1) {
    return {
      leftEye: "normal",
      rightEye: "normal",
      lookX: 0,
      lookY: 0,
      pupilDilation: 0.5,
      mouth: "neutral",
      mouthOpenness: 0,
      brightness: 0.8,
      phase: "DISCONNECT",
      glitch: true,
    };
  }

  // 0.1–0.5s: Confused — wide eyes, rapid lookX sweep (searching)
  if (elapsed < 0.5) {
    const t = (elapsed - 0.1) / 0.4;
    return {
      leftEye: "wide",
      rightEye: "wide",
      lookX: Math.sin(t * Math.PI * 8) * 0.4,
      lookY: 0,
      pupilDilation: 0.7,
      mouth: "open",
      mouthOpenness: 0.5,
      brightness: 0.9,
      phase: "DISCONNECT",
    };
  }

  // 0.5–1.0s: Settle back to normal
  const t = Math.min(1, (elapsed - 0.5) / 0.5);
  const e = easeInOutCubic(t);
  return {
    leftEye: e > 0.5 ? "normal" : "wide",
    rightEye: e > 0.5 ? "normal" : "wide",
    lookX: lerp(0.2, 0, e),
    lookY: 0,
    pupilDilation: lerp(0.7, 0.5, e),
    mouth: "neutral",
    mouthOpenness: 0,
    brightness: lerp(0.9, 1.0, e),
    phase: "DISCONNECT",
  };
}

/** Agent override — local control feedback for recording, planning, and apply. */
function tickAgent(): FaceState | null {
  const { armed, latestEvent } = useAgentStore.getState();
  const phase = latestEvent?.phase;
  if (!phase || (!armed && phase !== "error")) return null;
  const eventAgeMs = latestEvent?.timestamp ? Date.now() - Date.parse(latestEvent.timestamp) : 0;
  if (phase === "error" && eventAgeMs > 3_500) return null;

  const t = performance.now() / 1000;
  const pulse = Math.sin(t * Math.PI * 3) * 0.5 + 0.5;

  switch (phase) {
    case "listening":
      return {
        leftEye: "wide",
        rightEye: "wide",
        lookX: Math.sin(t * 4) * 0.18,
        lookY: -0.08,
        pupilDilation: 0.75 + pulse * 0.18,
        mouth: "open",
        mouthOpenness: 0.35 + pulse * 0.25,
        brightness: 1.0 + pulse * 0.08,
        phase: "IDLE",
      };

    case "transcribing":
      return {
        leftEye: "half",
        rightEye: "half",
        lookX: 0,
        lookY: 0.22,
        pupilDilation: 0.55 + pulse * 0.1,
        mouth: "contemplative",
        mouthOpenness: 0,
        brightness: 0.95 + pulse * 0.05,
        phase: "IDLE",
      };

    case "thinking":
    case "searching_features":
    case "planning":
    case "dj_deciding":
      return {
        leftEye: "normal",
        rightEye: pulse > 0.85 ? "wide" : "normal",
        lookX: Math.sin(t * 2.2) * 0.38,
        lookY: Math.cos(t * 1.6) * 0.12,
        pupilDilation: 0.55 + pulse * 0.2,
        mouth: "contemplative",
        mouthOpenness: 0,
        brightness: 0.98 + pulse * 0.06,
        phase: "IDLE",
      };

    case "applying":
      return {
        leftEye: "wide",
        rightEye: "wide",
        lookX: Math.sin(t * 9) * 0.28,
        lookY: Math.cos(t * 7) * 0.1,
        pupilDilation: 0.8 + pulse * 0.15,
        mouth: "smile",
        mouthOpenness: 0,
        brightness: 1.05 + pulse * 0.08,
        phase: "IDLE",
      };

    case "error":
      return {
        leftEye: "wide",
        rightEye: "wide",
        lookX: Math.sin(t * 18) * 0.35,
        lookY: 0,
        pupilDilation: 0.7,
        mouth: "open",
        mouthOpenness: 0.5,
        brightness: 0.9,
        phase: "DISCONNECT",
        glitch: pulse > 0.6,
      };

    default:
      return null;
  }
}

// =============================================================================
// PET INTERACTION — Progressive delight from rubbing between the eyes
// =============================================================================

/**
 * tickPet — Override face state when petting is active.
 *
 * The vibe is ^.^ — happy squish, not overwhelm.
 *
 * Intensity progression:
 *   0.0–0.2: Notices the touch — eyes soften, small smile
 *   0.2–0.5: Enjoying it — eyes half-close (^.^), bigger smile, leans in
 *   0.5–0.85: Full ^.^ — eyes closed happy, big smile, sparkles start
 *   0.85+: BLISS — eyes squish tight, max smile, sparkles everywhere, gentle sway
 */
function tickPet(
  r: FaceRefs,
  pet: PetInput,
  dt: number,
): FaceState | null {
  const p = r.pet;
  const now = performance.now();
  const timeSinceMove = (now - pet.lastMoveTime) / 1000;

  // Actively petting = in zone + mouse moving recently
  const isPetting =
    pet.inZone && pet.moveSpeed > PET_SPEED_THRESHOLD && timeSinceMove < 0.15;

  if (isPetting) {
    p.intensity = Math.min(1, p.intensity + PET_RAMP_UP);
    p.decayDelay = PET_DECAY_DELAY;
  } else {
    p.decayDelay = Math.max(0, p.decayDelay - dt);
    if (p.decayDelay <= 0) {
      p.intensity = Math.max(0, p.intensity - PET_RAMP_DOWN);
    }
  }

  // Not active enough to override normal face
  if (p.intensity < 0.01) {
    p.blissTimer = 0;
    return null;
  }

  const intensity = p.intensity;

  // Lean gently into the touch
  const leanX = (pet.normalizedX - 0.5) * intensity * 0.5;
  const leanY = (pet.normalizedY - 0.5) * intensity * 0.3;

  // Bliss state — sustained ^.^ with sparkles and gentle happy sway
  if (intensity >= PET_BLISS_THRESHOLD) {
    p.blissTimer += dt;
    const blissProgress = Math.min(1, p.blissTimer / PET_BLISS_DURATION);
    const sway = Math.sin(now / 400) * 0.15;
    return {
      leftEye: "happy",
      rightEye: "happy",
      lookX: sway,
      lookY: 0,
      pupilDilation: 0.8,
      mouth: "smile",
      mouthOpenness: 0,
      brightness: 1.0 + Math.sin(now / 300) * 0.04,
      phase: r.phase,
      sparkle: (0.8 + Math.sin(now / 200) * 0.2) * blissProgress,
    };
  }

  // Progressive ^.^ response
  let eyeShape: EyeShape;
  if (intensity < 0.2) eyeShape = "normal";
  else if (intensity < 0.5) eyeShape = "half";
  else eyeShape = "happy";

  // Sparkles fade in above 0.4 intensity
  const sparkle = intensity > 0.4 ? (intensity - 0.4) / 0.6 : 0;

  return {
    leftEye: eyeShape,
    rightEye: eyeShape,
    lookX: leanX,
    lookY: leanY,
    pupilDilation: 0.5 + intensity * 0.3,
    mouth: "smile",
    mouthOpenness: 0,
    brightness: 1.0 + (intensity > 0.3 ? Math.sin(now / 300) * 0.02 : 0),
    phase: r.phase,
    sparkle,
  };
}

// =============================================================================
// AUDIO + SUB-STATE MANAGEMENT
// =============================================================================

/** Smooth raw audio features into the global composite signal. */
function updateSmoothedAudio(r: FaceRefs) {
  const { stems } = useAudioActivityStore.getState();
  const energies = [
    stems.drums?.energy_smooth ?? 0,
    stems.bass?.energy_smooth ?? 0,
    stems.vocals?.energy_smooth ?? 0,
    stems.other?.energy_smooth ?? 0,
  ];

  const overallEnergy = Math.max(...energies);
  const avgEnergy = energies.reduce((a, b) => a + b, 0) / energies.length;
  const hasTransient =
    (stems.drums?.transient ?? 0) > 0.5 || (stems.drums?.flash ?? 0) > 0.5;

  r.audio.overallEnergy = smooth(r.audio.overallEnergy, overallEnergy);
  r.audio.avgEnergy = smooth(r.audio.avgEnergy, avgEnergy);
  r.audio.transientRate = smooth(
    r.audio.transientRate,
    hasTransient ? 1.0 : 0.0,
  );
}

/** Apply hysteresis-gated sub-state transition. */
function updateSubState(r: FaceRefs, dt: number) {
  const detected = detectSubState(r.audio);
  if (detected !== r.subState) {
    // New target — reset accumulator if target changed
    if (detected !== r.hysteresisTarget) {
      r.hysteresisTarget = detected;
      r.hysteresisAccum = 0;
    }
    r.hysteresisAccum += dt;
    if (r.hysteresisAccum >= r.hysteresisWindow) {
      r.subState = detected;
      r.hysteresisAccum = 0;
    }
  } else {
    r.hysteresisAccum = 0;
  }
}

/** Transition to a new phase, resetting phase-specific state. */
function setPhase(r: FaceRefs, phase: FacePhase, secs: number) {
  r.phase = phase;
  r.phaseStart = secs;

  if (phase === "IDLE") {
    r.idle.phase = "pause";
    r.idle.timer = 0;
    r.idle.pauseDuration = 0.5 + Math.random();
    r.idle.lookX = 0;
    r.idle.lookY = 0;
    r.idle.dilation = 0.5;
  }

  if (phase === "LISTENING") {
    r.silenceTimer = 0;
    r.hysteresisAccum = 0;
    r.blinkTimer = 1.0;
    r.blinkActive = false;
    r.swayPhase = 0;
  }
}

// =============================================================================
// HOOK
// =============================================================================

export function useFaceAnimation(): {
  tick: (secs: number) => FaceState;
  petRef: MutableRefObject<PetInput>;
} {
  const refs = useRef<FaceRefs>({
    phase: "BOOT",
    subState: "BOBBING",
    phaseStart: 0,
    lastSecs: 0,

    idle: {
      phase: "pause",
      beatIndex: -1,
      lastBeatIndex: -1,
      timer: 0,
      pauseDuration: 1.0,
      lookX: 0,
      lookY: 0,
      dilation: 0.5,
    },

    audio: { overallEnergy: 0, avgEnergy: 0, transientRate: 0 },
    bobAmpX: 0.15,
    bobAmpY: 0.08,

    hysteresisAccum: 0,
    hysteresisTarget: "BOBBING",
    hysteresisWindow: NORMAL_HYSTERESIS,

    lastAudioTime: 0,
    silenceTimer: 0,
    swayPhase: 0,
    blinkTimer: 1.0,
    blinkActive: false,

    pet: { intensity: 0, blissTimer: 0, decayDelay: 0 },
  });

  const petRef = useRef<PetInput>({
    normalizedX: 0.5,
    normalizedY: 0.5,
    inZone: false,
    moveSpeed: 0,
    lastMoveTime: 0,
    intensity: 0,
  });

  const tick = useCallback((secs: number): FaceState => {
    const r = refs.current;
    const dt = Math.min(secs - r.lastSecs, 0.1); // clamp for safety
    r.lastSecs = secs;
    const elapsed = secs - r.phaseStart;

    const agentState = tickAgent();
    if (agentState) return agentState;

    // Pet interaction override — works in any phase
    const petState = tickPet(r, petRef.current, dt);
    petRef.current.intensity = r.pet.intensity;
    petBridge.intensity = r.pet.intensity; // shared with arm hook
    if (petState) return petState;

    switch (r.phase) {
      // ----- BOOT → IDLE after 2.5s -----
      case "BOOT":
        if (elapsed >= BOOT_DURATION) {
          setPhase(r, "IDLE", secs);
          return tickIdle(r, dt);
        }
        return tickBoot(elapsed);

      // ----- IDLE → PERK_UP on music detection -----
      case "IDLE":
        if (checkMusicDetection()) {
          setPhase(r, "PERK_UP", secs);
          return tickPerkUp(0);
        }
        return tickIdle(r, dt);

      // ----- PERK_UP → LISTENING after 0.5s -----
      case "PERK_UP":
        if (elapsed >= PERK_UP_DURATION) {
          setPhase(r, "LISTENING", secs);
          return tickListening(r, dt);
        }
        return tickPerkUp(elapsed);

      // ----- LISTENING — main steady-state -----
      case "LISTENING": {
        const store = useAudioActivityStore.getState();

        // Disconnect → DISCONNECT phase
        if (!store.isReceiving) {
          setPhase(r, "DISCONNECT", secs);
          return tickDisconnect(0);
        }

        // Update smoothed audio
        updateSmoothedAudio(r);

        // Silence detection → IDLE
        if (r.audio.overallEnergy < 0.02) {
          r.silenceTimer += dt;
          if (r.silenceTimer >= SILENCE_TIMEOUT) {
            setPhase(r, "IDLE", secs);
            return tickIdle(r, dt);
          }
        } else {
          r.silenceTimer = 0;
        }

        // Seek detection — reduce hysteresis window
        const audioTime = store.audioTime;
        if (Math.abs(audioTime - r.lastAudioTime) > SEEK_THRESHOLD) {
          r.hysteresisWindow = SEEK_HYSTERESIS;
        } else {
          r.hysteresisWindow = NORMAL_HYSTERESIS;
        }
        r.lastAudioTime = audioTime;

        // Sub-state transition with hysteresis
        updateSubState(r, dt);

        return tickListening(r, dt);
      }

      // ----- DISCONNECT → IDLE after 1s -----
      case "DISCONNECT":
        if (elapsed >= DISCONNECT_DURATION) {
          setPhase(r, "IDLE", secs);
          return tickIdle(r, dt);
        }
        return tickDisconnect(elapsed);
    }
  }, []);

  return { tick, petRef };
}
