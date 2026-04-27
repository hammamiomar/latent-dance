/**
 * useArmAnimation — Articulated 3-joint arm dance.
 *
 * Drives shoulder, elbow, and wrist rotations for both arms via rAF.
 * Reads audio activity imperatively (no re-renders).
 *
 * Audio mapping per joint:
 *   Shoulder — drums energy + bass sway   (slow rock, big sweeps)
 *   Elbow   — drums transient             (sharp snap on beats, springs back)
 *   Wrist   — vocals + other_high energy  (fast flick, melodic response)
 *
 * Two-stage smoothing:
 *   1. Input EMA — smooths raw audio values (fast attack, slow release)
 *   2. Output lerp — smooths final rotation values for buttery DOM updates
 */

import { useRef, useEffect, useCallback } from "react";
import { useAudioActivityStore } from "../stores/useAudioActivityStore";
import { petBridge } from "../shared/petBridge";

// =============================================================================
// Constants
// =============================================================================

// Input smoothing (asymmetric EMA — fast attack, slow release)
const ATTACK = 0.25;
const RELEASE = 0.04;

// Output smoothing — lerp factor per frame (higher = snappier, lower = smoother)
const OUTPUT_SMOOTH = 0.12;

// Idle drift — different frequency per joint for organic feel
const IDLE = {
  shoulder: { speed: 0.6, rotation: 1.2, bob: 0.6 },
  elbow: { speed: 1.1, rotation: 1.5 },
  wrist: { speed: 1.8, rotation: 2.0 },
};

// Audio-driven ranges (degrees)
const SHOULDER = {
  maxRotation: 6, // drum wobble
  maxBob: 3, // transient snap (px)
  swayAmount: 3, // bass sway
};
const ELBOW_MAX = 8; // snap-flex on drum transients
const WRIST_MAX = 7; // vocal/high-freq flick

// Pet reaction — arms curl inward and cheer
const PET = {
  shoulderLift: -2, // px — arms lift up
  shoulderCurl: 5, // degrees — shoulders rotate inward
  elbowCurl: 10, // degrees — elbows bend (closing fingers)
  wristCurl: 12, // degrees — wrists flex inward
  cheerSpeed: 3, // Hz — happy sway during bliss
  cheerAmount: 4, // degrees — cheer sway amplitude
};

// =============================================================================
// Hook
// =============================================================================

export function useArmAnimation() {
  const leftShoulder = useRef<HTMLDivElement>(null);
  const leftElbow = useRef<HTMLDivElement>(null);
  const leftWrist = useRef<HTMLDivElement>(null);
  const rightShoulder = useRef<HTMLDivElement>(null);
  const rightElbow = useRef<HTMLDivElement>(null);
  const rightWrist = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  // Smoothed audio values + smoothed output rotations (mutable, no re-renders)
  const state = useRef({
    // Input audio (EMA smoothed)
    drumEnergy: 0,
    drumTransient: 0,
    bassEnergy: 0,
    vocalEnergy: 0,
    highEnergy: 0,
    // Output rotations (lerp smoothed for buttery DOM)
    lShoulderRot: 0,
    rShoulderRot: 0,
    shoulderBob: 0,
    lElbowRot: 0,
    rElbowRot: 0,
    lWristRot: 0,
    rWristRot: 0,
    prevTime: 0,
  });

  const animate = useCallback((time: number) => {
    const s = state.current;
    const dt = s.prevTime ? (time - s.prevTime) / 1000 : 0.016;
    s.prevTime = time;

    // Read audio store imperatively
    const { stems, isReceiving } = useAudioActivityStore.getState();

    // Stage 1: Asymmetric EMA on raw audio
    const ema = (current: number, target: number) => {
      const alpha = target > current ? ATTACK : RELEASE;
      return current + (target - current) * Math.min(1, alpha * dt * 60);
    };

    s.drumEnergy = ema(s.drumEnergy, isReceiving ? stems.drums.energy_smooth : 0);
    s.drumTransient = ema(s.drumTransient, isReceiving ? stems.drums.transient : 0);
    s.bassEnergy = ema(s.bassEnergy, isReceiving ? stems.bass.energy_smooth : 0);
    s.vocalEnergy = ema(s.vocalEnergy, isReceiving ? stems.vocals.energy_smooth : 0);
    s.highEnergy = ema(s.highEnergy, isReceiving ? stems.other_high.energy_smooth : 0);

    const t = time / 1000;

    // Audio intensity — idle fades but never disappears
    const audioLevel = Math.min(1, s.drumEnergy + s.bassEnergy + s.vocalEnergy * 0.5);
    const idleMix = 1 - audioLevel * 0.7;

    // ---- Pet interaction (read from shared bridge) ----
    const pet = petBridge.intensity;

    // ---- Target rotations (raw, before output smoothing) ----

    // Shoulder
    const shoulderIdle = Math.sin(t * IDLE.shoulder.speed) * IDLE.shoulder.rotation * idleMix;
    const shoulderDrum = Math.sin(t * 5) * s.drumEnergy * SHOULDER.maxRotation;
    const shoulderSway = Math.sin(t * 1.8) * s.bassEnergy * SHOULDER.swayAmount;
    // Pet: arms lift up + cheer sway during bliss
    const petCheer = pet > 0.85 ? Math.sin(t * PET.cheerSpeed) * PET.cheerAmount : 0;
    const petLift = pet * PET.shoulderLift;
    const petShoulderCurl = pet * PET.shoulderCurl;
    const targetBob = Math.sin(t * IDLE.shoulder.speed * 1.3 + 0.5) * IDLE.shoulder.bob * idleMix
      + s.drumTransient * SHOULDER.maxBob + petLift;

    const targetLShoulder = shoulderIdle + shoulderDrum + shoulderSway + petShoulderCurl + petCheer;
    const targetRShoulder = -(shoulderIdle + shoulderDrum) + shoulderSway - petShoulderCurl + petCheer;

    // Elbow — pet curls inward (positive = flex)
    const elbowIdle = Math.sin(t * IDLE.elbow.speed + 1.0) * IDLE.elbow.rotation * idleMix;
    const elbowSnap = s.drumTransient * ELBOW_MAX;
    const petElbowCurl = pet * PET.elbowCurl;

    const targetLElbow = elbowIdle + elbowSnap + petElbowCurl;
    const targetRElbow = -(elbowIdle + elbowSnap) - petElbowCurl;

    // Wrist — pet curls fingers closed
    const wristIdle = Math.sin(t * IDLE.wrist.speed + 2.0) * IDLE.wrist.rotation * idleMix;
    const wristFlick = (s.vocalEnergy + s.highEnergy) * 0.5 * WRIST_MAX * Math.sin(t * 6);
    const petWristCurl = pet * PET.wristCurl;

    const targetLWrist = wristIdle + wristFlick + petWristCurl;
    const targetRWrist = -(wristIdle + wristFlick) - petWristCurl;

    // ---- Stage 2: Output lerp for buttery motion ----
    const smooth = Math.min(1, OUTPUT_SMOOTH * dt * 60);
    s.lShoulderRot += (targetLShoulder - s.lShoulderRot) * smooth;
    s.rShoulderRot += (targetRShoulder - s.rShoulderRot) * smooth;
    s.shoulderBob += (targetBob - s.shoulderBob) * smooth;
    s.lElbowRot += (targetLElbow - s.lElbowRot) * smooth;
    s.rElbowRot += (targetRElbow - s.rElbowRot) * smooth;
    s.lWristRot += (targetLWrist - s.lWristRot) * smooth;
    s.rWristRot += (targetRWrist - s.rWristRot) * smooth;

    // ---- Apply to DOM ----
    if (leftShoulder.current) {
      leftShoulder.current.style.transform =
        `rotate(${s.lShoulderRot.toFixed(2)}deg) translateY(${s.shoulderBob.toFixed(1)}px)`;
    }
    if (rightShoulder.current) {
      rightShoulder.current.style.transform =
        `rotate(${s.rShoulderRot.toFixed(2)}deg) translateY(${s.shoulderBob.toFixed(1)}px)`;
    }
    if (leftElbow.current) {
      leftElbow.current.style.transform = `rotate(${s.lElbowRot.toFixed(2)}deg)`;
    }
    if (rightElbow.current) {
      rightElbow.current.style.transform = `rotate(${s.rElbowRot.toFixed(2)}deg)`;
    }
    if (leftWrist.current) {
      leftWrist.current.style.transform = `rotate(${s.lWristRot.toFixed(2)}deg)`;
    }
    if (rightWrist.current) {
      rightWrist.current.style.transform = `rotate(${s.rWristRot.toFixed(2)}deg)`;
    }

    rafRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [animate]);

  return {
    leftShoulder,
    leftElbow,
    leftWrist,
    rightShoulder,
    rightElbow,
    rightWrist,
  };
}
