/**
 * Juice utilities for animations and visual feedback.
 *
 * Based on juice_guidelines.md - "Creative" intensity level:
 * - Expressive animations (200-400ms, springs)
 * - Visual feedback on interactions
 * - No sounds (user preference)
 */

import type { Transition } from "motion/react";

// ============================================================================
// Spring Presets
// ============================================================================

/**
 * Spring configurations for different interaction types.
 * Using Motion's spring format (stiffness + damping).
 */
export const springs = {
  /** Snappy for buttons, quick responses */
  snappy: {
    type: "spring" as const,
    stiffness: 400,
    damping: 30,
  },

  /** Bouncy for entries, celebrations */
  bouncy: {
    type: "spring" as const,
    stiffness: 300,
    damping: 15,
  },

  /** Gentle for popups, modals */
  gentle: {
    type: "spring" as const,
    stiffness: 200,
    damping: 25,
  },

  /** Wobbly for playful elements, mascot */
  wobbly: {
    type: "spring" as const,
    stiffness: 180,
    damping: 12,
  },

  /** Stiff for precise movements */
  stiff: {
    type: "spring" as const,
    stiffness: 350,
    damping: 28,
  },

  /** Slow for dramatic reveals */
  slow: {
    type: "spring" as const,
    stiffness: 120,
    damping: 20,
  },
} satisfies Record<string, Transition>;

// ============================================================================
// Animation Variants
// ============================================================================

/**
 * Common animation variants for Motion components.
 */
export const variants = {
  /** Fade in/out */
  fade: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
    exit: { opacity: 0 },
  },

  /** Scale up from center */
  scaleUp: {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.9 },
  },

  /** Slide up from bottom */
  slideUp: {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 20 },
  },

  /** Pop in (bouncy) */
  popIn: {
    hidden: { opacity: 0, scale: 0.8 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: springs.bouncy,
    },
    exit: { opacity: 0, scale: 0.8 },
  },

  /** Stem circle pulse based on activity */
  stemPulse: {
    idle: { scale: 1 },
    active: (level: number) => ({
      scale: 1 + level * 0.08,
      transition: {
        type: "spring",
        stiffness: 300 - level * 100,
        damping: 20,
      },
    }),
  },
};

// ============================================================================
// Hover/Tap Animations
// ============================================================================

/**
 * Standard hover animation for interactive elements.
 */
export const hoverScale = {
  scale: 1.02,
  transition: springs.snappy,
};

/**
 * Standard tap animation for buttons.
 */
export const tapScale = {
  scale: 0.98,
};

/**
 * Win95 button press effect (shift down).
 */
export const win95Press = {
  y: 1,
  transition: { duration: 0 },
};

// ============================================================================
// Stagger Delays
// ============================================================================

/**
 * Stagger configuration for list animations.
 */
export const stagger = {
  /** Fast stagger for small lists */
  fast: {
    staggerChildren: 0.03,
    delayChildren: 0.05,
  },

  /** Normal stagger */
  normal: {
    staggerChildren: 0.06,
    delayChildren: 0.1,
  },

  /** Slow stagger for dramatic reveals */
  slow: {
    staggerChildren: 0.12,
    delayChildren: 0.15,
  },
};

// ============================================================================
// Haptic Patterns (for future use)
// ============================================================================

/**
 * Haptic feedback patterns using Vibration API.
 * These are for future enhancement if haptics are enabled.
 */
export const haptics = {
  tap: () => navigator.vibrate?.(10),
  success: () => navigator.vibrate?.([10, 50, 20]),
  error: () => navigator.vibrate?.([50, 30, 50, 30, 50]),
  warning: () => navigator.vibrate?.([30, 50, 30]),
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate spring duration based on stiffness and damping.
 * Useful for coordinating CSS with spring animations.
 */
export function springDuration(stiffness: number, damping: number): number {
  // Approximate duration based on spring parameters
  const omega = Math.sqrt(stiffness);
  const zeta = damping / (2 * omega);

  if (zeta >= 1) {
    // Overdamped
    return 1000 / omega;
  } else {
    // Underdamped
    return (4 * 1000) / (omega * zeta);
  }
}

/**
 * Interpolate a value based on activity level (0-1).
 */
export function lerpActivity(
  min: number,
  max: number,
  activity: number
): number {
  return min + (max - min) * Math.min(1, Math.max(0, activity));
}

/**
 * Get stem color from CSS custom property.
 */
export function getStemColor(stem: string): string {
  const colors: Record<string, string> = {
    bass: "var(--color-stem-bass)",
    drums: "var(--color-stem-drums)",
    vocals: "var(--color-stem-vocals)",
    other: "var(--color-stem-other)",
  };
  return colors[stem] || "var(--color-accent)";
}
