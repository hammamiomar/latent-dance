/**
 * petBridge — Shared mutable state for pet interaction.
 *
 * Module-scoped singleton. Written by useFaceAnimation's tick(),
 * read by useArmAnimation's rAF loop. No React, no store, no subscriptions.
 * Just a plain object — the lightest possible bridge between sibling components.
 */

export const petBridge = {
  /** Pet intensity (0–1), updated every face tick */
  intensity: 0,
};
