/**
 * Anchor layout contract: n=4 must reproduce the historical SAE corner
 * positions EXACTLY (so the four-slot backend renders identically), the
 * midpoints extend them in priority order, and the ellipse fallback keeps
 * arbitrary slot counts inside the belly.
 */

import { describe, expect, it } from "vitest";
import { slotAnchorPositions } from "./slotLayout";

describe("slotAnchorPositions", () => {
  it("n=4 reproduces the historical SAE corners exactly", () => {
    expect(slotAnchorPositions(4, 800, 600)).toEqual([
      { x: 120, y: 120 },       // TL — slot order[0]
      { x: 680, y: 120 },       // TR
      { x: 120, y: 420 },       // BL (bottom lifted 60px above the dock)
      { x: 680, y: 420 },       // BR
    ]);
  });

  it("n=6 adds the left/right edge midpoints after the corners", () => {
    const anchors = slotAnchorPositions(6, 800, 600);
    expect(anchors.slice(0, 4)).toEqual(slotAnchorPositions(4, 800, 600));
    expect(anchors[4]).toEqual({ x: 120, y: 300 });
    expect(anchors[5]).toEqual({ x: 680, y: 300 });
  });

  it("past 8 slots, falls back to an ellipse that stays inside the belly", () => {
    const n = 10;
    const anchors = slotAnchorPositions(n, 800, 600);
    expect(anchors).toHaveLength(n);
    // Starts at 12 o'clock
    expect(anchors[0].x).toBeCloseTo(400);
    expect(anchors[0].y).toBeCloseTo(120);
    for (const { x, y } of anchors) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(800);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(600);
    }
  });
});
