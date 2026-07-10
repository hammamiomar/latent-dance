/**
 * Slot anchor layout — where N steering orbs rest in the belly.
 *
 * Anchor priority: four corners, then left/right edge midpoints, then
 * top/bottom edge midpoints. n=4 reproduces the historical SAE corners
 * exactly (TL, TR, BL, BR — matching manifest slot order), n=6 adds the
 * side midpoints. Beyond 8 slots the layout falls back to an ellipse
 * inset by the same padding, starting at 12 o'clock and going clockwise.
 *
 * Orbs are free physics bodies — these are only their birth positions.
 */

export interface AnchorPosition {
  x: number;
  y: number;
}

const PADDING = 120;
/** The bottom row sits above the player dock — historical belly constant. */
const BOTTOM_LIFT = 60;

export function slotAnchorPositions(
  n: number,
  width: number,
  height: number,
): AnchorPosition[] {
  const bottom = height - PADDING - BOTTOM_LIFT;
  const anchors: AnchorPosition[] = [
    { x: PADDING, y: PADDING }, // top-left
    { x: width - PADDING, y: PADDING }, // top-right
    { x: PADDING, y: bottom }, // bottom-left
    { x: width - PADDING, y: bottom }, // bottom-right
    { x: PADDING, y: height / 2 }, // mid-left
    { x: width - PADDING, y: height / 2 }, // mid-right
    { x: width / 2, y: PADDING }, // top-mid
    { x: width / 2, y: bottom }, // bottom-mid
  ];
  if (n <= anchors.length) return anchors.slice(0, n);

  const rx = Math.max(40, width / 2 - PADDING);
  const ry = Math.max(40, height / 2 - PADDING);
  return Array.from({ length: n }, (_, i) => {
    const theta = -Math.PI / 2 + (2 * Math.PI * i) / n;
    return {
      x: width / 2 + rx * Math.cos(theta),
      y: height / 2 + ry * Math.sin(theta),
    };
  });
}
