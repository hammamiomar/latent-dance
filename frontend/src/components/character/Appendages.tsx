/**
 * Appendages — Articulated pixel-art arms gripping the belly screen.
 *
 * Each arm is split into 3 segments (upper arm, forearm, hand) using
 * clip-path on the same SVG image. Nested CSS transforms create a
 * forward kinematics chain: rotating the upper arm carries the forearm
 * and hand along automatically.
 *
 * Joint positions (from SVG path analysis):
 *   Elbow: ~57% across, ~60% down the viewBox
 *   Wrist: ~71% across, ~66% down the viewBox
 */

import { useArmAnimation } from "../../hooks/useArmAnimation";

// Arm segment image (shared across all 3 clips)
const ARM_SRC = "/arm.svg";
const ARM_IMG_STYLE = { imageRendering: "pixelated" as const };

// Clip regions (% of element bounds, with overlap at joints).
// Same for both sides — the scaleX(-1) on the <img> flips the pixels,
// and clip-path applies in the element's local (post-transform) space.
const CLIP_UPPER = "polygon(0% 0%, 62% 0%, 62% 65%, 56% 65%, 0% 8%)";
const CLIP_FOREARM = "polygon(53% 0%, 78% 0%, 78% 75%, 53% 75%)";
const CLIP_HAND = "polygon(55% 58%, 100% 58%, 100% 100%, 55% 100%)";

// Joint pivot points as % of the arm container.
// Right arm: elbow/wrist flip horizontally (100% - x) because the
// div is NOT mirrored — only the <img> inside is.
const JOINTS = {
  left: { elbow: "57% 60%", wrist: "71% 66%" },
  right: { elbow: "43% 60%", wrist: "29% 66%" },
};

/** One articulated arm with 3 segments in a nested FK chain. */
function ArticulatedArm({
  shoulderRef,
  elbowRef,
  wristRef,
  mirror,
  side,
}: {
  shoulderRef: React.RefObject<HTMLDivElement | null>;
  elbowRef: React.RefObject<HTMLDivElement | null>;
  wristRef: React.RefObject<HTMLDivElement | null>;
  mirror: boolean;
  side: "left" | "right";
}) {
  const imgMirror = mirror ? { ...ARM_IMG_STYLE, transform: "scaleX(-1)" } : ARM_IMG_STYLE;
  const joints = JOINTS[side];

  return (
    <div
      className="absolute pointer-events-none z-[10300]"
      style={{
        width: "20%",
        top: "-2%",
        [side]: "-2.5%",
        filter: "drop-shadow(3px 4px 4px rgba(0,0,0,0.6))",
      }}
    >
      {/* Shoulder pivot — rotates entire arm */}
      <div
        ref={shoulderRef}
        className="relative w-full"
        style={{ transformOrigin: mirror ? "top right" : "top left" }}
      >
        {/* Upper arm segment */}
        <img
          src={ARM_SRC}
          alt=""
          className="w-full h-auto"
          style={{ ...imgMirror, clipPath: CLIP_UPPER }}
          draggable={false}
        />

        {/* Elbow pivot — rotates forearm + hand */}
        <div
          ref={elbowRef}
          className="absolute inset-0"
          style={{ transformOrigin: joints.elbow }}
        >
          {/* Forearm segment */}
          <img
            src={ARM_SRC}
            alt=""
            className="w-full h-auto"
            style={{ ...imgMirror, clipPath: CLIP_FOREARM }}
            draggable={false}
          />

          {/* Wrist pivot — rotates hand only */}
          <div
            ref={wristRef}
            className="absolute inset-0"
            style={{ transformOrigin: joints.wrist }}
          >
            {/* Hand segment */}
            <img
              src={ARM_SRC}
              alt=""
              className="w-full h-auto"
              style={{ ...imgMirror, clipPath: CLIP_HAND }}
              draggable={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export function Appendages() {
  const refs = useArmAnimation();

  return (
    <>
      <ArticulatedArm
        shoulderRef={refs.leftShoulder}
        elbowRef={refs.leftElbow}
        wristRef={refs.leftWrist}
        mirror={false}
        side="left"
      />
      <ArticulatedArm
        shoulderRef={refs.rightShoulder}
        elbowRef={refs.rightElbow}
        wristRef={refs.rightWrist}
        mirror={true}
        side="right"
      />
    </>
  );
}
