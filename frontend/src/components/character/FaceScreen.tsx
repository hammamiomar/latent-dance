/**
 * FaceScreen - Narrow bezel monitor for the character's face.
 *
 * In square layout, the face is ~60% of body width, centered in the
 * TopZone flex row (vent slots fill the remaining space on each side).
 * Reports its bounding rect to useLayoutStore for the body shader.
 */

import { useRef, useEffect, type ReactNode } from "react";
import { useLayoutStore } from "../../stores/useLayoutStore";

export function FaceScreen({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    const update = () => {
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      useLayoutStore.getState().setFaceRect([
        rect.left / vw,
        rect.top / vh,
        rect.right / vw,
        rect.bottom / vh,
      ]);
    };

    const observer = new ResizeObserver(update);
    observer.observe(ref.current);
    update(); // Initial measurement
    return () => observer.disconnect();
  }, []);

  return (
    <div className="shrink-0 pt-3 pb-1" style={{ width: "60%" }}>
      <div ref={ref} className="face-screen w-full relative" style={{ height: 110 }}>
        {children || null}
      </div>
    </div>
  );
}
