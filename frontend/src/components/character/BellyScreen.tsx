/**
 * BellyScreen - Thick bezel monitor (the character's belly workspace).
 *
 * Reports inner content dimensions to parent via ResizeObserver.
 * Enforces square aspect ratio visually — the belly monitor is a square CRT.
 * Reports bounding rect to useLayoutStore for the body shader.
 */

import { useRef, useEffect, useCallback, type ReactNode } from "react";
import { useLayoutStore } from "../../stores/useLayoutStore";

interface BellyScreenProps {
  children: ReactNode;
  onResize: (size: { width: number; height: number }) => void;
}

export function BellyScreen({ children, onResize }: BellyScreenProps) {
  const outerRef = useRef<HTMLDivElement>(null);

  // Stable callback for ResizeObserver
  const onResizeRef = useRef(onResize);
  onResizeRef.current = onResize;

  // Throttle via rAF — fires at most once per frame (~16ms), no artificial delay.
  const rafRef = useRef<number | null>(null);

  const handleResize = useCallback(([entry]: ResizeObserverEntry[]) => {
    const w = Math.round(entry.contentRect.width);
    const h = Math.round(entry.contentRect.height);
    const size = Math.min(w, h);
    if (size === 0) return;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      onResizeRef.current({ width: size, height: size });

      // Update layout store for body shader
      const el = outerRef.current;
      if (el) {
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        useLayoutStore.getState().setBellyRect([
          rect.left / vw,
          rect.top / vh,
          rect.right / vw,
          rect.bottom / vh,
        ]);
      }

      rafRef.current = null;
    });
  }, []);

  useEffect(() => {
    if (!outerRef.current) return;
    const observer = new ResizeObserver(handleResize);
    observer.observe(outerRef.current);
    return () => observer.disconnect();
  }, [handleResize]);

  return (
    <div className="relative h-full min-h-0 px-4 py-2 flex items-center justify-center">
      <div
        ref={outerRef}
        className="belly-screen relative"
        style={{ aspectRatio: "1", height: "100%", maxWidth: "100%" }}
      >
        <div className="absolute inset-0 overflow-clip">
          {children}
        </div>
      </div>
    </div>
  );
}
