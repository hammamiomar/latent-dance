import { useEffect, useRef, useImperativeHandle } from "react";
import type { Ref } from "react";
import { CANVAS_CONFIG } from "../constants";

interface CanvasProps {
  className?: string;
  ref?: Ref<CanvasHandle>; // ref from parent (App.tsx) - we'll fill it with methods
}

// the "handle" - methods we expose to parent
export interface CanvasHandle {
  renderFrame: (data: ArrayBuffer) => Promise<void>;
  clear: () => void;
  /** Get the raw canvas element for sampling (Phase 3D video sync) */
  getCanvas: () => HTMLCanvasElement | null;
}

export const Canvas = ({ className, ref }: CanvasProps) => {
  // two separate refs here:
  // 1. canvasRef - points to the actual <canvas> DOM element (filled by React automatically)
  // 2. ref (prop) - the handle we give to parent so they can call our methods

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const decodeInFlightRef = useRef(false);
  const pendingDataRef = useRef<ArrayBuffer | null>(null);

  //Initialize canvas context and handle window resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // get the 2d context with perf optimizations
    ctxRef.current = canvas.getContext("2d", {
      alpha: false, // skip transparency processing
      desynchronized: true, // lower latency, allows slight tearing (invisible at 30fps)
    });

    const handleResize = () => {
      if (!canvas) return;

      const { clientWidth, clientHeight } = canvas;

      // canvas has two sizes: CSS size (how big it looks) and buffer size (actual resolution)
      // we sync them so the image is crisp
      if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
        canvas.width = clientWidth;
        canvas.height = clientHeight;

        // Fill with black after resize
        if (ctxRef.current) {
          ctxRef.current.fillStyle = CANVAS_CONFIG.BACKGROUND_COLOR;
          ctxRef.current.fillRect(0, 0, clientWidth, clientHeight);
        }
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  // this fills the ref prop with our methods
  // so parent can do: canvasRef.current.renderFrame(data)
  useImperativeHandle(ref, () => ({
    renderFrame: async (data: ArrayBuffer) => {
      const ctx = ctxRef.current;
      const canvas = canvasRef.current;
      if (!ctx || !canvas) return;

      // Always keep only the latest frame to avoid decode backlog
      pendingDataRef.current = data;
      if (decodeInFlightRef.current) return;

      decodeInFlightRef.current = true;
      try {
        while (pendingDataRef.current) {
          const next = pendingDataRef.current;
          pendingDataRef.current = null;

          const blob = new Blob([next], { type: "image/jpeg" });

          // decode jpeg using hardware acceleration (GPU) - happens off main thread
          const img = await createImageBitmap(blob);

          // Calculate scaling to fill canvas while maintaining aspect ratio
          // (cover behavior, like CSS background-size: cover)
          const scale = Math.max(
            canvas.width / img.width,
            canvas.height / img.height,
          );

          const scaledWidth = img.width * scale;
          const scaledHeight = img.height * scale;

          // Center the image
          const x = (canvas.width - scaledWidth) / 2;
          const y = (canvas.height - scaledHeight) / 2;

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

          // CRITICAL: manually close or we leak memory (30 bitmaps/sec = crash in 30 seconds)
          img.close();
        }
      } catch (error) {
        console.error("[Canvas] Error rendering frame:", error);
      } finally {
        decodeInFlightRef.current = false;
      }
    },

    /**
     * Clear canvas to black
     */
    clear: () => {
      const ctx = ctxRef.current;
      const canvas = canvasRef.current;
      if (!ctx || !canvas) return;

      ctx.fillStyle = CANVAS_CONFIG.BACKGROUND_COLOR;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    },

    /**
     * Get raw canvas element for sampling (Phase 3D video sync)
     */
    getCanvas: () => canvasRef.current,
  }));

  // React automatically sets canvasRef.current to this <canvas> DOM element after render
  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        display: "block",
        width: "100%",
        height: "100%",
      }}
    />
  );
};
