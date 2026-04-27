/**
 * Canvas Sampling Hook
 *
 * Samples the video canvas at 10Hz and extracts color/lighting data.
 * Uses a small offscreen canvas (32x32) for efficient analysis.
 *
 * Extraction pipeline:
 * 1. Draw canvas to 32x32 offscreen canvas
 * 2. Read pixel data
 * 3. Calculate dominant color (weighted average)
 * 4. Calculate overall brightness (luminance)
 * 5. Find 2 brightest 8x8 regions (hot spots)
 *
 * Performance target: <1ms per sample at 10Hz
 */

import { useEffect, useRef, useCallback } from "react";
import { useCanvasLightingStore, type HotSpot } from '../stores/useCanvasLightingStore';

const SAMPLE_SIZE = 32; // Downsample to 32x32
const SAMPLE_INTERVAL_MS = 100; // 10Hz sampling
const REGION_SIZE = 8; // 8x8 regions for hot spot detection

interface CanvasSamplingOptions {
  /** Sampling interval in ms (default 100 = 10Hz) */
  intervalMs?: number;
  /** Enable/disable sampling */
  enabled?: boolean;
}

/**
 * Hook to sample a canvas element and extract lighting data
 */
export function useCanvasSampling(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: CanvasSamplingOptions = {}
) {
  const { intervalMs = SAMPLE_INTERVAL_MS, enabled = true } = options;

  // Offscreen canvas for efficient sampling
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const updateFromSample = useCanvasLightingStore((s) => s.updateFromSample);
  const setIsSampling = useCanvasLightingStore((s) => s.setIsSampling);

  // Initialize offscreen canvas
  useEffect(() => {
    if (typeof document === 'undefined') return;

    const offscreen = document.createElement('canvas');
    offscreen.width = SAMPLE_SIZE;
    offscreen.height = SAMPLE_SIZE;
    offscreenRef.current = offscreen;
    ctxRef.current = offscreen.getContext('2d', { willReadFrequently: true });
  }, []);

  // Core sampling function
  const sampleCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const offscreen = offscreenRef.current;
    const ctx = ctxRef.current;

    if (!canvas || !offscreen || !ctx) return;

    // Early exit if canvas has no dimensions
    if (canvas.width === 0 || canvas.height === 0) return;

    try {
      // Draw source canvas to small offscreen (automatic downsampling)
      ctx.drawImage(canvas, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);

      // Read pixel data
      const imageData = ctx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
      const pixels = imageData.data;

      // === CALCULATE DOMINANT COLOR ===
      // Weighted average with brightness bias (brighter pixels contribute more)
      let totalR = 0, totalG = 0, totalB = 0;
      let totalWeight = 0;
      let totalLuminance = 0;

      // Also track regions for hot spots
      const regionCount = SAMPLE_SIZE / REGION_SIZE;
      const regionBrightness: number[][] = [];
      for (let ry = 0; ry < regionCount; ry++) {
        regionBrightness[ry] = [];
        for (let rx = 0; rx < regionCount; rx++) {
          regionBrightness[ry][rx] = 0;
        }
      }

      for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i] / 255;
        const g = pixels[i + 1] / 255;
        const b = pixels[i + 2] / 255;

        // Luminance weight (brighter colors have more influence)
        const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const weight = 0.3 + luminance * 0.7; // Base + brightness contribution

        totalR += r * weight;
        totalG += g * weight;
        totalB += b * weight;
        totalWeight += weight;
        totalLuminance += luminance;

        // Track region brightness
        const pixelIndex = i / 4;
        const px = pixelIndex % SAMPLE_SIZE;
        const py = Math.floor(pixelIndex / SAMPLE_SIZE);
        const rx = Math.floor(px / REGION_SIZE);
        const ry = Math.floor(py / REGION_SIZE);
        if (rx < regionCount && ry < regionCount) {
          regionBrightness[ry][rx] += luminance;
        }
      }

      const pixelCount = pixels.length / 4;
      const dominantColor: [number, number, number] = totalWeight > 0
        ? [
            totalR / totalWeight,
            totalG / totalWeight,
            totalB / totalWeight,
          ]
        : [0.3, 0.28, 0.22];

      const brightness = totalLuminance / pixelCount;

      // === FIND HOT SPOTS ===
      // Find the two brightest regions
      const pixelsPerRegion = REGION_SIZE * REGION_SIZE;
      const regions: { x: number; y: number; intensity: number }[] = [];

      for (let ry = 0; ry < regionCount; ry++) {
        for (let rx = 0; rx < regionCount; rx++) {
          regions.push({
            x: (rx + 0.5) / regionCount, // Center of region, normalized
            y: (ry + 0.5) / regionCount,
            intensity: regionBrightness[ry][rx] / pixelsPerRegion,
          });
        }
      }

      // Sort by intensity descending
      regions.sort((a, b) => b.intensity - a.intensity);

      const hotSpots: [HotSpot, HotSpot] = [
        regions[0] || { x: 0.5, y: 0.5, intensity: 0 },
        regions[1] || { x: 0.5, y: 0.5, intensity: 0 },
      ];

      // Update store
      updateFromSample({ dominantColor, brightness, hotSpots });

    } catch {
      // Canvas may be tainted by cross-origin content
      // Silently fail and use defaults
    }
  }, [canvasRef, updateFromSample]);

  // Set up sampling interval
  useEffect(() => {
    if (!enabled) {
      setIsSampling(false);
      return;
    }

    setIsSampling(true);
    const intervalId = setInterval(sampleCanvas, intervalMs);

    // Initial sample
    sampleCanvas();

    return () => {
      clearInterval(intervalId);
      setIsSampling(false);
    };
  }, [enabled, intervalMs, sampleCanvas, setIsSampling]);

  return { sampleCanvas };
}

/**
 * Hook variant that doesn't require a ref - for manual sampling
 */
export function useCanvasSamplingManual() {
  const updateFromSample = useCanvasLightingStore((s) => s.updateFromSample);

  // Cache offscreen canvas + context (same pattern as useCanvasSampling)
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const sampleCanvas = useCallback(
    (canvas: HTMLCanvasElement) => {
      // Lazy-init offscreen canvas on first call
      if (!offscreenRef.current) {
        const offscreen = document.createElement('canvas');
        offscreen.width = SAMPLE_SIZE;
        offscreen.height = SAMPLE_SIZE;
        offscreenRef.current = offscreen;
        ctxRef.current = offscreen.getContext('2d', { willReadFrequently: true });
      }
      const ctx = ctxRef.current;

      if (!ctx || canvas.width === 0 || canvas.height === 0) return;

      try {
        ctx.drawImage(canvas, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
        const imageData = ctx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
        const pixels = imageData.data;

        let totalR = 0, totalG = 0, totalB = 0;
        let totalWeight = 0;
        let totalLuminance = 0;

        for (let i = 0; i < pixels.length; i += 4) {
          const r = pixels[i] / 255;
          const g = pixels[i + 1] / 255;
          const b = pixels[i + 2] / 255;
          const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          const weight = 0.3 + luminance * 0.7;

          totalR += r * weight;
          totalG += g * weight;
          totalB += b * weight;
          totalWeight += weight;
          totalLuminance += luminance;
        }

        const pixelCount = pixels.length / 4;
        const dominantColor: [number, number, number] = totalWeight > 0
          ? [totalR / totalWeight, totalG / totalWeight, totalB / totalWeight]
          : [0.3, 0.28, 0.22];

        const brightness = totalLuminance / pixelCount;

        // Simplified hot spots (center + offset)
        const hotSpots: [HotSpot, HotSpot] = [
          { x: 0.5, y: 0.4, intensity: brightness },
          { x: 0.5, y: 0.6, intensity: brightness * 0.8 },
        ];

        updateFromSample({ dominantColor, brightness, hotSpots });
      } catch {
        // Silent fail
      }
    },
    [updateFromSample]
  );

  return { sampleCanvas };
}
