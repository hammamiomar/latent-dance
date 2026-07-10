/**
 * PerfOverlay - Diagnostic stats display (Shift+D toggle).
 *
 * Extracted from Visualizer so each app shell can position it independently.
 * Browser mode: overlays the canvas. Desktop mode: renders in the face screen.
 *
 * Self-sufficient: backend stats come from usePerfStore (~2Hz, which is also
 * the re-render tick), client-side counters from the render-free clientPerf
 * object — sampled here at that same 2Hz.
 */

import { usePerfStore } from "../../stores/usePerfStore";
import { clientPerf } from "../../lib/clientPerf";

export function PerfOverlay() {
  const perfStats = usePerfStore((s) => s.stats);
  return (
    <div className="perf-overlay">
      <div className="perf-row">
        <span className="perf-label">WS FPS</span>
        <span className="perf-value">{clientPerf.wsFps.toFixed(1)}</span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Drift</span>
        <span className="perf-value">{clientPerf.driftSec.toFixed(2)}s</span>
      </div>
      <div className="perf-row perf-gap" />
      <div className="perf-row">
        <span className="perf-label">Gen FPS</span>
        <span className="perf-value">
          {perfStats.genFps !== undefined ? perfStats.genFps.toFixed(1) : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Queue</span>
        <span className="perf-value">
          {perfStats.queueDepth !== undefined ? perfStats.queueDepth : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Steer</span>
        <span className="perf-value">
          {perfStats.avgSteerMs !== undefined ? `${perfStats.avgSteerMs.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Infer</span>
        <span className="perf-value">
          {perfStats.avgInferMs !== undefined ? `${perfStats.avgInferMs.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">D2H</span>
        <span className="perf-value">
          {perfStats.avgD2hMs !== undefined ? `${perfStats.avgD2hMs.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Encode</span>
        <span className="perf-value">
          {perfStats.encodeMs !== undefined ? `${perfStats.encodeMs.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Total</span>
        <span className="perf-value">
          {perfStats.avgTotalMs !== undefined ? `${perfStats.avgTotalMs.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row perf-gap" />
      <div className="perf-row">
        <span className="perf-label">Meas FPS</span>
        <span className="perf-value">
          {perfStats.measuredFps !== undefined ? perfStats.measuredFps.toFixed(1) : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Del p50</span>
        <span className="perf-value">
          {perfStats.deliveryP50Ms !== undefined ? `${perfStats.deliveryP50Ms.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Del p95</span>
        <span className="perf-value">
          {perfStats.deliveryP95Ms !== undefined ? `${perfStats.deliveryP95Ms.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Jitter</span>
        <span className="perf-value">
          {perfStats.jitterMeanMs !== undefined ? `${perfStats.jitterMeanMs.toFixed(1)}ms` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Drops</span>
        <span className="perf-value">
          {perfStats.dropRate !== undefined ? `${(perfStats.dropRate * 100).toFixed(1)}%` : "--"}
        </span>
      </div>
      <div className="perf-row">
        <span className="perf-label">Lookahead</span>
        <span className="perf-value">
          {perfStats.lookaheadMs !== undefined ? `${perfStats.lookaheadMs.toFixed(0)}ms` : "--"}
        </span>
      </div>
    </div>
  );
}
