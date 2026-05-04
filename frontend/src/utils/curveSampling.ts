export type CurveTrend = 'rising' | 'falling' | 'stable';

export function sampleCurve(
  curve: Float32Array | null | undefined,
  timestamps: Float32Array | null | undefined,
  time: number,
): number {
  if (!curve || !timestamps || curve.length === 0 || timestamps.length === 0) return 0;

  const n = Math.min(curve.length, timestamps.length);
  if (n === 1) return curve[0] ?? 0;

  const firstTime = timestamps[0] ?? 0;
  const lastTime = timestamps[n - 1] ?? firstTime;
  if (time <= firstTime) return curve[0] ?? 0;
  if (time >= lastTime) return curve[n - 1] ?? 0;

  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if ((timestamps[mid] ?? 0) <= time) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const t0 = timestamps[lo] ?? firstTime;
  const t1 = timestamps[hi] ?? t0;
  const v0 = curve[lo] ?? 0;
  const v1 = curve[hi] ?? v0;
  if (t1 <= t0) return v0;

  const alpha = (time - t0) / (t1 - t0);
  return v0 + (v1 - v0) * alpha;
}

export function sampleTrend(
  curve: Float32Array | null | undefined,
  timestamps: Float32Array | null | undefined,
  time: number,
  windowSec = 2.0,
): CurveTrend {
  const now = sampleCurve(curve, timestamps, time);
  const past = sampleCurve(curve, timestamps, time - windowSec);
  const delta = now - past;
  if (delta > 0.05) return 'rising';
  if (delta < -0.05) return 'falling';
  return 'stable';
}

export function sampleWindowStats(
  curve: Float32Array | null | undefined,
  timestamps: Float32Array | null | undefined,
  currentTime: number,
  startTime: number,
  endTime: number,
) {
  if (!curve || !timestamps || curve.length === 0 || timestamps.length === 0) {
    return { start: startTime, end: endTime, mean: 0, min: 0, max: 0, trend: 'stable' as CurveTrend };
  }

  const n = Math.min(curve.length, timestamps.length);
  const values: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const t = timestamps[i] ?? 0;
    if (t >= startTime && t <= endTime) values.push(curve[i] ?? 0);
  }

  if (values.length === 0) {
    const value = sampleCurve(curve, timestamps, currentTime);
    return { start: startTime, end: endTime, mean: value, min: value, max: value, trend: 'stable' as CurveTrend };
  }

  const sum = values.reduce((acc, value) => acc + value, 0);
  return {
    start: startTime,
    end: endTime,
    mean: sum / values.length,
    min: Math.min(...values),
    max: Math.max(...values),
    trend: sampleTrend(curve, timestamps, currentTime, Math.max(1, Math.min(4, endTime - startTime))),
  };
}

export function findSectionIndex(sections: number[], time: number): number {
  if (sections.length === 0) return 0;

  let index = 0;
  for (let i = 0; i < sections.length; i += 1) {
    if (sections[i] <= time) {
      index = i;
    } else {
      break;
    }
  }
  return index;
}

export function sampleLockIndexCurves(
  lockIndex: Record<string, Float32Array>,
  timestamps: Float32Array | null | undefined,
  time: number,
): Record<string, number> {
  const coupling: Record<string, number> = {};
  for (const [pair, curve] of Object.entries(lockIndex)) {
    coupling[pair] = sampleCurve(curve, timestamps, time);
  }
  return coupling;
}
