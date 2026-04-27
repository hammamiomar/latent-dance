/**
 * StrengthRangeSlider - Dual-thumb slider for SAE steering strength bounds
 *
 * Min/max slider that maps physics output (0-1) to strength range.
 * Features Win95 styling with emerald glow fill between thumbs.
 * Number inputs allow arbitrary values beyond the slider range.
 */

import { useCallback, useState, useRef, useEffect } from 'react';
import type { StrengthRange } from '../../types/sae';

// =============================================================================
// CONSTANTS
// =============================================================================

// Slider visual range (inputs can go beyond this)
const SLIDER_MIN = -50;
const SLIDER_MAX = 50;

// =============================================================================
// STRENGTH RANGE SLIDER
// =============================================================================

interface StrengthRangeSliderProps {
  value: StrengthRange;
  onChange: (value: StrengthRange) => void;
  sliderMin?: number;
  sliderMax?: number;
  step?: number;
  label?: string;
}

export function StrengthRangeSlider({
  value,
  onChange,
  sliderMin = SLIDER_MIN,
  sliderMax = SLIDER_MAX,
  step = 0.5,
  label = 'STRENGTH RANGE',
}: StrengthRangeSliderProps) {
  // Local state for text inputs (allows typing before committing)
  const [minInput, setMinInput] = useState(value.strengthMin.toString());
  const [maxInput, setMaxInput] = useState(value.strengthMax.toString());
  const [homeInput, setHomeInput] = useState(value.stageHome.toString());

  // Sync local state when value prop changes
  useEffect(() => {
    setMinInput(value.strengthMin.toString());
    setMaxInput(value.strengthMax.toString());
    setHomeInput(value.stageHome.toString());
  }, [value.strengthMin, value.strengthMax, value.stageHome]);

  // For slider positioning, clamp to visual range
  const min = sliderMin;
  const max = sliderMax;
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<'min' | 'max' | null>(null);

  // Convert pixel position to value
  const positionToValue = useCallback(
    (clientX: number): number => {
      if (!trackRef.current) return 0;
      const rect = trackRef.current.getBoundingClientRect();
      const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const rawValue = min + percent * (max - min);
      // Snap to step
      return Math.round(rawValue / step) * step;
    },
    [min, max, step]
  );

  // Handle mouse move during drag
  const handleMove = useCallback(
    (clientX: number) => {
      if (!dragging) return;
      const newValue = positionToValue(clientX);

      if (dragging === 'min') {
        const clampedMin = Math.max(min, Math.min(max, newValue));
        onChange({
          ...value,
          strengthMin: clampedMin,
        });
      } else {
        const clampedMax = Math.max(min, Math.min(max, newValue));
        onChange({
          ...value,
          strengthMax: clampedMax,
        });
      }
    },
    [dragging, positionToValue, value, min, max, onChange]
  );

  // Mouse events for drag
  useEffect(() => {
    if (!dragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      e.preventDefault();
      handleMove(e.clientX);
    };

    const handleMouseUp = () => {
      setDragging(null);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragging, handleMove]);

  // Double-click to reset to defaults
  const handleDoubleClick = useCallback(
    (thumb: 'min' | 'max') => {
      if (thumb === 'min') {
        onChange({ ...value, strengthMin: -30 });
      } else {
        onChange({ ...value, strengthMax: 30 });
      }
    },
    [value, onChange]
  );

  // Calculate percentages for positioning (clamp to visual range)
  const clampedMin = Math.max(min, Math.min(max, value.strengthMin));
  const clampedMax = Math.max(min, Math.min(max, value.strengthMax));
  const minPercent = ((clampedMin - min) / (max - min)) * 100;
  const maxPercent = ((clampedMax - min) / (max - min)) * 100;
  const rangeStart = Math.min(minPercent, maxPercent);
  const rangeEnd = Math.max(minPercent, maxPercent);
  const homeClamped = Math.max(min, Math.min(max, value.stageHome));
  const homePercent = ((homeClamped - min) / (max - min)) * 100;

  return (
    <div className="flex flex-col gap-2">
      {/* Label */}
      {label && (
        <span
          className="text-xs uppercase tracking-wide"
          style={{ color: 'var(--color-text-muted)' }}
        >
          {label}
        </span>
      )}

      {/* Value inputs */}
      <div className="flex justify-between items-center gap-2">
        <div className="flex items-center gap-1">
          <span
            className="text-xxs uppercase"
            style={{ color: 'var(--color-text-dim)' }}
          >
            min
          </span>
          <input
            type="number"
            value={minInput}
            onChange={(e) => setMinInput(e.target.value)}
            onBlur={() => {
              const parsed = parseFloat(minInput);
              if (!isNaN(parsed)) {
                onChange({
                  ...value,
                  strengthMin: parsed,
                });
              } else {
                setMinInput(value.strengthMin.toString());
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                (e.target as HTMLInputElement).blur();
              }
            }}
            className="win95-input w-14 text-xs font-mono text-center"
            style={{ color: 'var(--color-accent-dim)' }}
            step={step}
          />
        </div>
        <div className="flex items-center gap-1">
          <span
            className="text-xxs uppercase"
            style={{ color: 'var(--color-text-dim)' }}
          >
            max
          </span>
          <input
            type="number"
            value={maxInput}
            onChange={(e) => setMaxInput(e.target.value)}
            onBlur={() => {
              const parsed = parseFloat(maxInput);
              if (!isNaN(parsed)) {
                onChange({
                  ...value,
                  strengthMax: parsed,
                });
              } else {
                setMaxInput(value.strengthMax.toString());
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                (e.target as HTMLInputElement).blur();
              }
            }}
            className="win95-input w-14 text-xs font-mono text-center"
            style={{ color: 'var(--color-accent)' }}
            step={step}
          />
        </div>
        <div className="flex items-center gap-1">
          <span
            className="text-xxs uppercase"
            style={{ color: 'var(--color-text-dim)' }}
          >
            home
          </span>
          <input
            type="number"
            value={homeInput}
            onChange={(e) => setHomeInput(e.target.value)}
            onBlur={() => {
              const parsed = parseFloat(homeInput);
              if (!isNaN(parsed)) {
                onChange({
                  ...value,
                  stageHome: parsed,
                });
              } else {
                setHomeInput(value.stageHome.toString());
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                (e.target as HTMLInputElement).blur();
              }
            }}
            className="win95-input w-14 text-xs font-mono text-center"
            style={{ color: 'var(--color-text-primary)' }}
            step={step}
          />
        </div>
      </div>

      {/* Slider track */}
      <div
        ref={trackRef}
        className="strength-range relative h-6 cursor-pointer"
        onMouseDown={(e) => {
          // Click on track - move nearest handle
          const clickValue = positionToValue(e.clientX);
          const distToMin = Math.abs(clickValue - value.strengthMin);
          const distToMax = Math.abs(clickValue - value.strengthMax);

          if (distToMin <= distToMax) {
            setDragging('min');
            onChange({
              ...value,
              strengthMin: Math.max(min, Math.min(max, clickValue)),
            });
          } else {
            setDragging('max');
            onChange({
              ...value,
              strengthMax: Math.max(min, Math.min(max, clickValue)),
            });
          }
        }}
      >
        {/* Track background */}
        <div className="strength-range__track" />

        {/* Selected range fill with glow */}
        <div
          className="strength-range__fill"
          style={{
            left: `${rangeStart}%`,
            width: `${rangeEnd - rangeStart}%`,
          }}
        />

        {/* Stage home marker */}
        <div
          className="absolute top-0 bottom-0 w-px"
          style={{
            left: `${homePercent}%`,
            background: 'var(--color-text-dim)',
            opacity: 0.6,
          }}
        />

        {/* Min handle */}
        <div
          className="strength-range__thumb"
          style={{
            left: `${minPercent}%`,
            background:
              dragging === 'min'
                ? 'var(--color-void-elevated)'
                : 'var(--color-win95-face)',
            borderColor:
              dragging === 'min'
                ? 'var(--color-win95-dark) var(--color-win95-light) var(--color-win95-light) var(--color-win95-dark)'
                : undefined,
            zIndex: dragging === 'min' ? 3 : 2,
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setDragging('min');
          }}
          onDoubleClick={() => handleDoubleClick('min')}
          title="Double-click to reset to -30"
        >
          {/* Handle grip lines */}
          <div className="absolute inset-x-0.5 top-1/2 -translate-y-1/2 flex flex-col gap-px">
            <div className="h-px" style={{ background: 'var(--color-win95-dark)' }} />
            <div className="h-px" style={{ background: 'var(--color-win95-light)' }} />
            <div className="h-px" style={{ background: 'var(--color-win95-dark)' }} />
          </div>
        </div>

        {/* Max handle */}
        <div
          className="strength-range__thumb"
          style={{
            left: `${maxPercent}%`,
            background:
              dragging === 'max'
                ? 'var(--color-void-elevated)'
                : 'var(--color-win95-face)',
            borderColor:
              dragging === 'max'
                ? 'var(--color-win95-dark) var(--color-win95-light) var(--color-win95-light) var(--color-win95-dark)'
                : undefined,
            zIndex: dragging === 'max' ? 3 : 2,
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            setDragging('max');
          }}
          onDoubleClick={() => handleDoubleClick('max')}
          title="Double-click to reset to 30"
        >
          {/* Handle grip lines */}
          <div className="absolute inset-x-0.5 top-1/2 -translate-y-1/2 flex flex-col gap-px">
            <div className="h-px" style={{ background: 'var(--color-win95-dark)' }} />
            <div className="h-px" style={{ background: 'var(--color-win95-light)' }} />
            <div className="h-px" style={{ background: 'var(--color-win95-dark)' }} />
          </div>
        </div>
      </div>

      {/* Range labels */}
      <div
        className="flex justify-between text-xxs"
        style={{ color: 'var(--color-text-dim)' }}
      >
        <span>◄ left</span>
        <span>right ►</span>
      </div>
    </div>
  );
}
