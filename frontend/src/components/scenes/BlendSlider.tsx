/**
 * BlendSlider - Crossfader between A and B destinations
 *
 * Features number input for direct position entry.
 * Smooth Win95 styling with accent color glow.
 */

import { useState, useRef, useCallback, useEffect } from 'react';

interface BlendSliderProps {
  position: number;
  onChange: (position: number) => void;
  accentColor: string;
  isReactive: boolean;
}

export function BlendSlider({ position, onChange, accentColor, isReactive }: BlendSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [inputValue, setInputValue] = useState(Math.round(position * 100).toString());

  // Sync input when position changes externally
  useEffect(() => {
    if (!isDragging) {
      setInputValue(Math.round(position * 100).toString());
    }
  }, [position, isDragging]);

  // Convert pixel position to value
  const positionToValue = useCallback((clientX: number): number => {
    if (!trackRef.current) return 0;
    const rect = trackRef.current.getBoundingClientRect();
    const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    // Snap to 1% increments
    return Math.round(percent * 100) / 100;
  }, []);

  // Handle drag
  const handleMove = useCallback((clientX: number) => {
    if (!isDragging || isReactive) return;
    const newValue = positionToValue(clientX);
    onChange(newValue);
  }, [isDragging, isReactive, positionToValue, onChange]);

  // Mouse events
  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      e.preventDefault();
      handleMove(e.clientX);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMove]);

  // Handle input blur
  const handleInputBlur = () => {
    const parsed = parseInt(inputValue);
    if (!isNaN(parsed)) {
      const clamped = Math.max(0, Math.min(100, parsed));
      onChange(clamped / 100);
      setInputValue(clamped.toString());
    } else {
      setInputValue(Math.round(position * 100).toString());
    }
  };

  const thumbPercent = position * 100;

  return (
    <div className="flex flex-col gap-2">
      {/* Labels row */}
      <div className="flex items-center justify-between">
        <span
          className="text-xs font-bold"
          style={{ color: accentColor }}
        >
          A
        </span>

        {/* Center: Mode indicator or input */}
        <div className="flex items-center gap-2">
          {isReactive ? (
            <span
              className="text-xxs uppercase tracking-wider"
              style={{ color: 'var(--color-text-dim)' }}
            >
              ◈ AUDIO-DRIVEN
            </span>
          ) : (
            <div className="flex items-center gap-1">
              <input
                type="number"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onBlur={handleInputBlur}
                onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                className="win95-input w-12 text-xs font-mono text-center"
                style={{ color: accentColor }}
                min={0}
                max={100}
              />
              <span className="text-xxs" style={{ color: 'var(--color-text-dim)' }}>%</span>
            </div>
          )}
        </div>

        <span
          className="text-xs font-bold"
          style={{ color: accentColor }}
        >
          B
        </span>
      </div>

      {/* Slider track */}
      <div
        ref={trackRef}
        className="relative h-6 cursor-pointer"
        style={{
          background: 'var(--color-void-deep)',
          border: '2px solid',
          borderColor: 'var(--color-win95-dark) var(--color-win95-light) var(--color-win95-light) var(--color-win95-dark)',
          opacity: isReactive ? 0.6 : 1,
          pointerEvents: isReactive ? 'none' : 'auto',
        }}
        onMouseDown={(e) => {
          if (isReactive) return;
          setIsDragging(true);
          const newValue = positionToValue(e.clientX);
          onChange(newValue);
        }}
      >
        {/* Fill from A to position */}
        <div
          className="absolute top-0 bottom-0 left-0"
          style={{
            width: `${thumbPercent}%`,
            background: `linear-gradient(90deg, ${accentColor}20, ${accentColor}50)`,
            boxShadow: `inset 0 0 10px ${accentColor}20`,
          }}
        />

        {/* Center mark */}
        <div
          className="absolute top-0 bottom-0 w-px"
          style={{
            left: '50%',
            background: 'var(--color-text-dim)',
            opacity: 0.5,
          }}
        />

        {/* Tick marks */}
        {[0.25, 0.75].map((tick) => (
          <div
            key={tick}
            className="absolute top-0 bottom-0 w-px"
            style={{
              left: `${tick * 100}%`,
              background: 'var(--color-win95-dark)',
              opacity: 0.3,
            }}
          />
        ))}

        {/* Thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 select-none"
          style={{
            left: `${thumbPercent}%`,
            width: 14,
            height: 22,
            background: isDragging ? 'var(--color-void-elevated)' : 'var(--color-win95-face)',
            border: '2px solid',
            borderColor: isDragging
              ? 'var(--color-win95-dark) var(--color-win95-light) var(--color-win95-light) var(--color-win95-dark)'
              : 'var(--color-win95-light) var(--color-win95-dark) var(--color-win95-dark) var(--color-win95-light)',
            boxShadow: isDragging ? `0 0 8px ${accentColor}40` : 'none',
            cursor: isDragging ? 'grabbing' : 'grab',
            zIndex: 2,
          }}
          onMouseDown={(e) => {
            if (isReactive) return;
            e.stopPropagation();
            setIsDragging(true);
          }}
        >
          {/* Grip lines */}
          <div className="absolute inset-x-1 top-1/2 -translate-y-1/2 flex flex-col gap-0.5">
            <div className="h-px" style={{ background: 'var(--color-win95-dark)' }} />
            <div className="h-px" style={{ background: 'var(--color-win95-light)' }} />
            <div className="h-px" style={{ background: 'var(--color-win95-dark)' }} />
          </div>
        </div>
      </div>

      {/* Scale labels */}
      <div
        className="flex justify-between text-xxs"
        style={{ color: 'var(--color-text-dim)' }}
      >
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
