/**
 * CompositionPanel - Controls the CompositionEngine (latent space SLERP travel).
 *
 * Replaces LatentDestinationPanel. The backend CompositionEngine uses
 * circle-walking in latent space — this panel exposes its actual parameters:
 *   - Seeds A/B (anchor points)
 *   - Circle radius (distance — how far to walk)
 *   - Mode (auto / pulse / continuous)
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Win95Button } from '../ui/Win95Window';
import { DestinationPanelWrapper } from './DestinationPanelWrapper';
import { useCompositionStore } from '../../stores/useCompositionStore';
import type { CompositionMode } from '../../types/composition';
import type { DestinationSlot, Destination } from '../../types/destinations';
import {
  SPACE_COLORS,
  SEED_DEBOUNCE_MS,
} from './types';

// =============================================================================
// SEED INPUT COMPONENT
// =============================================================================

interface SeedInputProps {
  slot: DestinationSlot;
  destination: Destination | null;
  accentColor: string;
  onSetSeed: (seed: number) => void;
  onClear: () => void;
}

function SeedInput({ slot, destination, accentColor, onSetSeed, onClear }: SeedInputProps) {
  const [seedValue, setSeedValue] = useState(destination?.seed?.toString() || '');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'pending' | 'saved'>('idle');
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const statusTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSentSeedRef = useRef<string>('');

  // Sync from prop when destination changes externally
  useEffect(() => {
    if (destination?.seed !== undefined) {
      const seedStr = destination.seed.toString();
      if (seedStr !== lastSentSeedRef.current) {
        setSeedValue(seedStr);
      }
    }
  }, [destination?.seed]);

  // Debounced auto-save
  useEffect(() => {
    if (!seedValue || seedValue === lastSentSeedRef.current) return;

    setSaveStatus('pending');

    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);

    saveTimeoutRef.current = setTimeout(() => {
      const parsed = parseInt(seedValue);
      if (!isNaN(parsed) && parsed >= 0) {
        lastSentSeedRef.current = seedValue;
        onSetSeed(parsed);
        setSaveStatus('saved');
        statusTimeoutRef.current = setTimeout(() => setSaveStatus('idle'), 1000);
      }
    }, SEED_DEBOUNCE_MS);

    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, [seedValue, onSetSeed]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
      if (statusTimeoutRef.current) clearTimeout(statusTimeoutRef.current);
    };
  }, []);

  const handleRandomSeed = () => {
    const randomSeed = Math.floor(Math.random() * 1000000);
    const randomStr = randomSeed.toString();
    setSeedValue(randomStr);
    lastSentSeedRef.current = randomStr;
    onSetSeed(randomSeed);
    setSaveStatus('saved');
    statusTimeoutRef.current = setTimeout(() => setSaveStatus('idle'), 1000);
  };

  const handleImmediateApply = useCallback(() => {
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    if (seedValue) {
      const parsed = parseInt(seedValue) || 42;
      lastSentSeedRef.current = seedValue;
      onSetSeed(parsed);
      setSaveStatus('saved');
      statusTimeoutRef.current = setTimeout(() => setSaveStatus('idle'), 1000);
    }
  }, [seedValue, onSetSeed]);

  const statusText = useMemo(() => {
    switch (saveStatus) {
      case 'pending': return '...';
      case 'saved': return '\u2713';
      default: return '';
    }
  }, [saveStatus]);

  return (
    <div
      className="flex flex-col gap-2 p-3"
      style={{
        background: 'var(--color-void-deep)',
        border: '2px solid',
        borderColor: 'var(--color-win95-dark) var(--color-win95-light) var(--color-win95-light) var(--color-win95-dark)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold" style={{ color: accentColor }}>
            {slot.toUpperCase()}
          </span>
          <span
            className="text-xxs transition-opacity duration-200"
            style={{
              color: saveStatus === 'saved' ? '#6a6' : 'var(--color-text-dim)',
              opacity: saveStatus === 'idle' ? 0 : 1,
            }}
          >
            {statusText}
          </span>
        </div>
        {destination && (
          <button
            className="text-xxs"
            style={{ color: 'var(--color-text-dim)' }}
            onClick={onClear}
          >
            clear
          </button>
        )}
      </div>

      {/* Current label */}
      {destination && (
        <div
          className="text-xs truncate"
          style={{ color: 'var(--color-text-primary)' }}
          title={destination.label}
        >
          {destination.label}
        </div>
      )}

      {/* Seed input */}
      <div className="flex gap-1">
        <div className="win95-inset flex-1 p-0">
          <input
            type="number"
            value={seedValue}
            onChange={(e) => setSeedValue(e.target.value)}
            placeholder="42"
            className="win95-input w-full text-xs font-mono"
            onKeyDown={(e) => e.key === 'Enter' && handleImmediateApply()}
          />
        </div>
        <Win95Button className="text-xxs px-2" onClick={handleRandomSeed} title="Random seed">
          RND
        </Win95Button>
      </div>
    </div>
  );
}

// =============================================================================
// MODE DESCRIPTIONS
// =============================================================================

const MODE_DESCRIPTIONS: Record<string, string> = {
  auto: 'Adaptive blending of beats + tonal drift',
  pulse: 'Pure beat-synced angular jumps',
  continuous: 'Smooth harmonic drift only',
};

// =============================================================================
// COMPOSITION PANEL
// =============================================================================

interface CompositionPanelProps {
  destinationA: Destination | null;
  destinationB: Destination | null;
  isOpen: boolean;
  onClose: () => void;
  orbPosition?: { x: number; y: number };
  containerSize?: { width: number; height: number };
  onSetSeed: (slot: DestinationSlot, seed: number) => void;
  onClearDestination: (slot: DestinationSlot) => void;
  onSetCompositionConfig: (config: { distance?: number; mode?: CompositionMode }) => void;
}

export function CompositionPanel({
  destinationA,
  destinationB,
  isOpen,
  onClose,
  orbPosition,
  containerSize,
  onSetSeed,
  onClearDestination,
  onSetCompositionConfig,
}: CompositionPanelProps) {
  const colors = SPACE_COLORS.latent;
  const distance = useCompositionStore((state) => state.distance);
  const mode = useCompositionStore((state) => state.mode);
  const setCompositionConfig = useCompositionStore((state) => state.setConfig);

  const handleDistanceChange = useCallback((value: number) => {
    setCompositionConfig({ distance: value });
    onSetCompositionConfig({ distance: value });
  }, [onSetCompositionConfig, setCompositionConfig]);

  const handleModeChange = useCallback((newMode: CompositionMode) => {
    setCompositionConfig({ mode: newMode });
    onSetCompositionConfig({ mode: newMode });
  }, [onSetCompositionConfig, setCompositionConfig]);

  return (
    <DestinationPanelWrapper
      title="COMPOSITION"
      accentColor={colors.accent}
      isOpen={isOpen}
      onClose={onClose}
      orbPosition={orbPosition}
      containerSize={containerSize}
      height={400}
      autoHeight
      side="left"
    >
      {/* Seeds A and B */}
      <div className="flex gap-2">
        <div className="flex-1">
          <SeedInput
            slot="a"
            destination={destinationA}
            accentColor={colors.accent}
            onSetSeed={(seed) => onSetSeed('a', seed)}
            onClear={() => onClearDestination('a')}
          />
        </div>
        <div className="flex-1">
          <SeedInput
            slot="b"
            destination={destinationB}
            accentColor={colors.accent}
            onSetSeed={(seed) => onSetSeed('b', seed)}
            onClear={() => onClearDestination('b')}
          />
        </div>
      </div>

      {/* Circle Radius Slider */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            CIRCLE RADIUS
          </span>
          <span
            className="text-xs font-mono"
            style={{ color: colors.accent }}
          >
            {distance.toFixed(1)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={4}
          step={0.1}
          value={distance}
          onChange={(e) => handleDistanceChange(parseFloat(e.target.value))}
          className="win95-slider w-full"
        />
        <div className="flex justify-between">
          <span className="text-xxs" style={{ color: 'var(--color-text-dim)' }}>static</span>
          <span className="text-xxs" style={{ color: 'var(--color-text-dim)' }}>rapid</span>
        </div>
      </div>

      {/* Mode Toggle */}
      <div className="flex flex-col gap-2">
        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          MODE
        </span>
        <div className="mode-toggle-3way">
          <button
            className={mode === 'auto' ? 'active' : ''}
            onClick={() => handleModeChange('auto')}
            style={mode === 'auto' ? { borderColor: colors.accent } : undefined}
          >
            AUTO
          </button>
          <button
            className={mode === 'pulse' ? 'active' : ''}
            onClick={() => handleModeChange('pulse')}
            style={mode === 'pulse' ? { borderColor: colors.accent } : undefined}
          >
            PULSE
          </button>
          <button
            className={mode === 'continuous' ? 'active' : ''}
            onClick={() => handleModeChange('continuous')}
            style={mode === 'continuous' ? { borderColor: colors.accent } : undefined}
          >
            CONTINUOUS
          </button>
        </div>
      </div>

      {/* Mode description */}
      <div
        className="text-xxs text-center pt-1"
        style={{ color: 'var(--color-text-dim)' }}
      >
        {MODE_DESCRIPTIONS[mode]}
      </div>
    </DestinationPanelWrapper>
  );
}
