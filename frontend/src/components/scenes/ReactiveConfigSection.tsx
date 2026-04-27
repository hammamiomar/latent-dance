/**
 * ReactiveConfigSection - Configuration UI for destination dance mode
 *
 * Dropdown-per-stem ranking, range sliders for all numeric controls.
 * Simple mode exposes anchors + sources; Complex mode reveals smoothing/curves/rank weights.
 */

import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Win95Select } from '../ui/Win95Select';
import { StrengthRangeSlider } from '../ui/StrengthRangeSlider';
import { AutoManualToggle } from '../ui/AutoManualToggle';
import type { ReactiveConfig, StemRankings } from '../../types/destinations';
import type {
  IntensityCurve,
  IntensitySource,
  PositionSource,
  Rank,
  SilenceBehavior,
} from '../../types/sae';
import { STEM_COLORS } from '../../data/features';

// =============================================================================
// OPTION LISTS
// =============================================================================

import {
  POSITION_SOURCE_OPTIONS,
  INTENSITY_SOURCE_OPTIONS,
  SILENCE_OPTIONS,
  INTENSITY_CURVE_OPTIONS,
} from '../../data/options';

const DEFAULT_RANK_WEIGHTS: Record<string, number> = {
  '1': 1.0,
  '2': 0.75,
  '3': 0.5,
  '4': 0.25,
  auto: 0.6,
};

// =============================================================================
// STEM RANKING SECTION - Dropdown per stem
// =============================================================================

const STEMS: Array<keyof StemRankings> = ['drums', 'bass', 'vocals', 'other'];

const STEM_RANK_OPTIONS: { value: string; label: string }[] = [
  { value: '1', label: '1 \u2014 Main' },
  { value: '2', label: '2 \u2014 Backup' },
  { value: '3', label: '3 \u2014 Background' },
  { value: '4', label: '4 \u2014 Subtle' },
  { value: 'auto', label: 'Auto' },
];

function rankToString(rank: Rank): string {
  return rank === null ? 'auto' : String(rank);
}

function stringToRank(s: string): Rank {
  return s === 'auto' ? null : (parseInt(s) as Rank);
}

interface StemRankingSectionProps {
  rankings: StemRankings;
  onChange: (rankings: StemRankings) => void;
}

function StemRankingSection({ rankings, onChange }: StemRankingSectionProps) {
  const handleRankChange = (stem: keyof StemRankings, value: string) => {
    onChange({ ...rankings, [stem]: stringToRank(value) });
  };

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xxs uppercase tracking-wide" style={{ color: 'var(--color-text-dim)' }}>
        Stem Rankings
      </span>
      <div className="flex flex-col gap-1.5">
        {STEMS.map((stem) => {
          const stemColor = STEM_COLORS[stem] || '#888';
          return (
            <div key={stem} className="flex items-center gap-2">
              <span
                className="text-xxs font-mono uppercase"
                style={{ color: stemColor, width: '48px', flexShrink: 0 }}
              >
                {stem}
              </span>
              <Win95Select
                value={rankToString(rankings[stem])}
                options={STEM_RANK_OPTIONS}
                onChange={(v) => handleRankChange(stem, v)}
                compact
              />
            </div>
          );
        })}
      </div>
      <span className="text-xxs" style={{ color: 'var(--color-text-dim)', fontSize: '8px' }}>
        Higher rank = more visual influence
      </span>
    </div>
  );
}

// =============================================================================
// WIN95 SLIDER
// =============================================================================

const SLIDER_THUMB_CSS = `
.win95-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 2px;
  background: var(--color-text-muted);
  cursor: pointer;
  border: 1px solid rgba(255,255,255,0.1);
}
.win95-slider::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 2px;
  background: var(--color-text-muted);
  cursor: pointer;
  border: 1px solid rgba(255,255,255,0.1);
}
`;

const sliderTrackStyle = {
  WebkitAppearance: 'none' as const,
  appearance: 'none' as const,
  width: '100%',
  height: '4px',
  background: 'var(--color-void-elevated)',
  borderRadius: '2px',
  outline: 'none',
};

function Win95Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  const display = format ? format(value) : String(value);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-xxs uppercase" style={{ color: 'var(--color-text-dim)' }}>
          {label}
        </span>
        <span className="text-xxs font-mono" style={{ color: 'var(--color-text-muted)' }}>
          {display}
        </span>
      </div>
      <input
        type="range"
        className="win95-slider"
        style={sliderTrackStyle}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

// =============================================================================
// RANK WEIGHTS
// =============================================================================

interface RankWeightsSectionProps {
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
}

function RankWeightsSection({ weights, onChange }: RankWeightsSectionProps) {
  const keys = ['1', '2', '3', '4', 'auto'] as const;
  const merged = { ...DEFAULT_RANK_WEIGHTS, ...weights };

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xxs uppercase tracking-wide" style={{ color: 'var(--color-text-dim)' }}>
        Rank Weights
      </span>
      <div className="flex flex-col gap-1.5">
        {keys.map((key) => (
          <Win95Slider
            key={key}
            label={key === 'auto' ? 'auto' : `r${key}`}
            value={merged[key]}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => onChange({ ...merged, [key]: v })}
            format={(v) => v.toFixed(2)}
          />
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// REACTIVE CONFIG SECTION
// =============================================================================

interface ReactiveConfigSectionProps {
  config: ReactiveConfig;
  accentColor: string;
  onChange: (config: Partial<ReactiveConfig>) => void;
  mode?: 'reactive' | 'linked';
  advanced?: boolean;
  onAdvancedChange?: (advanced: boolean) => void;
}

export function ReactiveConfigSection({
  config,
  accentColor,
  onChange,
  mode = 'reactive',
  advanced,
  onAdvancedChange,
}: ReactiveConfigSectionProps) {
  const [showAdvanced, setShowAdvanced] = useState(advanced ?? false);
  const isGlobal = mode === 'reactive';

  const stageLeft = config.stageLeft ?? -30;
  const stageHome = config.stageHome ?? 0;
  const stageRight = config.stageRight ?? 30;

  const positionSource = config.positionSource ?? 'auto';
  const intensitySource = config.intensitySource ?? 'energy_smooth';
  const silenceBehavior = config.silenceBehavior ?? 'hold_last';
  const positionSmoothingMs = config.positionSmoothingMs ?? 50;
  const driftMs = config.driftMs ?? 1500;
  const intensityCurve = config.intensityCurve ?? 'linear';

  useEffect(() => {
    if (advanced !== undefined) {
      setShowAdvanced(advanced);
    }
  }, [advanced]);

  const intensityGamma = config.intensityGamma ?? 1.0;

  return (
    <motion.div
      className="flex flex-col gap-3 pt-3"
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.2 }}
    >
      <style>{SLIDER_THUMB_CSS}</style>

      <div className="h-px" style={{ background: 'var(--color-void-elevated)' }} />

      <div className="flex items-center gap-2">
        <span className="text-xxs" style={{ color: accentColor }}>
          ◈
        </span>
        <span className="text-xs uppercase tracking-wide" style={{ color: accentColor }}>
          {isGlobal ? 'Global Dance' : 'Linked Dance'}
        </span>
      </div>

      <StrengthRangeSlider
        value={{
          strengthMin: stageLeft,
          strengthMax: stageRight,
          stageHome,
        }}
        onChange={(next) => onChange({
          stageLeft: next.strengthMin,
          stageRight: next.strengthMax,
          stageHome: next.stageHome,
        })}
        label="STAGE ANCHORS"
      />

      <div className="flex flex-col gap-1">
        <Win95Slider
          label="Transition Speed"
          value={config.blendSlewRate ?? 1.5}
          min={0.3}
          max={5.0}
          step={0.1}
          onChange={(v) => onChange({ blendSlewRate: v })}
          format={(v) => `${v.toFixed(1)}/s`}
        />
        <div
          className="flex justify-between text-xxs"
          style={{ color: 'var(--color-text-dim)' }}
        >
          <span>slow morph</span>
          <span>fast snap</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Win95Select<PositionSource>
          value={positionSource}
          options={POSITION_SOURCE_OPTIONS}
          onChange={(positionSource) => onChange({ positionSource })}
          label="POSITION"
          compact
        />
        <Win95Select<IntensitySource>
          value={intensitySource}
          options={INTENSITY_SOURCE_OPTIONS}
          onChange={(intensitySource) => onChange({ intensitySource })}
          label="INTENSITY"
          compact
        />
      </div>

      {isGlobal && (
        <StemRankingSection
          rankings={config.stemRankings || { drums: 1, bass: 2, vocals: null, other: null }}
          onChange={(stemRankings) => onChange({ stemRankings })}
        />
      )}

      <AutoManualToggle
        value={showAdvanced ? 'manual' : 'auto'}
        onChange={(value) => {
          const next = value === 'manual';
          setShowAdvanced(next);
          onAdvancedChange?.(next);
        }}
      />

      {showAdvanced && (
        <div>
          <div
            className="text-xxs uppercase tracking-wider mb-2"
            style={{ color: 'var(--color-text-dim)' }}
          >
            Advanced Settings
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Win95Select<SilenceBehavior>
              value={silenceBehavior}
              options={SILENCE_OPTIONS}
              onChange={(silence_behavior) => onChange({ silenceBehavior: silence_behavior })}
              label="SILENCE"
              compact
            />
            <Win95Select<IntensityCurve>
              value={intensityCurve}
              options={INTENSITY_CURVE_OPTIONS}
              onChange={(intensity_curve) => onChange({ intensityCurve: intensity_curve })}
              label="CURVE"
              compact
            />
          </div>

          <div className="grid grid-cols-2 gap-3 mt-2">
            <Win95Slider
              label="Smooth (ms)"
              value={positionSmoothingMs}
              min={10}
              max={500}
              step={10}
              onChange={(v) => onChange({ positionSmoothingMs: v })}
            />
            <Win95Slider
              label="Drift (ms)"
              value={driftMs}
              min={100}
              max={5000}
              step={100}
              onChange={(v) => onChange({ driftMs: v })}
            />
          </div>

          {intensityCurve === 'gamma' && (
            <div className="mt-2">
              <Win95Slider
                label="Gamma"
                value={intensityGamma}
                min={0.1}
                max={3.0}
                step={0.1}
                onChange={(v) => onChange({ intensityGamma: v })}
                format={(v) => v.toFixed(1)}
              />
            </div>
          )}

          {isGlobal && (
            <div className="mt-2">
              <RankWeightsSection
                weights={config.rankWeights || DEFAULT_RANK_WEIGHTS}
                onChange={(rankWeights) => onChange({ rankWeights })}
              />
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
