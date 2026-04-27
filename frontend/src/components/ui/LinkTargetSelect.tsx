/**
 * LinkTargetSelect - Grouped dropdown for selecting audio link targets
 *
 * Displays link targets grouped by category with color coding.
 * Categories: Physical stems, HPSS components, Sub-bands, Derived signals.
 */

import { useCallback, useState, useRef, useEffect } from 'react';
import type { LinkTarget } from '../../types/sae';
import {
  PHYSICAL_STEMS,
  HPSS_TARGETS,
  SUBBAND_TARGETS,
  DERIVED_TARGETS,
} from '../../types/sae';

// =============================================================================
// CATEGORY METADATA
// =============================================================================

type LinkCategory = 'physical' | 'hpss' | 'subband' | 'derived';

interface CategoryInfo {
  id: LinkCategory;
  label: string;
  color: string;
  targets: LinkTarget[];
}

const CATEGORIES: CategoryInfo[] = [
  { id: 'physical', label: 'PHYSICAL', color: 'var(--color-text-primary)', targets: PHYSICAL_STEMS },
  { id: 'hpss', label: 'HPSS', color: '#5aaa7a', targets: HPSS_TARGETS },
  { id: 'subband', label: 'SUB-BAND', color: '#7a9aaa', targets: SUBBAND_TARGETS },
  { id: 'derived', label: 'DERIVED', color: '#c9a040', targets: DERIVED_TARGETS },
];

function getCategoriesForMode(mode: 'simple' | 'complex', current: LinkTarget): CategoryInfo[] {
  if (mode === 'complex') return CATEGORIES;
  const allowed = new Set(PHYSICAL_STEMS);
  allowed.add(current);
  return CATEGORIES
    .map((cat) => ({
      ...cat,
      targets: cat.targets.filter((target) => allowed.has(target)),
    }))
    .filter((cat) => cat.targets.length > 0);
}

/** Display labels for link targets */
const TARGET_LABELS: Record<LinkTarget, string> = {
  // Physical
  bass: 'bass',
  drums: 'drums',
  vocals: 'vocals',
  other: 'other',
  // HPSS
  drums_harmonic: 'drums_harmonic',
  drums_percussive: 'drums_percussive',
  other_harmonic: 'other_harmonic',
  other_percussive: 'other_percussive',
  bass_harmonic: 'bass_harmonic',
  bass_percussive: 'bass_percussive',
  vocals_harmonic: 'vocals_harmonic',
  vocals_percussive: 'vocals_percussive',
  // Sub-bands
  drums_low: 'drums_low',
  drums_mid: 'drums_mid',
  drums_high: 'drums_high',
  other_mid: 'other_mid',
  other_high: 'other_high',
  // Derived
  tension: 'tension',
  tonal_distance: 'tonal_distance',
  global: 'global',
};

/** Get category for a link target */
function getCategoryForTarget(target: LinkTarget): CategoryInfo {
  for (const cat of CATEGORIES) {
    if (cat.targets.includes(target)) return cat;
  }
  return CATEGORIES[0]; // Fallback to physical
}

/** Get indicator symbol for category */
function getCategoryIndicator(category: LinkCategory): string {
  switch (category) {
    case 'physical': return '●';
    case 'hpss': return '◑';
    case 'subband': return '◐';
    case 'derived': return '◇';
    default: return '●';
  }
}

// =============================================================================
// LINK TARGET SELECT COMPONENT
// =============================================================================

interface LinkTargetSelectProps {
  value: LinkTarget;
  onChange: (value: LinkTarget) => void;
  label?: string;
  mode?: 'simple' | 'complex';
}

export function LinkTargetSelect({
  value,
  onChange,
  label = 'LINK TARGET',
  mode = 'complex',
}: LinkTargetSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const currentCategory = getCategoryForTarget(value);
  const categories = getCategoriesForMode(mode, value);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  const handleSelect = useCallback((target: LinkTarget) => {
    onChange(target);
    setIsOpen(false);
  }, [onChange]);

  return (
    <div className="relative flex flex-col gap-1" ref={containerRef}>
      {label && (
        <span
          className="text-xs uppercase tracking-wide"
          style={{ color: 'var(--color-text-muted)' }}
        >
          {label}
        </span>
      )}

      {/* Selected value button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="win95-select w-full text-left flex items-center gap-2"
        style={{
          color: currentCategory.color,
          fontWeight: 600,
        }}
      >
        <span style={{ color: currentCategory.color }}>
          {getCategoryIndicator(currentCategory.id)}
        </span>
        <span className="flex-1">{TARGET_LABELS[value]}</span>
        <span
          className="text-xxs"
          style={{ color: 'var(--color-text-dim)' }}
        >
          ▼
        </span>
      </button>

      {/* Dropdown menu - positioned below button, with scrollable list */}
      {isOpen && (
        <div
          className="absolute left-0 right-0 z-50"
          style={{
            top: '100%',
            marginTop: '4px',
            maxHeight: '320px',
            overflowY: 'scroll',
            overscrollBehavior: 'contain',
            background: 'var(--color-void-deep)',
            border: '2px solid',
            borderColor: 'var(--color-win95-light) var(--color-win95-dark) var(--color-win95-dark) var(--color-win95-light)',
            boxShadow: '4px 4px 0 rgba(0,0,0,0.3)',
          }}
        >
          {categories.map((category) => (
            <div key={category.id}>
              {/* Category header */}
              <div
                className="px-2 py-1 text-xxs uppercase tracking-wider flex items-center gap-2"
                style={{
                  background: 'var(--color-void-mid)',
                  borderBottom: '1px solid var(--color-win95-dark)',
                  color: category.color,
                }}
              >
                <span
                  className="w-1 h-3"
                  style={{ background: category.color }}
                />
                {category.label}
              </div>

              {/* Category options */}
              {category.targets.map((target) => (
                <button
                  key={target}
                  type="button"
                  onClick={() => handleSelect(target)}
                  className="w-full px-3 py-1.5 text-left text-sm flex items-center gap-2 hover:bg-[var(--color-accent-dim)] transition-colors"
                  style={{
                    color: value === target ? category.color : 'var(--color-text-primary)',
                    background: value === target ? 'var(--color-accent-dim)' : undefined,
                  }}
                >
                  <span
                    className="text-xxs"
                    style={{ color: category.color }}
                  >
                    {getCategoryIndicator(category.id)}
                  </span>
                  <span>{TARGET_LABELS[target]}</span>
                  {value === target && (
                    <span
                      className="ml-auto text-xxs"
                      style={{ color: category.color }}
                    >
                      ←
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// COMPACT VERSION (for inline use)
// =============================================================================

interface LinkTargetSelectCompactProps {
  value: LinkTarget;
  onChange: (value: LinkTarget) => void;
  mode?: 'simple' | 'complex';
}

export function LinkTargetSelectCompact({
  value,
  onChange,
  mode = 'complex',
}: LinkTargetSelectCompactProps) {
  const currentCategory = getCategoryForTarget(value);
  const categories = getCategoriesForMode(mode, value);

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as LinkTarget)}
      className="win95-select text-sm"
      style={{
        color: currentCategory.color,
        fontWeight: 600,
      }}
    >
      {categories.map((category) => (
        <optgroup key={category.id} label={category.label}>
          {category.targets.map((target) => (
            <option
              key={target}
              value={target}
              style={{ color: category.color }}
            >
              {getCategoryIndicator(category.id)} {TARGET_LABELS[target]}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}
