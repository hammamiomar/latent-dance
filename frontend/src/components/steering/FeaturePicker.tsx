/**
 * SAE Feature Picker — ID spinner + scrollable neighborhood.
 *
 * Layout:
 *   [ID spinner]  current label (read-only)
 *   [search input ...........]
 *   ┌─────────────────────────┐
 *   │ scrollable list (~200)  │
 *   │ centered on current ID  │
 *   └─────────────────────────┘
 *
 * The list shows features near the current ID by default.
 * Typing in the search box filters by label/category instead.
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import type { BlockCode } from '../../types/sae';
import { getBlockFeatures, getFeature, featuresLoaded } from '../../data/featureLoader';
import { getCategoryColor } from '../../data/features';

interface FeaturePickerProps {
  block: BlockCode;
  selectedId: number;
  selectedLabel: string;
  onChange: (featureId: number, featureLabel: string) => void;
  blockColor: string;
}

const NEIGHBORHOOD = 200;
const SEARCH_MAX = 200;

export function FeaturePicker({ block, selectedId, selectedLabel, onChange, blockColor }: FeaturePickerProps) {
  const [search, setSearch] = useState('');
  const listRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);
  const rafIds = useRef<number>(0);
  const features = getBlockFeatures(block);
  const loaded = featuresLoaded();

  const neighborhood = useMemo(() => {
    if (!loaded || features.length === 0) return [];
    const half = Math.floor(NEIGHBORHOOD / 2);
    const start = Math.max(0, selectedId - half);
    const end = Math.min(features.length, start + NEIGHBORHOOD);
    const adjustedStart = Math.max(0, end - NEIGHBORHOOD);
    return features.slice(adjustedStart, end);
  }, [features, selectedId, loaded]);

  const searchResults = useMemo(() => {
    if (!loaded || !search) return null;
    const q = search.toLowerCase();
    return features
      .filter(f =>
        f.label.toLowerCase().includes(q) ||
        f.category.toLowerCase().includes(q)
      )
      .slice(0, SEARCH_MAX);
  }, [features, search, loaded]);

  const displayList = searchResults ?? neighborhood;
  const isSearching = searchResults !== null;

  // Double-rAF: scroll after both React commit and browser layout
  useEffect(() => {
    if (isSearching || !listRef.current) return;
    const id1 = requestAnimationFrame(() => {
      const id2 = requestAnimationFrame(() => {
        selectedRef.current?.scrollIntoView({ block: 'center', behavior: 'instant' });
      });
      rafIds.current = id2;
    });
    rafIds.current = id1;
    return () => cancelAnimationFrame(rafIds.current);
  }, [selectedId, isSearching, displayList.length]);

  const handleSelect = useCallback(
    (id: number) => {
      const entry = getFeature(block, id);
      onChange(id, entry?.label ?? `#${id}`);
      setSearch('');
    },
    [block, onChange]
  );

  const handleIdChange = useCallback(
    (value: string) => {
      const id = parseInt(value);
      if (isNaN(id)) return;
      const clamped = Math.max(0, Math.min(5119, id));
      const entry = getFeature(block, clamped);
      onChange(clamped, entry?.label ?? `#${clamped}`);
    },
    [block, onChange]
  );

  if (!loaded) {
    return (
      <div className="win95-inset px-2 py-1">
        <span className="text-xs" style={{ color: 'var(--color-text-dim)' }}>
          Loading 5,120 features...
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {/* Row 1: ID spinner + current label */}
      <div className="flex items-center gap-2">
        <input
          type="number"
          min={0}
          max={5119}
          value={selectedId}
          onChange={(e) => handleIdChange(e.target.value)}
          className="win95-input font-mono text-center"
          style={{
            width: 64,
            color: blockColor,
            fontWeight: 700,
            fontSize: '0.8rem',
          }}
        />
        <div
          className="flex-1"
          style={{
            color: blockColor,
            fontWeight: 600,
            fontSize: '0.8rem',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={selectedLabel}
        >
          {selectedLabel}
        </div>
        <span
          className="shrink-0 px-1.5 py-0.5 uppercase tracking-wider"
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.55rem',
            fontWeight: 700,
            letterSpacing: '0.08em',
            color: getCategoryColor(getFeature(block, selectedId)?.category ?? 'unknown'),
            background: 'linear-gradient(180deg, #161614 0%, #1c1c1a 100%)',
            border: '1px solid rgba(80, 78, 72, 0.2)',
            borderRadius: '3px',
            boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.4)',
          }}
        >
          {getFeature(block, selectedId)?.category ?? '—'}
        </span>
      </div>

      {/* Row 2: Search input */}
      <div className="win95-inset p-0">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full px-2 py-1.5 bg-transparent border-none outline-none"
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            color: 'var(--color-text-primary)',
          }}
          placeholder="search labels..."
        />
      </div>

      {/* Row 3: Scrollable feature list */}
      <div
        ref={listRef}
        className="win95-inset"
        style={{
          maxHeight: 220,
          overflowY: 'scroll',
          overscrollBehavior: 'contain',
          padding: 0,
        }}
        onWheel={(e) => e.stopPropagation()}
      >
        {displayList.length === 0 && isSearching && (
          <div className="px-2 py-2" style={{ fontSize: '0.75rem', color: 'var(--color-text-dim)' }}>
            no matches for &ldquo;{search}&rdquo;
          </div>
        )}
        {displayList.map((f) => {
          const isActive = f.id === selectedId;
          return (
            <button
              key={f.id}
              ref={isActive ? selectedRef : undefined}
              onClick={() => handleSelect(f.id)}
              title={`#${f.id}  ${f.label}  [${f.category}]`}
              className="flex items-center gap-2 w-full px-2 py-1 text-left cursor-pointer border-none"
              style={{
                background: isActive ? `${blockColor}25` : 'transparent',
                borderLeft: isActive ? `2px solid ${blockColor}` : '2px solid transparent',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                lineHeight: 1.4,
                color: isActive ? blockColor : 'var(--color-text-primary)',
                fontWeight: isActive ? 600 : 400,
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.currentTarget.style.background = 'var(--color-void-elevated)';
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.currentTarget.style.background = 'transparent';
              }}
            >
              <span
                className="inline-block w-2 h-2 rounded-full shrink-0"
                style={{ background: getCategoryColor(f.category) }}
              />
              <span className="opacity-40 shrink-0" style={{ width: 36, textAlign: 'right' }}>
                {f.id}
              </span>
              <span className="flex-1" style={{ wordBreak: 'break-word' }}>{f.label}</span>
              <span className="opacity-30 shrink-0">{f.category}</span>
            </button>
          );
        })}
      </div>

      {/* Footer hint */}
      <div className="text-center" style={{ fontSize: '0.6rem', color: 'var(--color-text-dim)' }}>
        {isSearching
          ? `${displayList.length} results`
          : `features ${neighborhood[0]?.id ?? 0}–${neighborhood[neighborhood.length - 1]?.id ?? 0}`
        }
      </div>
    </div>
  );
}
