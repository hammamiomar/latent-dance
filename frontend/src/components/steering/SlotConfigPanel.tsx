/**
 * SlotConfigPanel - configuration panel for one steering slot.
 *
 * Slot identity (name, color, description) comes from the capability
 * manifest via OrbSystem; the config values come from the slot store. All
 * writes go through lib/slotControls (optimistic store update + wire send).
 *
 * Layout:
 * 1. Link Target
 * 2. SAE Feature
 * 3. Stage Bounds (strength range)
 * 4. Intensity Source (always visible, full-width)
 * 5. Auto/Manual toggle
 * 6. Advanced: Spatial Mask + Curve segmented control + Gamma slider
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { AnimatePresence } from 'motion/react';
import { usePanelDrag } from '../../hooks/usePanelDrag';
import { Win95Button } from '../ui/Win95Window';
import { Win95Select } from '../ui/Win95Select';
import { LinkTargetSelectCompact } from '../ui/LinkTargetSelect';
import { StrengthRangeSlider } from '../ui/StrengthRangeSlider';
import { AutoManualToggle } from '../ui/AutoManualToggle';
import {
  handleSlotLinkTargetChange,
  handleSlotFeatureChange,
  handleSlotStrengthRangeChange,
  handleSlotAutoConfigChange,
  handleSlotSpatialModeChange,
  handleSlotSpatialMaskChange,
  handleSlotIntensitySourceChange,
  handleSlotIntensityCurveChange,
  handleSlotIntensityGammaChange,
  handleSlotSaeRankChange,
  handleToggleSlot,
} from '../../lib/slotControls';
import type {
  SlotMapping,
  LinkTarget,
  Rank,
  SpatialMode,
  StrengthRange,
  IntensitySource,
  IntensityCurve,
} from '../../types/sae';
import { FeaturePicker } from './FeaturePicker';
import { SpatialGrid } from './SpatialGrid';
import { getSlotFeatures } from '../../data/featureLoader';

// =============================================================================
// CONSTANTS
// =============================================================================

const POPUP_WIDTH = 320;
const POPUP_HEIGHT = 680;

import {
  INTENSITY_SOURCE_OPTIONS,
  INTENSITY_CURVE_OPTIONS,
} from '../../data/options';

const SPATIAL_OPTIONS: { value: SpatialMode; label: string; description: string }[] = [
  { value: 'draw', label: 'Draw', description: 'Paint 16×16 spatial mask' },
  { value: 'pitch_aligned', label: 'Pitch Aligned', description: 'Auto from audio pitch' },
];

/** Rank dropdown options */
const RANK_OPTIONS: { value: Rank; short: string; label: string }[] = [
  { value: 1, short: 'R1', label: 'Lead' },
  { value: 2, short: 'R2', label: 'Supp' },
  { value: 3, short: 'R3', label: 'Back' },
  { value: 4, short: 'R4', label: 'Sub' },
  { value: null, short: 'A', label: 'Auto' },
];

// =============================================================================
// SLOT CONFIG PANEL
// =============================================================================

interface SlotConfigPanelProps {
  slot: string;
  mapping: SlotMapping;
  /** Manifest display metadata for the slot */
  displayName: string;
  description: string;
  accentColor: string;
  isOpen: boolean;
  onClose: () => void;
  orbPosition?: { x: number; y: number };
  containerSize?: { width: number; height: number };
}

export function SlotConfigPanel({
  slot,
  mapping,
  displayName,
  description,
  accentColor,
  isOpen,
  onClose,
  orbPosition,
  containerSize,
}: SlotConfigPanelProps) {
  const popupRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const prevScrollHeightRef = useRef(0);

  // Panel position state — compute eagerly from orbPosition to avoid flash at (200,200)
  const [panelPosition, setPanelPosition] = useState(() => {
    const pos = orbPosition || { x: 200, y: 200 };
    const padding = 20;
    const orbRadius = 50;
    // Use containerSize if available, else reasonable default
    const bounds = (containerSize && containerSize.width > 0) ? containerSize : { width: 600, height: 600 };
    let x = pos.x + orbRadius + padding;
    let y = pos.y - POPUP_HEIGHT / 2;
    if (x + POPUP_WIDTH > bounds.width - padding) x = pos.x - orbRadius - padding - POPUP_WIDTH;
    if (x < padding) x = padding;
    if (y < padding) y = padding;
    const maxH = Math.min(POPUP_HEIGHT, bounds.height - padding * 2);
    if (y + maxH > bounds.height - padding) y = bounds.height - padding - maxH;
    if (y < padding) y = padding;
    return { x, y };
  });
  const [showScrollHint, setShowScrollHint] = useState(false);
  const [rankOpen, setRankOpen] = useState(false);
  const rankRef = useRef<HTMLDivElement>(null);

  // Container bounds — use prop (physics world dimensions = belly screen content area).
  // Previously used popupRef.parentElement.getBoundingClientRect() but on first render
  // the ref isn't attached yet, falling back to window dimensions (1440px in Electrobun
  // vs ~500px belly screen) which broke both positioning and scroll maxHeight.
  const getBounds = useCallback(() => {
    if (containerSize && containerSize.width > 0) return containerSize;
    const parent = popupRef.current?.parentElement;
    if (!parent) return { width: 600, height: 600 };
    const rect = parent.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }, [containerSize]);

  // Scroll state — detect when content overflows and user hasn't scrolled to bottom
  const checkScrollState = useCallback(() => {
    const el = contentRef.current;
    if (!el) { setShowScrollHint(false); return; }
    const canScroll = el.scrollHeight > el.clientHeight + 5;
    const nearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 10;
    setShowScrollHint(canScroll && !nearBottom);
  }, []);

  // Calculate initial popup position relative to orb
  const calculateInitialPosition = useCallback(() => {
    const pos = orbPosition || { x: 200, y: 200 };
    const padding = 20;
    const orbRadius = 50;
    const bounds = getBounds();

    let x = pos.x + orbRadius + padding;
    let y = pos.y - POPUP_HEIGHT / 2;

    if (x + POPUP_WIDTH > bounds.width - padding) {
      x = pos.x - orbRadius - padding - POPUP_WIDTH;
    }
    if (x < padding) x = padding;
    if (y < padding) y = padding;

    // Clamp bottom: ensure panel fits within parent bounds.
    // Use actual available height (not POPUP_HEIGHT) so bottom orbs
    // push the panel up enough to be fully visible.
    const maxPanelHeight = Math.min(POPUP_HEIGHT, bounds.height - padding * 2);
    if (y + maxPanelHeight > bounds.height - padding) {
      y = bounds.height - padding - maxPanelHeight;
    }
    if (y < padding) y = padding;

    return { x, y };
  }, [orbPosition, getBounds]);

  // Click outside handler
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        const target = e.target as Element;
        if (!target.closest(`[data-slot="${slot}"]`)) {
          onClose();
        }
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose, slot]);

  // Handlers — adapt child (value) callbacks to the slotControls write path
  const handleLinkTargetChange = useCallback(
    (linkTarget: LinkTarget) => handleSlotLinkTargetChange(slot, linkTarget),
    [slot]
  );

  const handleFeatureChange = useCallback(
    (featureId: number, featureLabel: string) => {
      handleSlotFeatureChange(slot, featureId, featureLabel);
    },
    [slot]
  );

  const handleStrengthRangeChange = useCallback(
    (range: StrengthRange) => handleSlotStrengthRangeChange(slot, range),
    [slot]
  );

  const handleAutoConfigChange = useCallback(
    (mode: 'auto' | 'manual') => handleSlotAutoConfigChange(slot, mode === 'auto'),
    [slot]
  );

  const handleSpatialModeChange = useCallback(
    (value: SpatialMode) => handleSlotSpatialModeChange(slot, value),
    [slot]
  );

  const handleSpatialMaskChange = useCallback(
    (mask: number[]) => handleSlotSpatialMaskChange(slot, mask),
    [slot]
  );

  const handleIntensitySourceChange = useCallback(
    (source: IntensitySource) => handleSlotIntensitySourceChange(slot, source),
    [slot]
  );

  const handleIntensityCurveChange = useCallback(
    (curve: IntensityCurve) => handleSlotIntensityCurveChange(slot, curve),
    [slot]
  );

  const handleIntensityGammaChange = useCallback(
    (value: number) => handleSlotIntensityGammaChange(slot, value),
    [slot]
  );

  const handleToggle = useCallback(() => handleToggleSlot(slot), [slot]);

  const handleRankSelect = useCallback(
    (rank: Rank) => {
      handleSlotSaeRankChange(slot, rank);
      setRankOpen(false);
    },
    [slot]
  );

  // Randomize all settings
  const handleRandomize = useCallback(() => {
    const allFeatures = getSlotFeatures(slot);
    if (allFeatures.length > 0) {
      const randomFeature = allFeatures[Math.floor(Math.random() * allFeatures.length)];
      handleSlotFeatureChange(slot, randomFeature.id, randomFeature.label);
    }

    // Random strength range
    const minVal = Math.random() * 10;
    const maxVal = minVal + 5 + Math.random() * 15;
    handleSlotStrengthRangeChange(slot, {
      strengthMin: Math.round(minVal * 10) / 10,
      strengthMax: Math.round(maxVal * 10) / 10,
      stageHome: mapping.strengthRange.stageHome,
    });

  }, [slot, mapping.strengthRange.stageHome]);

  // Close rank dropdown on click outside
  useEffect(() => {
    if (!rankOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (rankRef.current && !rankRef.current.contains(e.target as Node)) {
        setRankOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [rankOpen]);

  // Position calculation on open — only recompute when panel opens or slot changes,
  // NOT when orbPosition updates (that would make the panel follow the orb during drag)
  useEffect(() => {
    if (isOpen) {
      setPanelPosition(calculateInitialPosition());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, slot]);

  // Re-check scroll state when content layout changes
  useEffect(() => {
    if (isOpen) {
      checkScrollState();
      const timer = setTimeout(checkScrollState, 100);
      return () => clearTimeout(timer);
    }
  }, [isOpen, mapping.autoConfig, mapping.spatialMode, checkScrollState]);

  // Native wheel event — must use addEventListener (not React onWheel) because
  // Matter.js adds a wheel listener on the container that calls preventDefault(),
  // killing scroll. React's synthetic events use delegation so fire too late.
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    const stop = (e: WheelEvent) => e.stopPropagation();
    el.addEventListener('wheel', stop);
    return () => el.removeEventListener('wheel', stop);
  }, [isOpen]);

  // Generic auto-scroll: when new content appears (conditional rendering adds DOM
  // nodes), detect the scrollHeight jump and smooth-scroll to show the new section.
  // Handles all cases: auto→manual toggle, gamma slider, spatial grid, etc.
  useEffect(() => {
    const el = contentRef.current;
    if (!el || !isOpen) {
      prevScrollHeightRef.current = 0;
      return;
    }
    prevScrollHeightRef.current = el.scrollHeight;

    const observer = new MutationObserver(() => {
      requestAnimationFrame(() => {
        if (!el) return;
        const newHeight = el.scrollHeight;
        if (newHeight > prevScrollHeightRef.current + 50) {
          el.scrollTo({ top: newHeight, behavior: 'smooth' });
          setTimeout(checkScrollState, 300);
        }
        prevScrollHeightRef.current = newHeight;
        checkScrollState();
      });
    });

    observer.observe(el, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [isOpen, checkScrollState]);

  // Title bar drag (shared floating-panel behavior)
  const { isDraggingRef, onTitleBarMouseDown } = usePanelDrag({
    position: panelPosition,
    setPosition: setPanelPosition,
    getBounds,
    panelWidth: POPUP_WIDTH,
    panelHeight: POPUP_HEIGHT,
  });

  const currentCurve = mapping.intensityCurve || 'linear';

  return (
    <AnimatePresence>
      {isOpen && (
        <div
          ref={popupRef}
          className="win95-panel absolute z-[150]"
          style={{
            left: panelPosition.x,
            top: panelPosition.y,
            width: POPUP_WIDTH,
            maxHeight: Math.min(680, getBounds().height - panelPosition.y - 20),
            display: 'flex',
            flexDirection: 'column',
            animation: 'panelFadeIn 0.15s ease-out',
          }}
          data-slot={slot}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {/* Title Bar (fixed, never scrolls) */}
          <div
            className="win95-title-bar shrink-0"
            style={{
              background: `linear-gradient(90deg, ${accentColor}40 0%, var(--color-void-elevated) 100%)`,
              borderBottom: `2px solid ${accentColor}60`,
              cursor: isDraggingRef.current ? 'grabbing' : 'grab',
              userSelect: 'none',
            }}
            onMouseDown={onTitleBarMouseDown}
          >
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ background: accentColor, boxShadow: `0 0 6px ${accentColor}` }}
              />
              <span className="win95-title-bar__text">{displayName}</span>
              <span
                style={{
                  fontSize: '0.55rem',
                  fontFamily: 'var(--font-mono)',
                  color: accentColor,
                  opacity: 0.5,
                }}
              >
                {slot}
              </span>
            </div>
            <div className="win95-title-bar__buttons">
              <button className="win95-title-btn" onClick={onClose}>
                X
              </button>
            </div>
          </div>

          {/* Content — scrolls when taller than available space.
              Native wheel listener (useEffect) stops Matter.js from killing scroll. */}
          <div
            ref={contentRef}
            className="pt-2 pb-3 px-4 flex flex-col gap-2 overflow-y-auto flex-1 min-h-0"
            onScroll={checkScrollState}
          >
            {/* Header: Rank + Randomize + Enable (compact, no redundant slot name) */}
            <div className="flex items-center justify-end gap-1">
              {/* Rank Selector */}
              <div ref={rankRef} style={{ position: 'relative' }}>
                <Win95Button
                  className="text-xs px-2"
                  onClick={() => setRankOpen((prev) => !prev)}
                  title="SAE steering rank"
                  style={{
                    // Opacity encodes rank: R1 bright → R4 dim → Auto dimmest
                    opacity: mapping.saeRank != null
                      ? 1.0 - (mapping.saeRank - 1) * 0.15
                      : 0.45,
                  }}
                >
                  {mapping.saeRank != null ? `R${mapping.saeRank}` : 'A'}
                </Win95Button>

                {/* Rank Dropdown */}
                {rankOpen && (
                  <div
                    className="win95-panel"
                    style={{
                      position: 'absolute',
                      top: '100%',
                      left: 0,
                      marginTop: 2,
                      zIndex: 200,
                      minWidth: 110,
                      padding: '3px 0',
                    }}
                  >
                    {RANK_OPTIONS.map((opt) => {
                      const isActive = mapping.saeRank === opt.value;
                      const isAuto = opt.value === null;
                      return (
                        <div key={opt.short}>
                          {/* Separator before Auto */}
                          {isAuto && (
                            <div
                              style={{
                                height: 1,
                                margin: '3px 6px',
                                background: 'var(--color-void-elevated)',
                              }}
                            />
                          )}
                          <button
                            onClick={() => handleRankSelect(opt.value)}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                              width: '100%',
                              padding: '4px 8px',
                              border: 'none',
                              background: isActive ? `${accentColor}20` : 'transparent',
                              cursor: 'pointer',
                              fontFamily: 'var(--font-mono)',
                              fontSize: '10px',
                              color: isActive
                                ? 'var(--color-text-primary)'
                                : 'var(--color-text-muted)',
                              textAlign: 'left',
                            }}
                          >
                            <span style={{
                              color: isActive ? accentColor : 'var(--color-text-dim)',
                              fontSize: '8px',
                              width: 10,
                            }}>
                              {isActive ? (isAuto ? '◆' : '★') : (isAuto ? '◇' : '☆')}
                            </span>
                            <span style={{ fontWeight: isActive ? 600 : 400 }}>
                              {opt.short}
                            </span>
                            <span style={{
                              color: 'var(--color-text-dim)',
                              fontSize: '9px',
                            }}>
                              {opt.label}
                            </span>
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <Win95Button
                className="text-xs px-2"
                onClick={handleRandomize}
                title="Randomize settings"
              >
                RND
              </Win95Button>
              <Win95Button
                className={`text-xs px-3 ${mapping.enabled ? 'win95-button--primary' : ''}`}
                onClick={handleToggle}
                style={{
                  background: mapping.enabled ? `${accentColor}30` : undefined,
                  borderColor: mapping.enabled ? accentColor : undefined,
                }}
              >
                {mapping.enabled ? '● ON' : '○ OFF'}
              </Win95Button>
            </div>

            {/* Link Target Selector */}
            <div className="flex flex-col gap-1">
              <span
                className="text-xs uppercase tracking-wide"
                style={{ color: 'var(--color-text-muted)' }}
              >
                LINK TARGET
              </span>
              <LinkTargetSelectCompact
                value={mapping.linkTarget}
                onChange={handleLinkTargetChange}
                mode={mapping.autoConfig ? 'simple' : 'complex'}
              />
            </div>

            {/* Feature Selector */}
            <div className="flex flex-col gap-1">
              <span
                className="text-xs uppercase tracking-wide"
                style={{ color: 'var(--color-text-muted)' }}
              >
                SAE FEATURE
              </span>
              <FeaturePicker
                slot={slot}
                selectedId={mapping.featureId}
                selectedLabel={mapping.featureLabel}
                onChange={handleFeatureChange}
                accentColor={accentColor}
              />
            </div>

            {/* Stage Bounds */}
            <StrengthRangeSlider
              value={mapping.strengthRange}
              onChange={handleStrengthRangeChange}
              label="STAGE BOUNDS"
            />

            {/* Intensity Source */}
            <Win95Select<IntensitySource>
              value={mapping.intensitySource || 'energy_smooth'}
              options={INTENSITY_SOURCE_OPTIONS}
              onChange={handleIntensitySourceChange}
              label="INTENSITY SOURCE"
              compact
            />

            {/* Auto/Manual Toggle */}
            <AutoManualToggle
              value={mapping.autoConfig ? 'auto' : 'manual'}
              onChange={handleAutoConfigChange}
            />

            {/* Advanced Panel (MANUAL only) — conditionally rendered, not CSS-animated.
                The panel is fixed-size; this content lives in the scroll area. */}
            {!mapping.autoConfig && (
            <div>
              <div
                className="text-xxs uppercase tracking-wider mb-2"
                style={{ color: 'var(--color-text-dim)' }}
              >
                Advanced Settings
              </div>

              {/* Spatial Mode */}
              <Win95Select<SpatialMode>
                value={mapping.spatialMode}
                options={SPATIAL_OPTIONS}
                onChange={handleSpatialModeChange}
                label="SPATIAL"
                compact
              />

              {/* SpatialGrid — shown only in draw mode */}
              {mapping.spatialMode === 'draw' && (
                <SpatialGrid
                  mask={mapping.spatialMask}
                  onChange={handleSpatialMaskChange}
                  blockColor={accentColor}
                />
              )}

              {/* Curve */}
              <div className="flex flex-col gap-1.5 mt-3">
                <span
                  className="text-xxs uppercase tracking-wider"
                  style={{ color: 'var(--color-text-dim)' }}
                >
                  CURVE
                </span>
                <div
                  className="grid grid-cols-3"
                  style={{
                    border: '1px solid var(--color-void-elevated)',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}
                >
                  {INTENSITY_CURVE_OPTIONS.map((opt, i) => {
                    const isActive = currentCurve === opt.value;
                    return (
                      <button
                        key={opt.value}
                        onClick={() => handleIntensityCurveChange(opt.value)}
                        style={{
                          padding: '6px 4px',
                          background: isActive ? 'var(--color-void-elevated)' : 'transparent',
                          border: 'none',
                          borderRight: i < INTENSITY_CURVE_OPTIONS.length - 1
                            ? '1px solid var(--color-void-elevated)'
                            : 'none',
                          cursor: 'pointer',
                          transition: 'background 0.1s ease',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: '2px',
                        }}
                      >
                        <span
                          style={{
                            fontSize: '10px',
                            fontFamily: 'var(--font-mono)',
                            fontWeight: isActive ? 600 : 400,
                            color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-muted)',
                          }}
                        >
                          {opt.label}
                        </span>
                        <span
                          style={{
                            fontSize: '9px',
                            color: isActive ? 'var(--color-text-muted)' : 'var(--color-text-muted)',
                            lineHeight: 1.2,
                            textAlign: 'center',
                          }}
                        >
                          {opt.description}
                        </span>
                      </button>
                    );
                  })}
                </div>

                {currentCurve === 'gamma' && (
                  <div className="flex items-center gap-2 mt-1">
                    <span
                      className="text-xxs font-mono"
                      style={{ color: 'var(--color-text-dim)', flexShrink: 0 }}
                    >
                      {(mapping.intensityGamma ?? 1).toFixed(1)}
                    </span>
                    <input
                      type="range"
                      className="win95-slider"
                      min={0.1}
                      max={3.0}
                      step={0.1}
                      value={mapping.intensityGamma ?? 1}
                      onChange={(e) => handleIntensityGammaChange(parseFloat(e.target.value))}
                      style={{ flex: 1 }}
                    />
                  </div>
                )}
              </div>
            </div>
            )}

            {/* Slot Description */}
            <div
              className="text-xxs text-center pt-1"
              style={{ color: 'var(--color-text-dim)' }}
            >
              {description}
            </div>
          </div>

          {/* Scroll indicator — gradient fade + chevron when content overflows */}
          {showScrollHint && (
            <div
              className="absolute bottom-0 left-4 right-4 pointer-events-none"
              style={{
                height: 28,
                background: 'linear-gradient(transparent, rgba(19, 22, 20, 0.95))',
                display: 'flex',
                alignItems: 'flex-end',
                justifyContent: 'center',
                paddingBottom: 2,
              }}
            >
              <span
                className="scroll-hint-chevron"
                style={{
                  color: 'var(--color-text-dim)',
                  fontSize: 8,
                  letterSpacing: '0.15em',
                }}
              >
                ▼ ▼ ▼
              </span>
            </div>
          )}
        </div>
      )}
    </AnimatePresence>
  );
}
