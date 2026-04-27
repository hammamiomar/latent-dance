/**
 * PromptDestinationPanel - Panel for prompt space destinations (text prompts only)
 *
 * Clean, focused component - NO seed handling, NO space conditionals.
 * Type safety ensures only valid operations are available.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { AnimatePresence } from 'motion/react';
import { Win95Button } from '../ui/Win95Window';
import { DestinationPanelWrapper } from './DestinationPanelWrapper';
import { BlendSlider } from './BlendSlider';
import { ReactiveConfigSection } from './ReactiveConfigSection';
import { LinkTargetSelectCompact } from '../ui/LinkTargetSelect';
import type { DestinationSlot, Destination } from '../../types/destinations';
import type { LinkTarget } from '../../types/sae';
import type { BaseDestinationPanelProps } from './types';
import {
  SPACE_COLORS,
  PROMPT_DEBOUNCE_MS,
} from './types';

// =============================================================================
// PROMPT INPUT COMPONENT
// =============================================================================

interface PromptInputProps {
  slot: DestinationSlot;
  destination: Destination | null;
  accentColor: string;
  onSetPrompt: (prompt: string) => void;
  onClear: () => void;
}

function PromptInput({ slot, destination, accentColor, onSetPrompt, onClear }: PromptInputProps) {
  const [promptValue, setPromptValue] = useState(destination?.prompt || '');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'pending' | 'saved'>('idle');
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const statusTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSentPromptRef = useRef<string>('');

  // Sync from prop when destination changes externally
  useEffect(() => {
    if (destination?.prompt !== undefined) {
      if (destination.prompt !== lastSentPromptRef.current) {
        setPromptValue(destination.prompt);
      }
    }
  }, [destination?.prompt]);

  // Debounced auto-save
  useEffect(() => {
    if (!promptValue || promptValue === lastSentPromptRef.current) return;

    setSaveStatus('pending');

    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);

    saveTimeoutRef.current = setTimeout(() => {
      if (promptValue.trim()) {
        lastSentPromptRef.current = promptValue;
        onSetPrompt(promptValue);
        setSaveStatus('saved');
        statusTimeoutRef.current = setTimeout(() => setSaveStatus('idle'), 1000);
      }
    }, PROMPT_DEBOUNCE_MS);

    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, [promptValue, onSetPrompt]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
      if (statusTimeoutRef.current) clearTimeout(statusTimeoutRef.current);
    };
  }, []);

  const handleImmediateApply = useCallback(() => {
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    if (promptValue.trim()) {
      lastSentPromptRef.current = promptValue;
      onSetPrompt(promptValue);
      setSaveStatus('saved');
      statusTimeoutRef.current = setTimeout(() => setSaveStatus('idle'), 1000);
    }
  }, [promptValue, onSetPrompt]);

  const statusText = useMemo(() => {
    switch (saveStatus) {
      case 'pending': return '...';
      case 'saved': return '✓';
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

      {/* Prompt textarea */}
      <div className="flex flex-col gap-1">
        <div className="win95-inset p-0">
          <textarea
            value={promptValue}
            onChange={(e) => setPromptValue(e.target.value)}
            placeholder="cyberpunk portrait, neon lights..."
            className="win95-input w-full text-xs resize-none"
            rows={4}
            onKeyDown={(e) => {
              // Ctrl/Cmd + Enter for immediate submit
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                handleImmediateApply();
              }
            }}
          />
        </div>
        <span
          className="text-xxs self-end"
          style={{ color: 'var(--color-text-dim)' }}
        >
          auto-saves as you type
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// PROMPT DESTINATION PANEL
// =============================================================================

interface PromptDestinationPanelProps extends BaseDestinationPanelProps {
  /** Set prompt for a slot - ONLY valid callback for prompt space */
  onSetPrompt: (slot: DestinationSlot, prompt: string) => void;
  /** Current link target for linked mode */
  linkTarget: LinkTarget | null;
  /** Set link target for linked mode */
  onSetLinkTarget: (linkTarget: LinkTarget) => void;
}

export function PromptDestinationPanel({
  destinationA,
  destinationB,
  blendPosition,
  mode,
  reactiveConfig,
  isOpen,
  onClose,
  orbPosition,
  onSetPrompt,
  onClearDestination,
  onFreezeBlend,
  onSetBlendPosition,
  onSetMode,
  onSetReactiveConfig,
  linkTarget,
  onSetLinkTarget,
  containerSize,
}: PromptDestinationPanelProps) {
  const colors = SPACE_COLORS.prompt;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const linkTargetMode: 'simple' | 'complex' = showAdvanced ? 'complex' : 'simple';

  // Freeze buttons visible when both destinations are set and blend is between extremes
  const canFreeze = destinationA !== null && destinationB !== null
    && blendPosition > 0.01 && blendPosition < 0.99;

  return (
    <DestinationPanelWrapper
      title="PROMPT SCENES"
      accentColor={colors.accent}
      isOpen={isOpen}
      onClose={onClose}
      orbPosition={orbPosition}
      height={520}
      autoHeight
      side="right"
      containerSize={containerSize}
    >
      {/* Mode Toggle - 3-way: Slider / Reactive / Linked */}
      <div className="flex flex-col gap-2">
        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          MODE
        </span>
        <div className="mode-toggle-3way">
          <button
            className={mode === 'slider' ? 'active' : ''}
            onClick={() => onSetMode('slider')}
            style={mode === 'slider' ? { borderColor: colors.accent } : undefined}
          >
            ◇ SLIDER
          </button>
          <button
            className={mode === 'reactive' ? 'active' : ''}
            onClick={() => onSetMode('reactive')}
            style={mode === 'reactive' ? { borderColor: colors.accent } : undefined}
          >
            ◈ GLOBAL
          </button>
          <button
            className={mode === 'linked' ? 'active' : ''}
            onClick={() => onSetLinkTarget(linkTarget || 'tension')}
            style={mode === 'linked' ? { borderColor: colors.accent } : undefined}
          >
            ⟁ LINKED
          </button>
        </div>
      </div>

      {/* Linked Config Section */}
      <AnimatePresence>
        {mode === 'linked' && (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span
                className="text-xxs uppercase tracking-wider"
                style={{ color: 'var(--color-text-muted)' }}
              >
                LINK TARGET
              </span>
            </div>
            <LinkTargetSelectCompact
              value={linkTarget || 'tension'}
              onChange={onSetLinkTarget}
              mode={linkTargetMode}
            />
          </div>
        )}
      </AnimatePresence>

      {/* Destinations A and B */}
      <div className="flex gap-2">
        <div className="flex-1">
          <PromptInput
            slot="a"
            destination={destinationA}
            accentColor={colors.accent}
            onSetPrompt={(prompt) => onSetPrompt('a', prompt)}
            onClear={() => onClearDestination('a')}
          />
        </div>
        <div className="flex-1">
          <PromptInput
            slot="b"
            destination={destinationB}
            accentColor={colors.accent}
            onSetPrompt={(prompt) => onSetPrompt('b', prompt)}
            onClear={() => onClearDestination('b')}
          />
        </div>
      </div>

      {/* Freeze Blend Buttons — pin the current blend into a slot */}
      {canFreeze && (
        <div className="flex gap-2 justify-center">
          <Win95Button
            className="text-xxs px-2"
            onClick={() => onFreezeBlend('a')}
            title={`Freeze current blend (${Math.round(blendPosition * 100)}%) into A`}
            style={{ color: colors.accent }}
          >
            {'pin \u2192 A'}
          </Win95Button>
          <Win95Button
            className="text-xxs px-2"
            onClick={() => onFreezeBlend('b')}
            title={`Freeze current blend (${Math.round(blendPosition * 100)}%) into B`}
            style={{ color: colors.accent }}
          >
            {'pin \u2192 B'}
          </Win95Button>
        </div>
      )}

      {/* Blend Slider (slider mode only) */}
      {mode === 'slider' && (
        <BlendSlider
          position={blendPosition}
          onChange={onSetBlendPosition}
          accentColor={colors.accent}
          isReactive={false}
        />
      )}

      {/* Dance Config (global/linked) */}
      <AnimatePresence>
        {mode !== 'slider' && (
          <ReactiveConfigSection
            config={reactiveConfig}
            accentColor={colors.accent}
            onChange={onSetReactiveConfig}
            mode={mode === 'linked' ? 'linked' : 'reactive'}
            advanced={showAdvanced}
            onAdvancedChange={setShowAdvanced}
          />
        )}
      </AnimatePresence>

      {/* Info text */}
      <div
        className="text-xxs text-center pt-1"
        style={{ color: 'var(--color-text-dim)' }}
      >
        {mode === 'slider'
          ? 'Drag slider to blend between A and B'
          : mode === 'reactive'
            ? 'Global mix drives the dance stage'
            : `${linkTarget || 'tension'} drives the dance stage`
        }
      </div>
    </DestinationPanelWrapper>
  );
}
