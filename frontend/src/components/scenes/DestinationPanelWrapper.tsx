/**
 * DestinationPanelWrapper - Shared panel shell with title bar, drag, positioning
 */

import { useRef, useEffect, useCallback, useState, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { usePanelDrag } from '../../hooks/usePanelDrag';

const PANEL_WIDTH = 340;

interface DestinationPanelWrapperProps {
  /** Panel title */
  title: string;
  /** Accent color for the title bar */
  accentColor: string;
  /** Is panel open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Position of the orb (center point) */
  orbPosition?: { x: number; y: number };
  /** Panel height */
  height: number;
  /** Auto-expand height based on content */
  autoHeight?: boolean;
  /** Whether panel should appear on left (latent) or right (prompt) of orb */
  side: 'left' | 'right';
  /** Panel content */
  children: ReactNode;
  /** Container dimensions (physics world = belly screen content area) */
  containerSize?: { width: number; height: number };
}

export function DestinationPanelWrapper({
  title,
  accentColor,
  isOpen,
  onClose,
  orbPosition,
  height,
  autoHeight = false,
  side,
  children,
  containerSize,
}: DestinationPanelWrapperProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [measuredHeight, setMeasuredHeight] = useState<number>(height);
  const [panelPosition, setPanelPosition] = useState({ x: 200, y: 200 });
  const didInitPositionRef = useRef(false);
  const prevScrollHeightRef = useRef(0);
  const effectiveHeight = autoHeight ? measuredHeight : height;

  // Container bounds — prefer prop (physics world dims = belly screen content area).
  // DOM fallback uses window dimensions which is wrong in desktop mode (belly << window).
  const getBounds = useCallback(() => {
    if (containerSize && containerSize.width > 0) return containerSize;
    const parent = panelRef.current?.parentElement;
    if (!parent) return { width: 600, height: 600 };
    const rect = parent.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }, [containerSize]);

  // Calculate initial position
  const calculateInitialPosition = useCallback(() => {
    const bounds = getBounds();
    const pos = orbPosition || { x: bounds.width / 2, y: bounds.height / 2 };
    const padding = 20;
    const orbRadius = 40;

    // Position based on side
    let x = side === 'left'
      ? pos.x + orbRadius + padding
      : pos.x - orbRadius - padding - PANEL_WIDTH;

    let y = pos.y - effectiveHeight / 2;

    // Constrain to container
    if (x + PANEL_WIDTH > bounds.width - padding) {
      x = bounds.width - padding - PANEL_WIDTH;
    }
    if (x < padding) x = padding;
    if (y < padding) y = padding;
    if (y + effectiveHeight > bounds.height - padding) {
      y = bounds.height - padding - effectiveHeight;
    }

    return { x, y };
  }, [orbPosition, side, effectiveHeight, getBounds]);

  // Click outside handler
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        const target = e.target as Element;
        if (!target.closest('.destination-orb')) {
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
  }, [isOpen, onClose]);

  // Set initial position when opened or height changes
  useEffect(() => {
    if (!isOpen) {
      didInitPositionRef.current = false;
      return;
    }
    if (!didInitPositionRef.current) {
      setPanelPosition(calculateInitialPosition());
      didInitPositionRef.current = true;
    }
  }, [isOpen, calculateInitialPosition]);

  // Measure content height when autoHeight is enabled
  useEffect(() => {
    if (!isOpen || !autoHeight) return;
    const node = panelRef.current;
    if (!node) return;

    const update = () => {
      const padding = 20;
      const maxHeight = Math.max(200, getBounds().height - padding * 2);
      const next = Math.min(node.scrollHeight, maxHeight);
      setMeasuredHeight((prev) => (Math.abs(prev - next) > 1 ? next : prev));
    };

    update();
    const ro = new ResizeObserver(update);
    ro.observe(node);
    window.addEventListener('resize', update);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', update);
    };
  }, [isOpen, autoHeight, getBounds]);

  // Native wheel event — must use addEventListener (not React onWheel) because
  // Matter.js adds a wheel listener on the container that calls preventDefault(),
  // killing scroll. React's synthetic events fire too late (event delegation) to
  // stop the native event from reaching Matter.js. Native stopPropagation fires
  // before the event bubbles to the Matter.js container.
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    const stop = (e: WheelEvent) => e.stopPropagation();
    el.addEventListener('wheel', stop);
    return () => el.removeEventListener('wheel', stop);
  }, [isOpen]);

  // Auto-scroll when content grows (new sections appear via conditional rendering).
  // MutationObserver fires on DOM insertions; if scrollHeight jumps >50px a new
  // section appeared and we smooth-scroll to show it.
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
        }
        prevScrollHeightRef.current = newHeight;
      });
    });

    observer.observe(el, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [isOpen]);

  // Title bar drag (shared floating-panel behavior)
  const { isDraggingRef, onTitleBarMouseDown } = usePanelDrag({
    position: panelPosition,
    setPosition: setPanelPosition,
    getBounds,
    panelWidth: PANEL_WIDTH,
    panelHeight: effectiveHeight,
  });

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={panelRef}
          className="win95-panel absolute z-[150] flex flex-col"
          style={{
            left: panelPosition.x,
            top: panelPosition.y,
            width: PANEL_WIDTH,
            maxHeight: Math.min(680, getBounds().height - panelPosition.y - 20),
          }}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.2 }}
        >
          {/* Title Bar */}
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
              <span className="win95-title-bar__text">
                {title}
              </span>
            </div>
            <div className="win95-title-bar__buttons">
              <button className="win95-title-btn" onClick={onClose}>
                X
              </button>
            </div>
          </div>

          {/* Content — scrolls when taller than available space */}
          <div
            ref={contentRef}
            className="p-4 flex flex-col gap-4 flex-1 min-h-0 overflow-y-auto"
          >
            {children}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
