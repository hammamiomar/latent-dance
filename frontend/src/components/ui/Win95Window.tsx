/**
 * Win95Window - Reusable draggable window with title bar.
 *
 * Features:
 * - 3D beveled borders (classic Win95 look)
 * - Draggable by title bar
 * - Minimize/close buttons
 * - Semi-transparent with backdrop blur
 */

import { useRef, useState, useCallback, type ReactNode, type CSSProperties } from "react";
import { motion, AnimatePresence } from "motion/react";
import { useDrag } from "@use-gesture/react";
import { springs } from "../../hooks/animations";

interface Win95WindowProps {
  /** Window title displayed in title bar */
  title: string;
  /** Window content */
  children: ReactNode;
  /** Initial position */
  initialPosition?: { x: number; y: number };
  /** Whether window is visible */
  isOpen: boolean;
  /** Called when close button clicked */
  onClose?: () => void;
  /** Called when minimize button clicked */
  onMinimize?: () => void;
  /** Whether window is minimized (collapsed to title bar) */
  isMinimized?: boolean;
  /** Optional width */
  width?: number | string;
  /** Optional class name */
  className?: string;
  /** Optional z-index */
  zIndex?: number;
}

export function Win95Window({
  title,
  children,
  initialPosition = { x: 100, y: 100 },
  isOpen,
  onClose,
  onMinimize,
  isMinimized = false,
  width = 320,
  className = "",
  zIndex = 150,
}: Win95WindowProps) {
  // Position state
  const [position, setPosition] = useState(initialPosition);
  const containerRef = useRef<HTMLDivElement>(null);

  // Drag handler
  const bind = useDrag(
    ({ offset: [x, y], first, last }) => {
      if (first) {
        // Bring to front on drag start
        if (containerRef.current) {
          containerRef.current.style.zIndex = String(zIndex + 100);
        }
      }
      if (last && containerRef.current) {
        containerRef.current.style.zIndex = String(zIndex);
      }
      setPosition({ x, y });
    },
    {
      from: () => [position.x, position.y],
      bounds: () => {
        const parent = containerRef.current?.parentElement;
        const w = parent?.clientWidth ?? window.innerWidth;
        const h = parent?.clientHeight ?? window.innerHeight;
        return { left: -100, right: w - 100, top: 0, bottom: h - 50 };
      },
    }
  );

  // Handle double-click on title bar to toggle minimize
  const handleTitleDoubleClick = useCallback(() => {
    onMinimize?.();
  }, [onMinimize]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={containerRef}
          className={`win95-panel absolute ${className}`}
          style={{
            left: position.x,
            top: position.y,
            width,
            zIndex,
          }}
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{
            opacity: 1,
            scale: 1,
            y: 0,
            transition: springs.gentle,
          }}
          exit={{
            opacity: 0,
            scale: 0.9,
            y: 20,
            transition: { duration: 0.15 },
          }}
        >
          {/* Title Bar */}
          <div
            {...bind()}
            className="win95-title-bar"
            onDoubleClick={handleTitleDoubleClick}
          >
            <span className="win95-title-bar__text">{title}</span>
            <div className="win95-title-bar__buttons">
              {onMinimize && (
                <button
                  className="win95-title-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onMinimize();
                  }}
                  aria-label="Minimize"
                >
                  _
                </button>
              )}
              {onClose && (
                <button
                  className="win95-title-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClose();
                  }}
                  aria-label="Close"
                >
                  X
                </button>
              )}
            </div>
          </div>

          {/* Content */}
          <AnimatePresence>
            {!isMinimized && (
              <motion.div
                className="p-2"
                initial={{ height: 0, opacity: 0 }}
                animate={{
                  height: "auto",
                  opacity: 1,
                  transition: springs.snappy,
                }}
                exit={{
                  height: 0,
                  opacity: 0,
                  transition: { duration: 0.1 },
                }}
              >
                {children}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ============================================================================
// Win95 Button Component
// ============================================================================

interface Win95ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "icon";
  disabled?: boolean;
  className?: string;
  title?: string;
  style?: CSSProperties;
}

export function Win95Button({
  children,
  onClick,
  variant = "default",
  disabled = false,
  className = "",
  title,
  style,
}: Win95ButtonProps) {
  const variantClass =
    variant === "primary"
      ? "win95-button--primary"
      : variant === "icon"
        ? "win95-button--icon"
        : "";

  return (
    <motion.button
      className={`win95-button ${variantClass} ${className}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={style}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98, y: 1 }}
      transition={springs.snappy}
    >
      {children}
    </motion.button>
  );
}

// ============================================================================
// Win95 Slider
// ============================================================================

interface Win95SliderProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  vertical?: boolean;
  className?: string;
}

export function Win95Slider({
  value,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  vertical = false,
  className = "",
}: Win95SliderProps) {
  return (
    <input
      type="range"
      className={`win95-slider ${vertical ? "win95-slider--vertical" : ""} ${className}`}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(parseFloat(e.target.value))}
    />
  );
}
