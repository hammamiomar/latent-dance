/**
 * usePanelDrag — title-bar drag for floating Win95 panels over the belly.
 *
 * One implementation for SlotConfigPanel and DestinationPanelWrapper
 * (previously two hand-rolled copies that had drifted: one forgot
 * stopPropagation, so dragging its title bar could also grab the Matter.js
 * body underneath). Listeners attach to the document on mousedown and
 * detach on mouseup, so the drag keeps tracking outside the panel.
 */

import { useCallback, useRef } from 'react';

interface PanelDragOptions {
  position: { x: number; y: number };
  setPosition: (position: { x: number; y: number }) => void;
  /** Live container bounds — panels clamp inside them while dragging. */
  getBounds: () => { width: number; height: number };
  panelWidth: number;
  panelHeight: number;
  padding?: number;
}

export function usePanelDrag({
  position,
  setPosition,
  getBounds,
  panelWidth,
  panelHeight,
  padding = 20,
}: PanelDragOptions) {
  const isDraggingRef = useRef(false);
  const dragOffsetRef = useRef({ x: 0, y: 0 });

  const onTitleBarMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();
      // Keep the Matter.js mouse constraint from grabbing an orb underneath
      e.stopPropagation();

      isDraggingRef.current = true;
      dragOffsetRef.current = {
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      };

      const handleMouseMove = (event: MouseEvent) => {
        if (!isDraggingRef.current) return;
        const bounds = getBounds();
        setPosition({
          x: Math.max(
            padding,
            Math.min(bounds.width - panelWidth - padding, event.clientX - dragOffsetRef.current.x),
          ),
          y: Math.max(
            padding,
            Math.min(bounds.height - panelHeight - padding, event.clientY - dragOffsetRef.current.y),
          ),
        });
      };

      const handleMouseUp = () => {
        isDraggingRef.current = false;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [position, setPosition, getBounds, panelWidth, panelHeight, padding],
  );

  return { isDraggingRef, onTitleBarMouseDown };
}
