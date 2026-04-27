/**
 * CharacterBody - Brushed dark aluminum body for the Computer Buddy.
 *
 * Fills the window, flex column. The metal surface is rendered by an R3F
 * Canvas (BodyCanvas) behind all HTML content. Dragging is handled by
 * native macOS performWindowDragWithEvent via Bun FFI.
 */

import type { ReactNode } from "react";
import { BodyCanvas } from "./BodyCanvas";

export function CharacterBody({ children }: { children: ReactNode }) {
  return (
    <div className="w-screen h-screen flex flex-col overflow-hidden
                    font-mono select-none bg-transparent">
      <div
        className="relative flex flex-col w-full flex-1 min-h-0
                    character-body overflow-visible"
      >
        <BodyCanvas />
        {children}
      </div>
    </div>
  );
}
