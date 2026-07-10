/**
 * Orb render data — the single index-identity contract for the belly.
 *
 * The slot store's `order` array is the only source of slot ordering: the
 * physics body at index i, the orb DOM element at index i, the 3D flower at
 * index i, and the tendril at index i all render slots[order[i]]. This
 * module zips that order with the manifest's display metadata so OrbSystem,
 * BellyScene, and PlantStems draw from one array and can never disagree
 * about which orb is which slot.
 */

import { useMemo } from 'react';
import { useSlotStore } from '../stores/useSlotStore';
import { useCapabilities } from '../stores/useSessionStore';
import type { BackendCapabilities } from '../types/wire/capabilities';
import type { LinkTarget, PhysicalStem, Rank, SlotMapping } from '../types/sae';

export interface OrbRenderData {
  /** Canonical slot name ("down.2.1", "slot_0", ...) */
  slot: string;
  displayName: string;
  shortName: string;
  description: string;
  /** Manifest hex color */
  color: string;
  enabled: boolean;
  linkTarget: LinkTarget;
  saeRank: Rank;
}

export function buildOrbRenderData(
  slots: Record<string, SlotMapping>,
  order: string[],
  capabilities: BackendCapabilities | null,
): OrbRenderData[] {
  const infoByName = new Map((capabilities?.slots ?? []).map((s) => [s.name, s]));
  return order.map((name) => {
    const info = infoByName.get(name);
    const mapping = slots[name];
    return {
      slot: name,
      displayName: info?.display_name ?? name,
      shortName: info?.short_name ?? name,
      description: info?.description ?? '',
      color: info?.color ?? '#888888',
      enabled: mapping?.enabled ?? false,
      linkTarget: mapping?.linkTarget ?? 'other',
      saeRank: mapping?.saeRank ?? null,
    };
  });
}

/** Subscribe to the slot store + manifest and memoize the render array. */
export function useOrbRenderData(): OrbRenderData[] {
  const slots = useSlotStore((s) => s.slots);
  const order = useSlotStore((s) => s.order);
  const capabilities = useCapabilities();
  return useMemo(
    () => buildOrbRenderData(slots, order, capabilities),
    [slots, order, capabilities],
  );
}

/**
 * Collapse any link target to the physical stem whose channels drive the
 * visuals (flower tint, prominence, tendril audio). Sub-bands and HPSS
 * variants inherit their parent stem; derived targets (tension,
 * tonal_distance, global) read as atmosphere → 'other'.
 */
export function physicalStemOf(linkTarget: LinkTarget | undefined): PhysicalStem {
  if (!linkTarget) return 'other';
  if (linkTarget.startsWith('drums')) return 'drums';
  if (linkTarget.startsWith('vocals')) return 'vocals';
  if (linkTarget.startsWith('bass')) return 'bass';
  return 'other';
}
