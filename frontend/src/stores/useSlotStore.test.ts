/**
 * The core rendering contract: rendering is capability-driven. A 6-slot
 * manifest must yield 6 orb render records — in manifest order, with
 * manifest names and colors, and one physics anchor per record — with
 * zero frontend edits.
 * buildOrbRenderData + slotAnchorPositions are the only index↔slot sources,
 * so pinning them here pins every renderer that consumes them.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { useSlotStore } from "./useSlotStore";
import { buildOrbRenderData } from "../lib/orbRenderData";
import { slotAnchorPositions } from "../lib/slotLayout";
import { parseBackendCapabilities } from "../types/wire/capabilities";
import saeManifest from "../../../tests/fixtures/capabilities.sae_steering.json";
import mockManifest from "../../../tests/fixtures/capabilities.mock.json";

const SAE_CAPS = parseBackendCapabilities(saeManifest);

/** The mock backend's golden manifest: six slots and a non-SAE feature
 * range, validated through the same parser production uses. HAMBA_MODE=mock
 * serves exactly this — the test and the live backend cannot drift apart. */
const SIX_SLOT_CAPS = parseBackendCapabilities(mockManifest);

function initFrom(caps: typeof SAE_CAPS) {
  useSlotStore.setState({ slots: {}, order: [] });
  useSlotStore.getState().initFromCapabilities(caps);
}

describe("useSlotStore.initFromCapabilities", () => {
  beforeEach(() => {
    useSlotStore.setState({ slots: {}, order: [] });
  });

  it("materializes the SAE manifest with the historical default rig", () => {
    initFrom(SAE_CAPS);
    const { slots, order } = useSlotStore.getState();

    expect(order).toEqual(["down.2.1", "mid.0", "up.0.0", "up.0.1"]);
    // Index-cycled seeds: foundation, voice, hits, air
    expect(slots["down.2.1"]).toMatchObject({ linkTarget: "bass", saeRank: 1, intensitySource: "energy_smooth" });
    expect(slots["mid.0"]).toMatchObject({ linkTarget: "vocals", saeRank: 2 });
    expect(slots["up.0.0"]).toMatchObject({ linkTarget: "drums", saeRank: 1, intensitySource: "transient" });
    expect(slots["up.0.1"]).toMatchObject({ linkTarget: "other_high", saeRank: null });

    for (const slot of order) {
      expect(slots[slot].enabled).toBe(false);
      expect(slots[slot].featureId).toBeGreaterThanOrEqual(0);
      expect(slots[slot].featureId).toBeLessThanOrEqual(5119);
      expect(slots[slot].spatialMask).toHaveLength(256);
    }
  });

  it("keeps the user's config when the same vocabulary re-arrives (WS hello)", () => {
    initFrom(SAE_CAPS);
    useSlotStore.getState().setSlotEnabled("mid.0", true);
    useSlotStore.getState().setSlotFeature("mid.0", 1234, "spiral haze");

    useSlotStore.getState().initFromCapabilities(SAE_CAPS);

    expect(useSlotStore.getState().slots["mid.0"]).toMatchObject({
      enabled: true,
      featureId: 1234,
      featureLabel: "spiral haze",
    });
  });

  it("rebuilds when a different backend vocabulary arrives", () => {
    initFrom(SAE_CAPS);
    useSlotStore.getState().initFromCapabilities(SIX_SLOT_CAPS);

    const { slots, order } = useSlotStore.getState();
    expect(order).toHaveLength(6);
    expect(slots["down.2.1"]).toBeUndefined();
    // Feature ids respect the new backend's range
    const [, idMax] = SIX_SLOT_CAPS.feature_id_range;
    for (const slot of order) {
      expect(slots[slot].featureId).toBeLessThanOrEqual(idMax);
    }
  });

  it("ignores writes to slot names the backend never declared", () => {
    initFrom(SAE_CAPS);
    useSlotStore.getState().setSlotEnabled("slot_99", true);
    expect(useSlotStore.getState().slots["slot_99"]).toBeUndefined();
  });
});

describe("6-slot manifest renders 6 orbs (exit criterion)", () => {
  it("yields one render record per manifest slot, in manifest order", () => {
    initFrom(SIX_SLOT_CAPS);
    const { slots, order } = useSlotStore.getState();

    const orbs = buildOrbRenderData(slots, order, SIX_SLOT_CAPS);

    expect(orbs).toHaveLength(6);
    orbs.forEach((orb, i) => {
      expect(orb.slot).toBe(SIX_SLOT_CAPS.slots[i].name);
      expect(orb.displayName).toBe(SIX_SLOT_CAPS.slots[i].display_name);
      expect(orb.color).toBe(SIX_SLOT_CAPS.slots[i].color);
    });
    // Default link targets cycle past the 4 seed entries
    expect(orbs[4].linkTarget).toBe("bass");
    expect(orbs[5].linkTarget).toBe("vocals");
  });

  it("gets one physics anchor per render record — body i is slot order[i]", () => {
    initFrom(SIX_SLOT_CAPS);
    const { slots, order } = useSlotStore.getState();

    const orbs = buildOrbRenderData(slots, order, SIX_SLOT_CAPS);
    const anchors = slotAnchorPositions(order.length, 800, 600);

    expect(anchors).toHaveLength(orbs.length);
  });
});
