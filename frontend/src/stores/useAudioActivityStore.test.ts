/**
 * overallActivity derivation: the store computes the heart/destination glow
 * value once per telemetry message from the stems that enabled slots
 * listen to. This pins the averaging and the virtual→base stem collapse.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { useAudioActivityStore } from "./useAudioActivityStore";
import { useSlotStore } from "./useSlotStore";
import { parseBackendCapabilities } from "../types/wire/capabilities";
import saeManifest from "../../../tests/fixtures/capabilities.sae_steering.json";
import type { AllStems, ExtendedStemActivityMessage, StemChannelData } from "../types/sae";

const SAE_CAPS = parseBackendCapabilities(saeManifest);

const SILENT: StemChannelData = {
  envelope: 0,
  energy_smooth: 0,
  transient: 0,
  flux: 0,
  brightness: 0,
  flash: 0,
  sustain: 0,
};

function makeMessage(
  energy: Partial<Record<AllStems, number>>,
): ExtendedStemActivityMessage {
  const stems = Object.fromEntries(
    (["bass", "drums", "vocals", "other", "drums_low", "drums_mid", "drums_high", "other_mid", "other_high"] as AllStems[])
      .map((stem) => [stem, { ...SILENT, energy_smooth: energy[stem] ?? 0 }]),
  ) as Record<AllStems, StemChannelData>;
  return { type: "extended_activity", audio_time: 1, stems };
}

describe("useAudioActivityStore.overallActivity", () => {
  beforeEach(() => {
    useSlotStore.setState({ slots: {}, order: [] });
    useSlotStore.getState().initFromCapabilities(SAE_CAPS);
    useAudioActivityStore.getState().reset();
  });

  it("averages energy across the base stems of enabled slots", () => {
    const store = useSlotStore.getState();
    store.setSlotLinkTarget("down.2.1", "drums_high");
    store.setSlotEnabled("down.2.1", true);
    store.setSlotLinkTarget("mid.0", "bass");
    store.setSlotEnabled("mid.0", true);
    store.setSlotEnabled("up.0.0", false);
    store.setSlotEnabled("up.0.1", false);

    // drums_high collapses to its base stem: the drums channel is read,
    // not the virtual band
    useAudioActivityStore.getState().updateFromMessage(
      makeMessage({ drums: 0.8, drums_high: 0.1, bass: 0.4 }),
    );

    expect(useAudioActivityStore.getState().overallActivity).toBeCloseTo(0.6);
  });

  it("is zero when no slots are enabled, regardless of stem energy", () => {
    const store = useSlotStore.getState();
    for (const slot of store.order) {
      store.setSlotEnabled(slot, false);
    }

    useAudioActivityStore.getState().updateFromMessage(
      makeMessage({ drums: 1, bass: 1, vocals: 1, other: 1 }),
    );

    expect(useAudioActivityStore.getState().overallActivity).toBe(0);
  });
});
