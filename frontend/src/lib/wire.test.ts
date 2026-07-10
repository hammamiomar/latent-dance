/**
 * Wire vocabulary tests: what the backend actually receives when the wire
 * functions fire — serialized through a fake socket, so JSON semantics
 * (undefined fields dropped) are part of the contract under test.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { attachSocket, send, __resetTransport } from "./transport";
import {
  sendAudioPause,
  sendAudioPlay,
  sendAudioSeek,
  sendAudioTimeUpdate,
  sendClearDestination,
  sendSetDestination,
  sendSetDestinationMode,
  sendStartSAESteering,
  sendStopGeneration,
} from "./wire";
import { handleSlotLinkTargetChange, handleToggleSlot } from "./slotControls";
import { useConnectionStore } from "../stores/useConnectionStore";
import { useSlotStore } from "../stores/useSlotStore";
import { parseBackendCapabilities } from "../types/wire/capabilities";
import saeManifest from "../../../tests/fixtures/capabilities.sae_steering.json";

const SAE_CAPS = parseBackendCapabilities(saeManifest);

let sent: Record<string, unknown>[] = [];

beforeEach(() => {
  sent = [];
  attachSocket({
    readyState: WebSocket.OPEN,
    send: (raw: string) => sent.push(JSON.parse(raw)),
  } as unknown as WebSocket);
});

afterEach(() => {
  __resetTransport();
});

describe("transport", () => {
  it("drops sends silently when no socket is attached", () => {
    __resetTransport();
    expect(() => send({ action: "audio_pause" })).not.toThrow();
    expect(sent).toEqual([]);
  });
});

describe("wire vocabulary", () => {
  it("audio sync messages carry the action names the backend dispatches on", () => {
    sendAudioPlay(10);
    sendAudioPause();
    sendAudioSeek(90);
    sendAudioTimeUpdate(42.5);

    expect(sent).toEqual([
      { action: "audio_play", time: 10 },
      { action: "audio_pause" },
      { action: "audio_seek", time: 90 },
      { action: "audio_timeupdate", time: 42.5 },
    ]);
  });

  it("set_destination omits the unused value field on the wire", () => {
    sendSetDestination("prompt", "b", "prompt", { prompt: "neon rain" });

    // No `seed` key at all — the backend must see absent, not null
    expect(sent).toEqual([{
      action: "set_destination",
      space: "prompt",
      slot: "b",
      destination_type: "prompt",
      prompt: "neon rain",
      replace_mode: "direct",
    }]);
  });

  it("clear_destination names the space and slot", () => {
    sendClearDestination("prompt", "b");
    expect(sent).toEqual([{ action: "clear_destination", space: "prompt", slot: "b" }]);
  });

  it("refuses to send linked mode through set_destination_mode", () => {
    sendSetDestinationMode("prompt", "linked");
    expect(sent).toEqual([]);
  });

  it("start/stop generation optimistically flip isGenerating", () => {
    useConnectionStore.getState().setGenerating(false);

    sendStartSAESteering("song-1");
    expect(useConnectionStore.getState().isGenerating).toBe(true);
    expect(sent).toContainEqual({ action: "start_sae_steering", audio_id: "song-1" });

    sendStopGeneration();
    expect(useConnectionStore.getState().isGenerating).toBe(false);
    expect(sent).toContainEqual({ action: "stop_generation" });
  });
});

describe("slot controls", () => {
  beforeEach(() => {
    useSlotStore.setState({ slots: {}, order: [] });
    useSlotStore.getState().initFromCapabilities(SAE_CAPS);
  });

  it("speak the update_slot_config vocabulary with a slot key", () => {
    handleSlotLinkTargetChange("mid.0", "drums_low");
    handleToggleSlot("mid.0");

    // The backend keys on `slot`; the legacy block dialect is agent-only
    expect(sent).toEqual([
      { action: "update_slot_config", slot: "mid.0", link_target: "drums_low" },
      { action: "update_slot_config", slot: "mid.0", enabled: true },
    ]);
    expect(useSlotStore.getState().slots["mid.0"]).toMatchObject({
      linkTarget: "drums_low",
      enabled: true,
    });
  });
});
