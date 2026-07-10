import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useSlotStore } from "../stores/useSlotStore";
import { useCompositionStore } from "../stores/useCompositionStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import { attachSocket, __resetTransport } from "../lib/transport";
import { parseBackendCapabilities } from "../types/wire/capabilities";
import saeManifest from "../../../tests/fixtures/capabilities.sae_steering.json";
import type { AgentVisualAction } from "../types/agentBridge";
import { applyAgentVisualAction } from "./agentPlanApply";

const SAE_CAPS = parseBackendCapabilities(saeManifest);

/**
 * Wire messages captured as the backend would receive them: JSON round-trip
 * through the fake socket, so undefined fields disappear exactly as they do
 * on the real connection.
 */
let sent: Record<string, unknown>[] = [];

function ofAction(action: string) {
  return sent.filter((message) => message.action === action);
}

describe("agentPlanApply", () => {
  beforeEach(() => {
    // Fresh rig exactly the way production gets one: from the manifest
    useSlotStore.setState({ slots: {}, order: [] });
    useSlotStore.getState().initFromCapabilities(SAE_CAPS);
    useCompositionStore.getState().reset();
    useDestinationStore.getState().reset();
    sent = [];
    attachSocket({
      readyState: WebSocket.OPEN,
      send: (raw: string) => sent.push(JSON.parse(raw)),
    } as unknown as WebSocket);
  });

  afterEach(() => {
    __resetTransport();
  });

  it("freezes the current backend blend without mutating local destination slots", () => {
    useDestinationStore.getState().setDestination("prompt", "b", {
      type: "prompt",
      label: "old prompt",
      prompt: "old prompt",
    });

    const change = applyAgentVisualAction({
      action: "freeze_blend",
      space: "prompt",
      target_slot: "b",
    });

    expect(sent).toContainEqual({ action: "freeze_blend", space: "prompt", target_slot: "b" });
    expect(useDestinationStore.getState().prompt.destinationB?.prompt).toBe("old prompt");
    expect(change).toMatchObject({
      action: "freeze_blend",
      target: "prompt:b",
    });
  });

  it("sets prompt destinations in local state and forwards the backend command", () => {
    const action = {
      action: "set_destination",
      space: "prompt",
      slot: "b",
      destination_type: "prompt",
      prompt: "muscly turtle in neon rain",
    } satisfies AgentVisualAction;

    applyAgentVisualAction(action);

    expect(useDestinationStore.getState().prompt.destinationB).toMatchObject({
      type: "prompt",
      label: "muscly turtle in neo...",
      prompt: "muscly turtle in neon rain",
    });
    // The wire payload has no `seed` key at all — undefined is dropped by JSON
    expect(ofAction("set_destination")).toEqual([{
      action: "set_destination",
      space: "prompt",
      slot: "b",
      destination_type: "prompt",
      prompt: "muscly turtle in neon rain",
      replace_mode: "direct",
    }]);
  });

  it("can build first latent, prompt, and composition state from empty slots", () => {
    const actions = [
      {
        action: "set_destination",
        space: "latent",
        slot: "a",
        destination_type: "seed",
        seed: 11,
      },
      {
        action: "set_destination",
        space: "latent",
        slot: "b",
        destination_type: "seed",
        seed: 22,
      },
      {
        action: "set_destination",
        space: "prompt",
        slot: "a",
        destination_type: "prompt",
        prompt: "calm watercolor road",
      },
      {
        action: "set_destination",
        space: "prompt",
        slot: "b",
        destination_type: "prompt",
        prompt: "bright swirling market",
      },
      {
        action: "set_composition_config",
        distance: 0.7,
        mode: "pulse",
      },
    ] satisfies AgentVisualAction[];

    actions.forEach((action) => applyAgentVisualAction(action));

    const destinations = useDestinationStore.getState();
    expect(destinations.latent.destinationA?.seed).toBe(11);
    expect(destinations.latent.destinationB?.seed).toBe(22);
    expect(destinations.prompt.destinationA?.prompt).toBe("calm watercolor road");
    expect(destinations.prompt.destinationB?.prompt).toBe("bright swirling market");
    expect(useCompositionStore.getState()).toMatchObject({
      distance: 0.7,
      mode: "pulse",
    });
    expect(ofAction("set_destination")).toHaveLength(4);
    expect(sent).toContainEqual({
      action: "set_composition_config",
      distance: 0.7,
      mode: "pulse",
    });
  });

  it("updates block config locally and forwards the same control message", () => {
    const action = {
      action: "update_block_config",
      block: "up.0.1",
      link_target: "drums_high",
      feature_id: 4,
      feature_label: "glittering hi-hat sparks",
      enabled: true,
      stage_left: -2,
      stage_home: 0,
      stage_right: 18,
    } satisfies AgentVisualAction;

    applyAgentVisualAction(action);

    const block = useSlotStore.getState().slots["up.0.1"];
    expect(block.linkTarget).toBe("drums_high");
    expect(block.featureId).toBe(4);
    expect(block.featureLabel).toBe("glittering hi-hat sparks");
    expect(block.enabled).toBe(true);
    expect(block.strengthRange).toMatchObject({
      strengthMin: -2,
      stageHome: 0,
      strengthMax: 18,
    });
    expect(ofAction("update_block_config")).toEqual([action]);
  });

  it("keeps a runtime guard for action types that bypass validation", () => {
    expect(() => applyAgentVisualAction({
      action: "unknown_action",
    } as unknown as AgentVisualAction)).toThrow("Unsupported agent action");
  });
});
