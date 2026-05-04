import { beforeEach, describe, expect, it, vi } from "vitest";
import { useBlockStore } from "../stores/useBlockStore";
import { useCompositionStore } from "../stores/useCompositionStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import type { AgentVisualAction } from "../types/agentBridge";
import { applyAgentVisualAction, type AgentPlanApplySenders } from "./agentPlanApply";

function makeSenders() {
  return {
    sendUpdateBlockConfig: vi.fn(),
    sendSetDestination: vi.fn(),
    sendClearDestination: vi.fn(),
    sendFreezeBlend: vi.fn(),
    sendSetBlendPosition: vi.fn(),
    sendSetDestinationMode: vi.fn(),
    sendSetReactiveConfig: vi.fn(),
    sendSetDestinationLink: vi.fn(),
    sendSetCompositionConfig: vi.fn(),
  } satisfies AgentPlanApplySenders;
}

describe("agentPlanApply", () => {
  beforeEach(() => {
    useBlockStore.getState().resetToDefaults();
    useCompositionStore.getState().reset();
    useDestinationStore.getState().reset();
  });

  it("freezes the current backend blend without mutating local destination slots", () => {
    const senders = makeSenders();
    useDestinationStore.getState().setDestination("prompt", "b", {
      type: "prompt",
      label: "old prompt",
      prompt: "old prompt",
    });

    const change = applyAgentVisualAction({
      action: "freeze_blend",
      space: "prompt",
      target_slot: "b",
    }, senders);

    expect(senders.sendFreezeBlend).toHaveBeenCalledWith("prompt", "b");
    expect(useDestinationStore.getState().prompt.destinationB?.prompt).toBe("old prompt");
    expect(change).toMatchObject({
      action: "freeze_blend",
      target: "prompt:b",
    });
  });

  it("sets prompt destinations in local state and forwards the backend command", () => {
    const senders = makeSenders();
    const action = {
      action: "set_destination",
      space: "prompt",
      slot: "b",
      destination_type: "prompt",
      prompt: "muscly turtle in neon rain",
    } satisfies AgentVisualAction;

    applyAgentVisualAction(action, senders);

    expect(useDestinationStore.getState().prompt.destinationB).toMatchObject({
      type: "prompt",
      label: "muscly turtle in neo...",
      prompt: "muscly turtle in neon rain",
    });
    expect(senders.sendSetDestination).toHaveBeenCalledWith(
      "prompt",
      "b",
      "prompt",
      { seed: undefined, prompt: "muscly turtle in neon rain" },
      "direct",
    );
  });

  it("can build first latent, prompt, and composition state from empty slots", () => {
    const senders = makeSenders();
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

    actions.forEach((action) => applyAgentVisualAction(action, senders));

    const destinations = useDestinationStore.getState();
    expect(destinations.latent.destinationA?.seed).toBe(11);
    expect(destinations.latent.destinationB?.seed).toBe(22);
    expect(destinations.prompt.destinationA?.prompt).toBe("calm watercolor road");
    expect(destinations.prompt.destinationB?.prompt).toBe("bright swirling market");
    expect(useCompositionStore.getState()).toMatchObject({
      distance: 0.7,
      mode: "pulse",
    });
    expect(senders.sendSetDestination).toHaveBeenCalledTimes(4);
    expect(senders.sendSetCompositionConfig).toHaveBeenCalledWith({
      distance: 0.7,
      mode: "pulse",
    });
  });

  it("updates block config locally and forwards the same control message", () => {
    const senders = makeSenders();
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

    applyAgentVisualAction(action, senders);

    const block = useBlockStore.getState().blockMappings["up.0.1"];
    expect(block.linkTarget).toBe("drums_high");
    expect(block.featureId).toBe(4);
    expect(block.featureLabel).toBe("glittering hi-hat sparks");
    expect(block.enabled).toBe(true);
    expect(block.strengthRange).toMatchObject({
      strengthMin: -2,
      stageHome: 0,
      strengthMax: 18,
    });
    expect(senders.sendUpdateBlockConfig).toHaveBeenCalledWith(action);
  });

  it("keeps a runtime guard for action types that bypass validation", () => {
    const senders = makeSenders();

    expect(() => applyAgentVisualAction({
      action: "unknown_action",
    } as unknown as AgentVisualAction, senders)).toThrow("Unsupported agent action");
  });
});
