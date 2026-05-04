import { describe, expect, it } from "vitest";
import {
  defaultMaxStalenessSec,
  validateAgentVisualPlan,
  type AgentPlanValidationContext,
} from "./agentPlanValidation";

const validContext: AgentPlanValidationContext = {
  armed: true,
  activeSession: true,
  bridgeConnected: true,
  currentAudioTime: 50,
};

const validPlan = {
  based_on_audio_time: 45,
  max_staleness_sec: 10,
  actions: [
    {
      action: "update_block_config",
      block: "up.0.0",
      link_target: "drums_high",
      feature_id: 129,
      spatial_mask: Array(256).fill(1),
      stage_left: -20,
      stage_home: 0,
      stage_right: 20,
    },
  ],
  feature_candidates: [
    {
      block: "up.0.0",
      id: 129,
      label: "sparkle detail",
      category: "object_detail",
      score: 9,
    },
  ],
};

function codes(plan: unknown, context = validContext) {
  const result = validateAgentVisualPlan(plan, context);
  return result.ok ? [] : result.errors.map((item) => item.code);
}

describe("agentPlanValidation", () => {
  it("accepts a valid fresh audio-reactive plan", () => {
    const result = validateAgentVisualPlan(validPlan, validContext);
    expect(result.ok).toBe(true);
  });

  it("rejects invalid blocks", () => {
    expect(codes({
      ...validPlan,
      actions: [{ ...validPlan.actions[0], block: "up.9.9" }],
    })).toContain("invalid_block");
  });

  it("rejects invalid link targets", () => {
    expect(codes({
      ...validPlan,
      actions: [{ ...validPlan.actions[0], link_target: "snare_top" }],
    })).toContain("invalid_link_target");
  });

  it("rejects invalid spatial masks", () => {
    expect(codes({
      ...validPlan,
      actions: [{ ...validPlan.actions[0], spatial_mask: Array(255).fill(1) }],
    })).toContain("invalid_spatial_mask");

    expect(codes({
      ...validPlan,
      actions: [{ ...validPlan.actions[0], spatial_mask: [2, ...Array(255).fill(1)] }],
    })).toContain("invalid_spatial_mask");
  });

  it("rejects invalid stage bounds", () => {
    expect(codes({
      ...validPlan,
      actions: [
        {
          ...validPlan.actions[0],
          stage_left: 10,
          stage_home: 0,
          stage_right: -10,
        },
      ],
    })).toContain("invalid_stage_bounds");
  });

  it("rejects invented feature ids", () => {
    expect(codes({
      ...validPlan,
      feature_candidates: [],
    })).toContain("invented_feature_id");
  });

  it("accepts feature ids from current config evidence", () => {
    const result = validateAgentVisualPlan(
      { ...validPlan, feature_candidates: [] },
      {
        ...validContext,
        currentFeatureIdsByBlock: { "up.0.0": 129 },
      },
    );

    expect(result.ok).toBe(true);
  });

  it("ignores stale timing metadata for durable live steering", () => {
    const result = validateAgentVisualPlan(
      {
        ...validPlan,
        based_on_audio_time: "rewound",
        based_on_wall_time_ms: "late",
        max_staleness_sec: "ignore me",
      },
      validContext,
    );

    expect(result.ok).toBe(true);
  });

  it("uses mode defaults for staleness", () => {
    expect(defaultMaxStalenessSec({ ...validContext, mode: "directive" })).toBe(20);
    expect(defaultMaxStalenessSec({ ...validContext, mode: "dj", djIntensity: "calm" })).toBe(12);
    expect(defaultMaxStalenessSec({ ...validContext, mode: "dj", djIntensity: "balanced" })).toBe(8);
    expect(defaultMaxStalenessSec({ ...validContext, mode: "dj", djIntensity: "active" })).toBe(4);
    expect(defaultMaxStalenessSec({ ...validContext, timingProfile: "section_timed" })).toBe(2);
  });

  it("allows idle staging but still rejects disarmed plans", () => {
    const idleResult = validateAgentVisualPlan(validPlan, {
      ...validContext,
      activeSession: false,
    });
    expect(idleResult.ok).toBe(true);

    const result = validateAgentVisualPlan(validPlan, {
      ...validContext,
      armed: false,
      activeSession: false,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.map((item) => item.code)).toEqual(expect.arrayContaining([
        "agent_disarmed",
      ]));
      expect(result.errors.map((item) => item.code)).not.toContain("inactive_session");
    }
  });

  it("allows a full first visual setup while playback is idle", () => {
    const result = validateAgentVisualPlan(
      {
        actions: [
          {
            action: "set_destination",
            space: "latent",
            slot: "a",
            destination_type: "seed",
            seed: 42,
          },
          {
            action: "set_destination",
            space: "latent",
            slot: "b",
            destination_type: "seed",
            seed: 4242,
          },
          {
            action: "set_destination",
            space: "prompt",
            slot: "a",
            destination_type: "prompt",
            prompt: "quiet watercolor road at dawn",
          },
          {
            action: "set_destination",
            space: "prompt",
            slot: "b",
            destination_type: "prompt",
            prompt: "bright market street with swirling paint",
          },
          {
            action: "set_destination_link",
            space: "prompt",
            link_target: "other_high",
          },
          {
            action: "set_reactive_config",
            space: "prompt",
            stage_left: 0,
            stage_home: 0.35,
            stage_right: 1,
            position_source: "brightness",
            intensity_source: "flux",
          },
          {
            action: "set_composition_config",
            distance: 0.8,
            mode: "auto",
          },
        ],
      },
      {
        ...validContext,
        activeSession: false,
        currentAudioTime: 0,
      },
    );

    expect(result.ok).toBe(true);
  });

  it("accepts freeze_blend action", () => {
    const result = validateAgentVisualPlan(
      {
        actions: [
          {
            action: "freeze_blend",
            space: "prompt",
            target_slot: "b",
          },
        ],
      },
      validContext,
    );

    expect(result.ok).toBe(true);
  });

  it("rejects freeze_blend without an active session", () => {
    const result = validateAgentVisualPlan(
      {
        actions: [
          {
            action: "freeze_blend",
            space: "prompt",
            target_slot: "b",
          },
        ],
      },
      {
        ...validContext,
        activeSession: false,
      },
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.map((item) => item.code)).toContain("inactive_session");
    }
  });

  it("rejects mismatched destination spaces and values", () => {
    expect(codes({
      actions: [{
        action: "set_destination",
        space: "latent",
        slot: "a",
        destination_type: "prompt",
        prompt: "wrong space",
      }],
    })).toContain("invalid_value");

    expect(codes({
      actions: [{
        action: "set_destination",
        space: "prompt",
        slot: "a",
        destination_type: "seed",
        seed: 42,
      }],
    })).toContain("invalid_value");

    expect(codes({
      actions: [{
        action: "set_destination",
        space: "prompt",
        slot: "a",
        destination_type: "prompt",
        prompt: "sunlit glass",
        seed: 42,
      }],
    })).toContain("invalid_value");
  });
});
