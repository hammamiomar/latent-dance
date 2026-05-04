import { describe, expect, it } from "vitest";
import {
  AUTO_DANCE_MIN_GAP_MS,
  AUTO_DANCE_STABLE_INTERVAL_MS,
  buildAutoDanceDirective,
  shouldTriggerAutoDance,
  type AutoDanceLastCheckpoint,
} from "./autoDance";
import type { AgentStateResponse } from "../types/agent";

function state(overrides: Partial<AgentStateResponse> = {}): AgentStateResponse {
  return {
    armed: true,
    mode: "directive",
    active_session: true,
    entry_context: {
      situation: "visualizer_playing",
      summary: "playing",
      recommended_next_step: "read music window",
      audio: {
        audio_id_present: true,
        upload_phase: "ready",
        duration: 120,
        current_time: 40,
        is_playing: true,
      },
      generation: {
        status: "connected",
        is_generating: true,
      },
      song_intelligence: {
        profile_available: true,
        analysis_available: true,
      },
      control: {
        enabled_block_count: 4,
        prompt_empty: false,
        latent_empty: false,
        fresh_blank_setup: false,
        composition: {},
      },
    },
    block_configs: {},
    destinations: {},
    song_profile: { sections: [0, 48, 80] },
    song_analysis_available: true,
    latest_event: null,
    ...overrides,
  };
}

function last(overrides: Partial<AutoDanceLastCheckpoint> = {}): AutoDanceLastCheckpoint {
  return {
    wallTimeMs: 0,
    audioTime: 0,
    sectionIndex: 0,
    ...overrides,
  };
}

describe("autoDance scheduler", () => {
  it("does not fire when ineligible", () => {
    const decision = shouldTriggerAutoDance({
      enabled: true,
      bridgeConnected: true,
      frontendConnected: true,
      submitting: true,
      state: state(),
      nowWallTimeMs: AUTO_DANCE_STABLE_INTERVAL_MS + 1,
      last: last(),
    });

    expect(decision.shouldTrigger).toBe(false);
  });

  it("fires near a new section boundary once the minimum gap has passed", () => {
    const decision = shouldTriggerAutoDance({
      enabled: true,
      bridgeConnected: true,
      frontendConnected: true,
      submitting: false,
      state: state(),
      nowWallTimeMs: AUTO_DANCE_MIN_GAP_MS + 1,
      last: last({ sectionIndex: null }),
    });

    expect(decision.shouldTrigger).toBe(true);
    expect(decision.reason).toBe("section_near");
  });

  it("fires after 15s of stable eligible playback", () => {
    const decision = shouldTriggerAutoDance({
      enabled: true,
      bridgeConnected: true,
      frontendConnected: true,
      submitting: false,
      state: state({
        entry_context: {
          ...state().entry_context!,
          audio: {
            ...state().entry_context!.audio,
            current_time: 20,
          },
        },
      }),
      nowWallTimeMs: AUTO_DANCE_STABLE_INTERVAL_MS + 1,
      last: last({ sectionIndex: 0 }),
    });

    expect(decision.shouldTrigger).toBe(true);
    expect(decision.reason).toBe("stable_interval");
  });

  it("builds directives that require durable song intelligence and a 45s window", () => {
    const directive = buildAutoDanceDirective({
      shouldTrigger: true,
      reason: "stable_interval",
      audioTime: 30,
      sectionIndex: 0,
      secondsToNextSection: 18,
    });

    expect(directive).toContain("hamba_get_song_analysis");
    expect(directive).toContain("hamba_get_music_window with lookahead=45");
    expect(directive).toContain("hamba_get_feature_palette");
    expect(directive).toContain("hamba_prepare_feature_palette");
    expect(directive).toContain("Use prepared palette candidates first");
    expect(directive).toContain("at least one visible evolution");
    expect(directive).toContain("1-3 coordinated visible mutations");
    expect(directive).toContain("stable groove means invent a new visual chapter");
    expect(directive).toContain("about every 15 seconds");
    expect(directive).toContain("never send `set_prompt`");
    expect(directive).toContain("Do not put `target` inside `set_reactive_config`");
    expect(directive).toContain("Use `sae_rank: 1`");
    expect(directive).toContain("Do not send `based_on_audio_time`");
  });

  it("carries recent human steering into autonomous checkpoints", () => {
    const directive = buildAutoDanceDirective({
      shouldTrigger: true,
      reason: "stable_interval",
      audioTime: 30,
      sectionIndex: 0,
      secondsToNextSection: 18,
    }, {
      recentHumanDirective: "make that bass darker",
      recentHumanDirectiveAgeMs: 12_400,
    });

    expect(directive).toContain("Recent human steer still active (12s ago)");
    expect(directive).toContain("make that bass darker");
    expect(directive).toContain("must not undo it");
  });
});
