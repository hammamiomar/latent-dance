from __future__ import annotations

import pytest
from pydantic import ValidationError

from hambajuba2ba.integrations.hermes.contracts import (
    AgentApplyRequest,
    AgentEvent,
    DirectiveIntentIR,
    MusicWindowResponse,
    SongAnalysisResponse,
)


def test_agent_apply_request_accepts_valid_audio_reactive_plan() -> None:
    plan = AgentApplyRequest.model_validate(
        {
            "based_on_audio_time": 42.5,
            "based_on_wall_time_ms": 123456,
            "max_staleness_sec": 8,
            "actions": [
                {
                    "action": "update_block_config",
                    "block": "up.0.0",
                    "link_target": "drums_high",
                    "feature_id": 129,
                    "feature_label": "silver sparkle detail",
                    "enabled": True,
                    "sae_rank": 2,
                    "spatial_mode": "draw",
                    "spatial_mask": [1.0] * 256,
                    "stage_left": -20,
                    "stage_home": 0,
                    "stage_right": 25,
                    "intensity_source": "transient",
                    "intensity_curve": "linear",
                },
                {
                    "action": "set_destination_link",
                    "space": "prompt",
                    "link_target": "tension",
                },
            ],
            "feature_candidates": [
                {
                    "block": "up.0.0",
                    "id": 129,
                    "label": "silver sparkle detail",
                    "category": "object_detail",
                    "score": 12.0,
                }
            ],
        }
    )

    assert plan.actions[0].action == "update_block_config"
    assert plan.actions[0].sae_rank == 1
    assert plan.based_on_audio_time == 42.5


def test_agent_apply_request_normalizes_update_block_shorthand() -> None:
    plan = AgentApplyRequest.model_validate(
        {
            "actions": [
                {
                    "action": "update_block_config",
                    "block": "up.0.0",
                    "link_target": "drums_high",
                    "rank": 1,
                    "intensity_source": "sustain",
                    "intensity_curve": "gamma",
                    "gamma": 0.85,
                }
            ]
        }
    )

    action = plan.actions[0]
    assert action.action == "update_block_config"
    assert action.sae_rank == 1
    assert action.intensity_source == "energy_smooth"
    assert action.intensity_gamma == 0.85


def test_agent_apply_request_normalizes_reactive_config_aliases() -> None:
    plan = AgentApplyRequest.model_validate(
        {
            "actions": [
                {
                    "action": "set_reactive_config",
                    "space": "prompt",
                    "stage": {"left": -28, "home": 4, "right": 32},
                    "intensity_source": "brightness",
                    "silence_behavior": "return_home",
                    "blend_slew": 1.15,
                    "smoothing": 0.42,
                    "target": "other_percussive",
                    "rank_weights": {
                        "bass": 0.5,
                        "drums": None,
                        "other": 1,
                        "vocals": "disabled",
                    },
                }
            ]
        }
    )

    action = plan.actions[0]
    assert action.action == "set_reactive_config"
    assert action.position_source == "brightness"
    assert action.intensity_source == "energy_smooth"
    assert action.stage_left == -28
    assert action.stage_home == 4
    assert action.stage_right == 32
    assert action.silence_behavior == "drift_center"
    assert action.blend_slew_rate == 1.15
    assert action.position_smoothing_ms == 420
    assert action.rank_weights == {"bass": 0.5, "other": 1.0}
    assert "target" not in action.model_dump()


def test_agent_apply_request_normalizes_composite_action_aliases() -> None:
    plan = AgentApplyRequest.model_validate(
        {
            "actions": [
                {
                    "action": "set_prompt",
                    "prompt_a": "blue sun",
                    "prompt_b": "silver moon",
                },
                {
                    "action": "set_composition",
                    "seed_a": 123,
                    "seed_b": 456,
                    "distance": 4.8,
                    "mode": "pulse",
                },
                {
                    "action": "set_destination",
                    "destination_type": "prompt_b",
                    "prompt": "chrome greenhouse",
                },
            ]
        }
    )

    assert [action.action for action in plan.actions] == [
        "set_destination",
        "set_destination",
        "set_destination",
        "set_destination",
        "set_composition_config",
        "set_destination",
    ]
    assert plan.actions[0].space == "prompt"
    assert plan.actions[0].slot == "a"
    assert plan.actions[0].destination_type == "prompt"
    assert plan.actions[2].space == "latent"
    assert plan.actions[2].slot == "a"
    assert plan.actions[2].destination_type == "seed"
    assert plan.actions[4].distance == 4.0
    assert plan.actions[5].slot == "b"
    assert plan.actions[5].destination_type == "prompt"


def test_agent_apply_request_accepts_freeze_blend_action() -> None:
    plan = AgentApplyRequest.model_validate(
        {
            "actions": [
                {
                    "action": "freeze_blend",
                    "space": "prompt",
                    "target_slot": "a",
                }
            ]
        }
    )

    assert plan.actions[0].action == "freeze_blend"


@pytest.mark.parametrize(
    "action",
    [
        {
            "action": "set_destination",
            "space": "latent",
            "slot": "a",
            "destination_type": "prompt",
            "prompt": "wrong space",
        },
        {
            "action": "set_destination",
            "space": "prompt",
            "slot": "a",
            "destination_type": "seed",
            "seed": 42,
        },
        {
            "action": "set_destination",
            "space": "prompt",
            "slot": "a",
            "destination_type": "prompt",
            "prompt": "sunlit glass",
            "seed": 42,
        },
        {
            "action": "set_destination_mode",
            "space": "prompt",
            "mode": "linked",
        },
        {
            "action": "set_destination_link",
            "space": "latent",
            "link_target": "tension",
        },
        {
            "action": "freeze_blend",
            "space": "latent",
            "target_slot": "a",
        },
        {
            "action": "set_reactive_config",
            "space": "latent",
            "stage_left": 0,
            "stage_home": 0.5,
            "stage_right": 1,
        },
        {
            "action": "set_blend_position",
            "space": "latent",
            "position": 0.5,
        },
    ],
)
def test_agent_apply_request_rejects_invalid_destination_actions(
    action: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AgentApplyRequest.model_validate({"actions": [action]})


@pytest.mark.parametrize(
    "action",
    [
        {"action": "update_block_config", "block": "side.9", "feature_id": 1},
        {
            "action": "update_block_config",
            "block": "up.0.1",
            "link_target": "snare_top",
        },
        {"action": "update_block_config", "block": "up.0.1", "feature_id": 5120},
        {
            "action": "update_block_config",
            "block": "up.0.1",
            "spatial_mask": [1.0] * 255,
        },
        {
            "action": "update_block_config",
            "block": "up.0.1",
            "stage_left": 20,
            "stage_home": 0,
            "stage_right": -10,
        },
        {"action": "set_blend_position", "space": "prompt", "position": 1.5},
    ],
)
def test_agent_apply_request_rejects_invalid_visual_actions(
    action: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AgentApplyRequest.model_validate({"actions": [action]})


def test_directive_intent_ir_models_clause_language() -> None:
    intent = DirectiveIntentIR.model_validate(
        {
            "directive": "make hi-hats sparkle",
            "clauses": [
                {
                    "text": "hi-hats sparkle",
                    "kind": "effect",
                    "effect": "sparkle",
                    "target_blocks": ["up.0.0"],
                    "drivers": [
                        {
                            "link_target": "drums_high",
                            "intensity_source": "transient",
                            "aliases": ["hi-hats"],
                        }
                    ],
                    "timing": "persistent",
                    "strength": "medium",
                    "confidence": 0.9,
                }
            ],
        }
    )

    assert intent.clauses[0].drivers[0].link_target == "drums_high"


def test_agent_event_accepts_structured_tool_log() -> None:
    event = AgentEvent.model_validate(
        {
            "mode": "directive",
            "phase": "searching_features",
            "summary": "Searched up.0.1: sparkle",
            "tool": {
                "name": "hamba_search_features",
                "status": "completed",
                "arguments": {"block": "up.0.1", "query": "sparkle"},
                "result_summary": {"candidate_count": 3},
            },
            "feature_candidates": [
                {"block": "up.0.1", "id": 914, "label": "bright sparkles"}
            ],
        }
    )

    assert event.tool is not None
    assert event.tool.name == "hamba_search_features"
    assert event.tool.result_summary["candidate_count"] == 3


def test_music_window_response_requires_sampling_timestamps() -> None:
    response = MusicWindowResponse.model_validate(
        {
            "active_session": True,
            "current_time": 12.0,
            "sampled_at_audio_time": 12.0,
            "sampled_at_wall_time_ms": 987654,
            "is_playing": True,
            "lookback": 8,
            "lookahead": 16,
            "song_intelligence_available": True,
            "section": {"index": 1, "seconds_remaining": 3.5},
            "at_current_time": {"tension": 0.7, "coupling": {"bass-drums": 0.8}},
            "lookahead_context": {"next_section_in": 3.5},
            "aggregate_windows": {"tension": {"recent": {"mean": 0.4}}},
            "dominant_targets": {"current": [{"target": "bass", "energy_smooth": 0.8}]},
            "snapshots": [],
        }
    )

    assert response.sampled_at_audio_time == response.current_time
    assert response.song_intelligence_available is True
    assert response.at_current_time == {"tension": 0.7, "coupling": {"bass-drums": 0.8}}
    assert response.dominant_targets == {"current": [{"target": "bass", "energy_smooth": 0.8}]}

    with pytest.raises(ValidationError):
        MusicWindowResponse.model_validate(
            {
                "active_session": True,
                "current_time": 12.0,
                "is_playing": True,
                "lookback": 8,
                "lookahead": 16,
                "snapshots": [],
            }
        )


def test_song_analysis_response_allows_metadata_analysis_payload() -> None:
    response = SongAnalysisResponse.model_validate(
        {
            "available": True,
            "audio_id": "track-1",
            "sampled_at_wall_time_ms": 123,
            "analysis": {
                "version": "hamba-song-analysis/v1",
                "anonymous": False,
                "metadata": {"filename": "Robert_Miles_-_Children.wav"},
                "metadata_policy": "filename included when available",
                "target_count": 20,
            },
        }
    )

    assert response.available is True
    assert response.analysis is not None
    assert response.analysis["anonymous"] is False
    assert response.analysis["metadata"]["filename"] == "Robert_Miles_-_Children.wav"
