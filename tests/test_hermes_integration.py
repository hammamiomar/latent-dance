from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from hambajuba2ba.integrations.hermes.client import HambaBridgeClient
from hambajuba2ba.integrations.hermes.contracts import FeatureEntry
from hambajuba2ba.integrations.hermes.embeddings import (
    EmbeddingIndex,
    safe_block_key,
    source_hash,
)
from hambajuba2ba.integrations.hermes.features import FeatureCatalog
from hambajuba2ba.integrations.hermes.query_embedder import create_default_query_embedder
from hambajuba2ba.integrations.hermes import mcp_server


def test_local_feature_catalog_search_returns_valid_block_ids():
    result = FeatureCatalog.load().search("up.0.1", "spotted animal pattern", limit=5)

    assert result.block == "up.0.1"
    assert result.candidates
    assert all(0 <= candidate.id < 5120 for candidate in result.candidates)


def test_feature_catalog_browse_and_search_are_deterministic_with_seed():
    catalog = _tiny_feature_catalog()

    browse_a = catalog.browse(
        "up.0.1",
        category="pattern",
        sample_count=3,
        seed="session-42/directive-3",
        temperature=0.9,
    )
    browse_b = catalog.browse(
        "up.0.1",
        category="pattern",
        sample_count=3,
        seed="session-42/directive-3",
        temperature=0.9,
    )

    assert [sample["id"] for sample in browse_a["samples"]] == [
        sample["id"] for sample in browse_b["samples"]
    ]

    search_a = catalog.search_details(
        "up.0.1",
        "sparkle glitter",
        category="pattern",
        limit=3,
        seed="session-42/directive-3",
        temperature=0.4,
    )
    search_b = catalog.search_details(
        "up.0.1",
        "sparkle glitter",
        category="pattern",
        limit=3,
        seed="session-42/directive-3",
        temperature=0.4,
    )

    assert [candidate["id"] for candidate in search_a["candidates"]] == [
        candidate["id"] for candidate in search_b["candidates"]
    ]


def test_feature_catalog_browse_returns_category_counts():
    result = _tiny_feature_catalog().browse(
        "up.0.1",
        sample_count=2,
        seed="counts",
    )

    assert result["block"] == "up.0.1"
    assert result["total_features"] == 6
    assert result["categories"]["pattern"]["count"] == 4
    assert result["categories"]["lighting"]["count"] == 1


def test_feature_search_uses_lexical_fallback_without_embeddings():
    result = _tiny_feature_catalog().search_details(
        "up.0.1",
        "sparkle glitter",
        category="pattern",
        limit=2,
        seed="lexical",
        temperature=0.0,
        semantic=True,
    )

    assert result["retrieval"]["lexical"] is True
    assert result["retrieval"]["semantic"] is False
    assert result["retrieval"]["fallback"] == "embedding artifact absent"
    assert result["candidates"][0]["label"] == "glittering sparkles and highlights"
    assert result["candidates"][0]["scores"]["lexical"] > 0


def test_feature_search_ignores_filler_words_in_directives():
    catalog = FeatureCatalog(
        {
            "up.0.1": (
                _feature_entry(1, "glittering sparkles and highlights", "pattern", 20.0),
                _feature_entry(2, "printed text on paper", "pattern", 45.0),
            )
        }
    )

    result = catalog.search_details(
        "up.0.1",
        "sparkles on the hi hats",
        category="pattern",
        limit=2,
        seed="stopwords",
        temperature=0.0,
        semantic=False,
    )

    assert result["candidates"][0]["id"] == 1


def test_feature_search_avoids_duplicate_labels_and_requested_feature_ids():
    catalog = _tiny_feature_catalog()

    result = catalog.search_details(
        "up.0.1",
        "sparkle glitter",
        category="pattern",
        limit=5,
        seed="duplicates",
        temperature=0.0,
    )
    labels = [candidate["label"] for candidate in result["candidates"]]

    assert len(labels) == len(set(labels))

    avoided_id = result["candidates"][0]["id"]
    avoided = catalog.search_details(
        "up.0.1",
        "sparkle glitter",
        category="pattern",
        limit=5,
        seed="duplicates",
        temperature=0.0,
        avoid_feature_ids=[avoided_id],
    )

    assert avoided_id not in [candidate["id"] for candidate in avoided["candidates"]]


def test_embedding_index_absent_returns_lexical_fallback(tmp_path):
    assert EmbeddingIndex.load_optional(tmp_path / "missing.npz") is None

    class DummyEmbedder:
        def encode(self, text: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    result = _tiny_feature_catalog(query_embedder=DummyEmbedder()).search_details(
        "up.0.1",
        "sparkle glitter",
        category="pattern",
        limit=1,
        seed="absent",
        temperature=0.0,
        semantic=True,
    )

    assert result["retrieval"]["semantic"] is False
    assert result["retrieval"]["fallback"] == "embedding artifact absent"


def test_embedding_index_optional_loader_rejects_unverified_artifacts(tmp_path):
    block = "up.0.1"
    safe = safe_block_key(block)
    catalog_json = tmp_path / f"{block}.json"
    catalog_json.write_text("[]")
    npz_path = tmp_path / "feature_embeddings.npz"
    np.savez(
        npz_path,
        **{
            f"{safe}_ids": np.array([1], dtype=np.int32),
            f"{safe}_vectors": np.array([[1.0, 0.0]], dtype=np.float16),
        },
    )

    assert EmbeddingIndex.load_optional(npz_path, catalog_dir=tmp_path) is None


def test_embedding_index_present_can_supply_semantic_candidates(tmp_path):
    block = "up.0.1"
    safe = safe_block_key(block)
    entries = _tiny_entries()
    catalog_json = tmp_path / f"{block}.json"
    catalog_json.write_text(
        json.dumps([entry.model_dump(by_alias=True) for entry in entries])
    )
    manifest_path = tmp_path / "feature_embeddings.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": "dummy-embedding-model",
                "dimension": 2,
                "source_hashes": {block: source_hash(catalog_json)},
            }
        )
    )
    npz_path = tmp_path / "feature_embeddings.npz"
    np.savez(
        npz_path,
        **{
            f"{safe}_ids": np.array([1, 4], dtype=np.int32),
            f"{safe}_vectors": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        },
    )

    class DummyEmbedder:
        def encode(self, text: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    embedding_index = EmbeddingIndex.load(
        npz_path,
        manifest_path=manifest_path,
        catalog_dir=tmp_path,
    )
    catalog = FeatureCatalog(
        {block: entries},
        embedding_index=embedding_index,
        query_embedder=DummyEmbedder(),
    )
    result = catalog.search_details(
        block,
        "semantic only phrase",
        limit=1,
        seed="semantic",
        temperature=0.0,
        semantic=True,
    )

    assert result["retrieval"]["semantic"] is True
    assert result["retrieval"]["embedding_model"] == "dummy-embedding-model"
    assert result["candidates"][0]["id"] == 1
    assert result["candidates"][0]["scores"]["semantic"] == 1.0


def test_feature_catalog_load_wires_default_query_embedder_for_artifacts(
    tmp_path,
    monkeypatch,
):
    block = "up.0.1"
    safe = safe_block_key(block)
    entries = _tiny_entries()
    for catalog_block in ("down.2.1", "mid.0", "up.0.0", "up.0.1"):
        block_entries = entries if catalog_block == block else ()
        (tmp_path / f"{catalog_block}.json").write_text(
            json.dumps([entry.model_dump(by_alias=True) for entry in block_entries])
        )

    manifest_path = tmp_path / "feature_embeddings.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": "dummy-embedding-model",
                "dimension": 2,
                "source_hashes": {block: source_hash(tmp_path / f"{block}.json")},
            }
        )
    )
    np.savez(
        tmp_path / "feature_embeddings.npz",
        **{
            f"{safe}_ids": np.array([1, 4], dtype=np.int32),
            f"{safe}_vectors": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        },
    )

    class DummyEmbedder:
        def encode(self, text: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "hambajuba2ba.integrations.hermes.features.create_default_query_embedder",
        lambda model_name=None: DummyEmbedder(),
    )

    catalog = FeatureCatalog.load(catalog_dir=tmp_path)
    result = catalog.search_details(
        block,
        "semantic only phrase",
        limit=1,
        seed="default-embedder",
        temperature=0.0,
    )

    assert result["retrieval"]["semantic"] is True
    assert result["retrieval"]["embedding_model"] == "dummy-embedding-model"
    assert result["candidates"][0]["id"] == 1


def test_default_query_embedder_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HAMBA_FEATURE_SEMANTIC", "0")

    assert create_default_query_embedder() is None


def test_feature_embedding_cli_has_package_entrypoint():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert (
        pyproject["project"]["scripts"]["hambajuba-feature-embeddings"]
        == "hambajuba2ba.integrations.hermes.feature_embedding_cli:main"
    )
    assert "sentence-transformers>=3.0" in pyproject["project"][
        "optional-dependencies"
    ]["hermes-semantic"]


def test_mcp_feature_tools_expose_browse_and_rich_search():
    if mcp_server.FastMCP is None:
        pytest.skip("mcp package is not installed")

    asyncio.run(_assert_mcp_feature_tools_expose_browse_and_rich_search())


async def _assert_mcp_feature_tools_expose_browse_and_rich_search():
    server = mcp_server.create_mcp_server(
        client=object(),
        feature_catalog=_tiny_feature_catalog(),
    )

    tools = await server.list_tools()
    tool_names = {tool.name for tool in tools}
    assert "hamba_browse_catalog" in tool_names
    assert "hamba_prepare_feature_palette" in tool_names
    assert "hamba_get_feature_palette" in tool_names

    _, browse = await server.call_tool(
        "hamba_browse_catalog",
        {
            "block": "up.0.1",
            "category": "pattern",
            "sample_count": 2,
            "seed": "mcp",
        },
    )
    assert browse["samples"]

    _, search = await server.call_tool(
        "hamba_search_features",
        {
            "block": "up.0.1",
            "query": "sparkle glitter",
            "category": "pattern",
            "limit": 2,
            "seed": "mcp",
            "semantic": False,
        },
    )
    assert search["retrieval"]["fallback"] == "semantic disabled"
    assert "scores" in search["candidates"][0]


def test_mcp_feature_tools_emit_agent_event_log_items():
    if mcp_server.FastMCP is None:
        pytest.skip("mcp package is not installed")

    asyncio.run(_assert_mcp_feature_tools_emit_agent_event_log_items())


async def _assert_mcp_feature_tools_emit_agent_event_log_items():
    class RecordingClient:
        def __init__(self) -> None:
            self.events = []

        async def report_phase(self, event):
            self.events.append(event)
            return {"accepted": True}

    client = RecordingClient()
    server = mcp_server.create_mcp_server(
        client=client,
        feature_catalog=_tiny_feature_catalog(),
    )

    _, search = await server.call_tool(
        "hamba_search_features",
        {
            "block": "up.0.1",
            "query": "sparkle glitter",
            "category": "pattern",
            "limit": 2,
            "seed": "events",
            "semantic": False,
        },
    )

    assert search["candidates"]
    event = client.events[-1]
    assert event["phase"] == "searching_features"
    assert event["tool"]["name"] == "hamba_search_features"
    assert event["tool"]["status"] == "completed"
    assert event["tool"]["arguments"]["query"] == "sparkle glitter"
    assert event["tool"]["result_summary"]["candidate_count"] == len(search["candidates"])
    assert event["feature_candidates"][0]["id"] == search["candidates"][0]["id"]


def test_mcp_blocks_fresh_blank_feature_search_until_song_analysis():
    if mcp_server.FastMCP is None:
        pytest.skip("mcp package is not installed")

    asyncio.run(_assert_mcp_blocks_fresh_blank_feature_search_until_song_analysis())


async def _assert_mcp_blocks_fresh_blank_feature_search_until_song_analysis():
    class FreshBlankClient:
        def __init__(self) -> None:
            self.events = []

        async def get_state(self):
            return {
                "armed": True,
                "mode": "directive",
                "active_session": False,
                "song_profile": {"bpm": 125},
                "song_analysis_available": True,
                "entry_context": {
                    "situation": "song_loaded_idle",
                    "control": {"fresh_blank_setup": True},
                },
                "control_state": {
                    "summary": {
                        "enabled_block_count": 0,
                        "prompt": {
                            "destination_a": None,
                            "destination_b": None,
                        },
                        "composition": {
                            "seed_a": None,
                            "seed_b": None,
                        },
                    }
                },
            }

        async def get_song_analysis(self):
            return {
                "available": True,
                "analysis": {
                    "target_count": 2,
                    "metadata_policy": "filename included when available",
                    "metadata": {"filename": "Test_Track.wav"},
                    "ranked_drivers": {
                        "primary_driver": [
                            {"target": "tension", "score": 0.9, "reasons": ["wide_swing"]}
                        ]
                    },
                },
            }

        async def report_phase(self, event):
            self.events.append(event)
            return {"accepted": True}

    client = FreshBlankClient()
    server = mcp_server.create_mcp_server(
        client=client,
        feature_catalog=_tiny_feature_catalog(),
    )

    await server.call_tool("hamba_get_state", {})

    with pytest.raises(Exception, match="Call hamba_get_song_analysis"):
        await server.call_tool(
            "hamba_search_features",
            {
                "block": "up.0.1",
                "query": "sparkle glitter",
                "category": "pattern",
                "limit": 2,
                "semantic": False,
            },
        )

    failed_event = client.events[-1]
    assert failed_event["tool"]["name"] == "hamba_search_features"
    assert failed_event["tool"]["status"] == "failed"
    assert "whole-song analysis" in failed_event["error"]

    await server.call_tool("hamba_get_song_analysis", {})
    await server.call_tool("hamba_get_state", {})
    _, search = await server.call_tool(
        "hamba_search_features",
        {
            "block": "up.0.1",
            "query": "sparkle glitter",
            "category": "pattern",
            "limit": 2,
            "semantic": False,
        },
    )

    assert search["candidates"]


def test_mcp_feature_lookup_budget_returns_remembered_candidates_for_apply():
    if mcp_server.FastMCP is None:
        pytest.skip("mcp package is not installed")

    asyncio.run(_assert_mcp_feature_lookup_budget_returns_remembered_candidates_for_apply())


async def _assert_mcp_feature_lookup_budget_returns_remembered_candidates_for_apply():
    class RecordingClient:
        def __init__(self) -> None:
            self.applies = []
            self.events = []

        async def report_phase(self, event):
            self.events.append(event)
            return {"accepted": True}

        async def apply_visual_plan(self, **payload):
            self.applies.append(payload)
            return {"accepted": True}

    client = RecordingClient()
    server = mcp_server.create_mcp_server(
        client=client,
        feature_catalog=_tiny_feature_catalog(),
    )

    budget_result = None
    for index in range(9):
        _, result = await server.call_tool(
            "hamba_search_features",
            {
                "block": "up.0.1",
                "query": "sparkle glitter",
                "category": "pattern",
                "limit": 2,
                "seed": f"budget-{index}",
                "semantic": False,
            },
        )
        budget_result = result

    assert budget_result is not None
    assert budget_result["search_budget"]["exhausted"] is True
    assert budget_result["remembered_feature_candidates"]
    assert "hamba_apply_visual_plan" in budget_result["search_budget"]["required_next_step"]

    await server.call_tool(
        "hamba_apply_visual_plan",
        {
            "actions": [
                {
                    "action": "update_block_config",
                    "block": "up.0.1",
                    "feature_id": budget_result["remembered_feature_candidates"][0]["id"],
                    "enabled": True,
                    "strength_max": 18,
                }
            ],
        },
    )

    assert client.applies
    assert client.applies[-1]["feature_candidates"]
    assert client.applies[-1]["feature_candidates"][0]["block"] == "up.0.1"


def test_mcp_feature_palette_survives_apply_and_marks_used_candidates():
    if mcp_server.FastMCP is None:
        pytest.skip("mcp package is not installed")

    asyncio.run(_assert_mcp_feature_palette_survives_apply_and_marks_used_candidates())


async def _assert_mcp_feature_palette_survives_apply_and_marks_used_candidates():
    class RecordingClient:
        def __init__(self) -> None:
            self.applies = []
            self.events = []

        async def report_phase(self, event):
            self.events.append(event)
            return {"accepted": True}

        async def apply_visual_plan(self, **payload):
            self.applies.append(payload)
            return {"accepted": True}

    client = RecordingClient()
    server = mcp_server.create_mcp_server(
        client=client,
        feature_catalog=_tiny_all_block_feature_catalog(),
    )

    _, prepared = await server.call_tool(
        "hamba_prepare_feature_palette",
        {
            "theme": "sparkle turtle moon",
            "divergence": 1.0,
            "per_block": 8,
        },
    )
    assert prepared["available"] is True
    assert prepared["epoch"] == 1
    assert prepared["total_candidate_count"] > 0
    assert prepared["bucket_mix"]["wildcard"] > 0

    _, palette = await server.call_tool(
        "hamba_get_feature_palette",
        {"limit_per_block": 4, "prefer_unused": True},
    )
    candidate = palette["palette"]["up.0.1"][0]

    await server.call_tool(
        "hamba_apply_visual_plan",
        {
            "actions": [
                {
                    "action": "update_block_config",
                    "block": "up.0.1",
                    "feature_id": candidate["id"],
                    "enabled": True,
                    "strength_max": 18,
                }
            ],
        },
    )

    _, after = await server.call_tool(
        "hamba_get_feature_palette",
        {"limit_per_block": 4, "prefer_unused": True},
    )

    assert client.applies
    assert candidate["id"] in [
        item["id"] for item in client.applies[-1]["feature_candidates"]
    ]
    assert after["available"] is True
    assert after["epoch"] == 1
    assert after["used_candidate_count"] == 1
    assert after["total_candidate_count"] == prepared["total_candidate_count"]


def test_hermes_client_rejects_remote_control_hosts():
    with pytest.raises(ValueError, match="local frontend bridge"):
        HambaBridgeClient(base_url="https://example.test")


def test_hermes_client_rejects_backend_agent_paths():
    client = HambaBridgeClient(base_url="http://127.0.0.1:14321")

    async def assert_rejected() -> None:
        with pytest.raises(ValueError, match="non-Hamba bridge"):
            await client._request("/api/agent/state")

    asyncio.run(assert_rejected())


def test_hermes_client_always_uses_mcp_bridge_role():
    client = HambaBridgeClient(base_url="ws://127.0.0.1:14321/agent/ws?role=frontend")

    assert client._websocket_url() == "ws://127.0.0.1:14321/agent/ws?role=mcp"


def test_hermes_client_round_trips_over_local_websocket_bridge():
    asyncio.run(_assert_hermes_client_round_trips_over_local_websocket_bridge())


async def _assert_hermes_client_round_trips_over_local_websocket_bridge():
    try:
        from websockets.asyncio.server import serve
    except ImportError:  # pragma: no cover - compatibility with older websockets
        from websockets.server import serve

    received = []

    async def handler(websocket):
        async for raw in websocket:
            message = json.loads(raw)
            received.append(message)
            await websocket.send(
                json.dumps(
                    {
                        "id": message["id"],
                        "type": "result",
                        "payload": {"ok": True},
                    }
                )
            )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = HambaBridgeClient(base_url=f"ws://127.0.0.1:{port}/agent/ws")
        result = await client.get_state()
        await client.close()

    assert result == {"ok": True}
    assert received[0]["type"] == "agent.get_state"


def test_hermes_client_omits_live_plan_timing_metadata():
    asyncio.run(_assert_hermes_client_omits_live_plan_timing_metadata())


async def _assert_hermes_client_omits_live_plan_timing_metadata():
    try:
        from websockets.asyncio.server import serve
    except ImportError:  # pragma: no cover - compatibility with older websockets
        from websockets.server import serve

    received = []

    async def handler(websocket):
        async for raw in websocket:
            message = json.loads(raw)
            received.append(message)
            await websocket.send(
                json.dumps(
                    {
                        "id": message["id"],
                        "type": "result",
                        "payload": {"accepted": True},
                    }
                )
            )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = HambaBridgeClient(base_url=f"ws://127.0.0.1:{port}/agent/ws")
        result = await client.apply_visual_plan(
            actions=[
                {
                    "action": "clear_destination",
                    "space": "prompt",
                    "slot": "a",
                }
            ],
            based_on_audio_time=12.5,
            based_on_wall_time_ms=123456,
            max_staleness_sec=4,
        )
        await client.close()

    assert result == {"accepted": True}
    assert received[0]["type"] == "agent.apply_visual_plan"
    assert "based_on_audio_time" not in received[0]["payload"]
    assert "based_on_wall_time_ms" not in received[0]["payload"]
    assert "max_staleness_sec" not in received[0]["payload"]


def test_mcp_music_window_and_visual_plan_round_trip_through_frontend_bridge():
    if mcp_server.FastMCP is None:
        pytest.skip("mcp package is not installed")

    asyncio.run(_assert_mcp_music_window_and_visual_plan_round_trip())


def test_mcp_visual_action_normalization_accepts_update_block_shorthand_and_clamps_stage():
    actions = mcp_server._normalize_visual_actions(
        [
            {
                "action": "update_block_config",
                "block": "up.0.0",
                "rank": 3,
                "gamma": 0.85,
                "intensity_source": "sustain",
                "strength_min": -64,
                "strength_max": 72,
            },
            {
                "action": "set_reactive_config",
                "space": "prompt",
                "stage": {"left": -58, "home": 0, "right": 58},
                "silence_behavior": "return_home",
                "blend_slew": 1.15,
                "smoothing": 0.42,
                "intensity_source": "brightness",
                "target": "other_percussive",
                "rank_weights": {
                    "bass": 0.5,
                    "drums": None,
                    "other": 1,
                    "vocals": "disabled",
                },
            },
            {
                "action": "set_prompt_destinations",
                "prompt_a": "sea sky",
                "prompt_b": "chrome garden",
            },
            {
                "action": "set_prompt",
                "prompt_a": "violet animals",
                "prompt_b": "hinge garden",
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
                "prompt": "blackwater moon",
            }
        ]
    )

    assert actions[:2] == [
        {
            "action": "update_block_config",
            "block": "up.0.0",
            "sae_rank": 1,
            "intensity_gamma": 0.85,
            "intensity_source": "energy_smooth",
            "strength_min": -50.0,
            "strength_max": 50.0,
        },
        {
            "action": "set_reactive_config",
            "space": "prompt",
            "stage_left": -50.0,
            "stage_home": 0.0,
            "stage_right": 50.0,
            "silence_behavior": "drift_center",
            "blend_slew_rate": 1.15,
            "position_smoothing_ms": 420.0,
            "position_source": "brightness",
            "intensity_source": "energy_smooth",
            "rank_weights": {"bass": 0.5, "other": 1.0},
        }
    ]
    assert actions[2:] == [
        {
            "action": "set_destination",
            "space": "prompt",
            "slot": "a",
            "destination_type": "prompt",
            "prompt": "sea sky",
        },
        {
            "action": "set_destination",
            "space": "prompt",
            "slot": "b",
            "destination_type": "prompt",
            "prompt": "chrome garden",
        },
        {
            "action": "set_destination",
            "space": "prompt",
            "slot": "a",
            "destination_type": "prompt",
            "prompt": "violet animals",
        },
        {
            "action": "set_destination",
            "space": "prompt",
            "slot": "b",
            "destination_type": "prompt",
            "prompt": "hinge garden",
        },
        {
            "action": "set_destination",
            "space": "latent",
            "slot": "a",
            "destination_type": "seed",
            "seed": 123,
        },
        {
            "action": "set_destination",
            "space": "latent",
            "slot": "b",
            "destination_type": "seed",
            "seed": 456,
        },
        {
            "action": "set_composition_config",
            "distance": 4.0,
            "mode": "pulse",
        },
        {
            "action": "set_destination",
            "destination_type": "prompt",
            "prompt": "blackwater moon",
            "slot": "b",
            "space": "prompt",
        },
    ]


async def _assert_mcp_music_window_and_visual_plan_round_trip():
    try:
        from websockets.asyncio.server import serve
    except ImportError:  # pragma: no cover - compatibility with older websockets
        from websockets.server import serve

    received = []

    async def handler(websocket):
        async for raw in websocket:
            message = json.loads(raw)
            received.append(message)
            if message["type"] == "agent.get_music_window":
                payload = {
                    "active_session": True,
                    "sampled_at_audio_time": 42.0,
                    "sampled_at_wall_time_ms": 123456,
                    "at_current_time": {"tension": 0.7},
                }
            elif message["type"] == "agent.get_song_analysis":
                payload = {
                    "available": True,
                    "audio_id": "track-1",
                    "analysis": {
                        "version": "hamba-song-analysis/v1",
                        "anonymous": False,
                        "metadata": {"filename": "storm_test.wav"},
                        "metadata_policy": "filename included when available",
                        "target_count": 2,
                        "ranked_drivers": {
                            "primary_driver": [
                                {"target": "bass", "score": 0.8, "reasons": ["wide_swing"]}
                            ]
                        },
                    },
                }
            elif message["type"] == "agent.apply_visual_plan":
                payload = {
                    "accepted": True,
                    "changes": [
                        {
                            "action": "set_destination",
                            "target": "prompt:b",
                            "after": message["payload"]["actions"][0],
                        }
                    ],
                }
            else:
                payload = {"ok": True}

            await websocket.send(
                json.dumps(
                    {
                        "id": message["id"],
                        "type": "result",
                        "payload": payload,
                    }
                )
            )

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = HambaBridgeClient(base_url=f"ws://127.0.0.1:{port}/agent/ws")
        mcp = mcp_server.create_mcp_server(
            client=client,
            feature_catalog=_tiny_feature_catalog(),
        )
        _, song_analysis = await mcp.call_tool("hamba_get_song_analysis", {})
        _, music_window = await mcp.call_tool(
            "hamba_get_music_window",
            {"lookback": 4, "lookahead": 12},
        )
        _, apply_result = await mcp.call_tool(
            "hamba_apply_visual_plan",
            {
                "actions": [
                    {
                        "type": "set_destination",
                        "space": "prompt",
                        "slot": "b",
                        "destination_type": "prompt",
                        "prompt": "storm of glitter",
                    }
                ],
                "based_on_audio_time": "song looped",
                "based_on_wall_time_ms": "late",
                "max_staleness_sec": "ignore",
                "transcript": "make the hats sparkle",
                "reason": "hi-hat shimmer test",
                "feature_candidates": [
                    {
                        "block": "up.0.1",
                        "id": 1,
                        "label": "glittering sparkles and highlights",
                    }
                ],
            },
        )
        await client.close()

    assert song_analysis["available"] is True
    assert song_analysis["analysis"]["anonymous"] is False
    assert song_analysis["analysis"]["metadata"]["filename"] == "storm_test.wav"
    assert music_window["active_session"] is True
    assert music_window["at_current_time"]["tension"] == 0.7
    assert apply_result["accepted"] is True
    received_types = [message["type"] for message in received]
    assert received_types[0] == "agent.get_song_analysis"
    assert "agent.get_music_window" in received_types
    assert received_types[-1] == "agent.apply_visual_plan"
    assert received_types.count("agent.report_phase") == 3
    music_window_request = next(
        message for message in received if message["type"] == "agent.get_music_window"
    )
    assert music_window_request["payload"] == {"lookback": 4, "lookahead": 12}
    report_payloads = [
        message["payload"]
        for message in received
        if message["type"] == "agent.report_phase"
    ]
    assert report_payloads[0]["tool"]["name"] == "hamba_get_song_analysis"
    assert report_payloads[1]["tool"]["name"] == "hamba_get_music_window"
    assert report_payloads[2]["tool"]["name"] == "hamba_apply_visual_plan"

    apply_payload = received[-1]["payload"]
    assert "based_on_audio_time" not in apply_payload
    assert "based_on_wall_time_ms" not in apply_payload
    assert "max_staleness_sec" not in apply_payload
    assert apply_payload["transcript"] == "make the hats sparkle"
    assert apply_payload["reason"] == "hi-hat shimmer test"
    assert apply_payload["actions"][0]["action"] == "set_destination"
    assert "type" not in apply_payload["actions"][0]
    assert apply_payload["feature_candidates"][0]["id"] == 1
    assert apply_payload["actions"][0]["prompt"] == "storm of glitter"


def test_gpu_backend_has_no_hermes_agent_router():
    main_source = Path("app/main.py").read_text()

    assert "app.include_router(agent.router)" not in main_source
    assert "agent_hub" not in main_source
    assert not Path("app/routers/agent.py").exists()


def _tiny_feature_catalog(**kwargs) -> FeatureCatalog:
    return FeatureCatalog({"up.0.1": _tiny_entries()}, **kwargs)


def _tiny_all_block_feature_catalog(**kwargs) -> FeatureCatalog:
    return FeatureCatalog(
        {
            "down.2.1": _tiny_entries(),
            "mid.0": _tiny_entries(),
            "up.0.0": _tiny_entries(),
            "up.0.1": _tiny_entries(),
        },
        **kwargs,
    )


def _tiny_entries() -> tuple[FeatureEntry, ...]:
    return (
        _feature_entry(1, "glittering sparkles and highlights", "pattern", 35.0),
        _feature_entry(2, "glittering sparkles and highlights", "pattern", 34.0),
        _feature_entry(3, "dark shadow dramatic lighting", "lighting", 30.0),
        _feature_entry(4, "turtle shell reptile texture", "texture", 28.0),
        _feature_entry(5, "striped giraffe animal print", "pattern", 32.0),
        _feature_entry(6, "soft floral dress pattern", "pattern", 22.0),
    )


def _feature_entry(
    feature_id: int,
    label: str,
    category: str,
    mean_activation: float,
) -> FeatureEntry:
    return FeatureEntry(
        id=feature_id,
        label=label,
        category=category,
        confidence="high",
        mean_activation=mean_activation,
    )
