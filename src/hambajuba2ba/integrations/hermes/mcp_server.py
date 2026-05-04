"""Hermes MCP entrypoint for local Hamba visualizer control."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from hambajuba2ba.integrations.hermes.client import HambaBridgeClient, client_from_env
from hambajuba2ba.integrations.hermes.contracts import STAGE_MAX, STAGE_MIN
from hambajuba2ba.integrations.hermes.features import (
    BLOCKS as FEATURE_BLOCKS,
    FeatureCatalog,
    get_default_catalog,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional integration dependency
    FastMCP = None  # type: ignore[assignment]


def _compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for candidate in candidates[:12]:
        item = {
            key: candidate[key]
            for key in (
                "block",
                "id",
                "feature_id",
                "label",
                "category",
                "score",
                "scores",
            )
            if key in candidate
        }
        compact.append(item)
    return compact


def _compact_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for sample in samples[:8]:
        item = {
            key: sample[key]
            for key in ("id", "label", "category", "confidence", "mean_activation")
            if key in sample
        }
        compact.append(item)
    return compact


def _browse_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    block = result.get("block")
    if isinstance(result.get("samples"), list):
        return [_candidate_with_block(sample, block) for sample in _compact_samples(result["samples"])]

    candidates: list[dict[str, Any]] = []
    categories = result.get("categories")
    if isinstance(categories, dict):
        for category, data in categories.items():
            if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
                continue
            for sample in data["samples"][:2]:
                if isinstance(sample, dict):
                    compact = _compact_samples([sample])[0]
                    compact.setdefault("category", category)
                    candidates.append(_candidate_with_block(compact, block))
            if len(candidates) >= 12:
                break
    return candidates[:12]


def _candidate_with_block(candidate: dict[str, Any], block: Any) -> dict[str, Any]:
    item = dict(candidate)
    if isinstance(block, str):
        item.setdefault("block", block)
    return item


def _browse_summary(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("samples"), list):
        return {
            "block": result.get("block"),
            "category": result.get("category"),
            "count": result.get("count"),
            "sample_count": len(result["samples"]),
            "samples": _compact_samples(result["samples"])[:5],
        }

    categories = result.get("categories")
    category_counts: dict[str, int] = {}
    if isinstance(categories, dict):
        for category, data in categories.items():
            if isinstance(data, dict) and isinstance(data.get("count"), int):
                category_counts[category] = data["count"]

    return {
        "block": result.get("block"),
        "total_features": result.get("total_features"),
        "category_counts": category_counts,
    }


def _state_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    control_state = result.get("control_state")
    summary = control_state.get("summary") if isinstance(control_state, dict) else {}
    prompt = summary.get("prompt") if isinstance(summary, dict) else {}
    composition = summary.get("composition") if isinstance(summary, dict) else {}
    enabled_block_count = summary.get("enabled_block_count") if isinstance(summary, dict) else None
    prompt_empty = (
        isinstance(prompt, dict)
        and prompt.get("destination_a") is None
        and prompt.get("destination_b") is None
    )
    latent_empty = (
        isinstance(composition, dict)
        and composition.get("seed_a") is None
        and composition.get("seed_b") is None
    )

    return {
        "armed": result.get("armed"),
        "mode": result.get("mode"),
        "active_session": result.get("active_session"),
        "has_song_profile": bool(result.get("song_profile")),
        "has_song_analysis": bool(result.get("song_analysis_available")),
        "has_control_state": bool(result.get("control_state")),
        "entry_context": result.get("entry_context"),
        "enabled_block_count": enabled_block_count,
        "prompt_empty": prompt_empty,
        "latent_empty": latent_empty,
        "composition": composition if isinstance(composition, dict) else {},
        "blank_start": enabled_block_count == 0 and prompt_empty and latent_empty,
    }


def _entry_context_value(
    summary: dict[str, Any],
    *path: str,
) -> Any:
    value: Any = summary.get("entry_context")
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _requires_fresh_entry_song_analysis(summary: dict[str, Any]) -> bool:
    """Return true when whole-song analysis must precede search/apply.

    This catches the common fresh loaded-idle blank setup path where Hermes can
    see that whole-song analysis exists. In that situation, feature
    search without reading the analysis tends to produce generic guesses.
    """
    situation = _entry_context_value(summary, "situation")
    fresh_blank = _entry_context_value(summary, "control", "fresh_blank_setup")
    return (
        summary.get("armed") is True
        and summary.get("active_session") is False
        and summary.get("has_song_analysis") is True
        and (summary.get("blank_start") is True or fresh_blank is True)
        and situation == "song_loaded_idle"
    )


def _fresh_entry_signature(summary: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable identity for the fresh blank song gate."""
    return (
        _entry_context_value(summary, "situation"),
        _entry_context_value(summary, "audio", "audio_id_present"),
        _entry_context_value(summary, "audio", "duration"),
        _entry_context_value(summary, "control", "fresh_blank_setup"),
        summary.get("blank_start"),
    )


def _clamp_stage_value(value: Any) -> Any:
    if not isinstance(value, (int, float)):
        return value
    return max(STAGE_MIN, min(STAGE_MAX, float(value)))


def _clamp_stage_fields(item: dict[str, Any]) -> None:
    for key in (
        "stage_left",
        "stage_home",
        "stage_right",
        "strength_min",
        "strength_max",
    ):
        if key in item:
            item[key] = _clamp_stage_value(item[key])


def _normalize_stage_object(item: dict[str, Any]) -> None:
    stage = item.pop("stage", None)
    if not isinstance(stage, dict):
        return
    aliases = {
        "left": "stage_left",
        "home": "stage_home",
        "center": "stage_home",
        "right": "stage_right",
    }
    for source, target in aliases.items():
        if target not in item and source in stage:
            item[target] = stage[source]


def _normalize_common_reactive_aliases(item: dict[str, Any]) -> None:
    _normalize_stage_object(item)
    item.pop("target", None)

    if "blend_slew" in item and "blend_slew_rate" not in item:
        item["blend_slew_rate"] = item.pop("blend_slew")
    else:
        item.pop("blend_slew", None)

    if "smoothing" in item and "position_smoothing_ms" not in item:
        smoothing = item.pop("smoothing")
        if isinstance(smoothing, (int, float)) and 0 <= smoothing <= 10:
            item["position_smoothing_ms"] = float(smoothing) * 1000
        else:
            item["position_smoothing_ms"] = smoothing
    else:
        item.pop("smoothing", None)

    silence = item.get("silence_behavior")
    if silence in {"return_home", "return_center", "center"}:
        item["silence_behavior"] = "drift_center"
    elif silence in {"hold", "hold_latest"}:
        item["silence_behavior"] = "hold_last"

    weights = item.get("rank_weights")
    if isinstance(weights, dict):
        normalized_weights = {
            str(key): float(value)
            for key, value in weights.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if normalized_weights:
            item["rank_weights"] = normalized_weights
        else:
            item.pop("rank_weights", None)


def _normalize_intensity_source_aliases(item: dict[str, Any], *, allow_position: bool) -> None:
    source = item.get("intensity_source")
    if source == "brightness":
        if allow_position and "position_source" not in item:
            item["position_source"] = "brightness"
        item["intensity_source"] = "energy_smooth"
    elif source in {"sustain", "sustained", "body", "smooth", "energy"}:
        item["intensity_source"] = "energy_smooth"
    elif source in {"attack", "hit", "hits", "percussive", "onset", "onsets"}:
        item["intensity_source"] = "transient"
    elif source in {"motion", "change", "texture_motion"}:
        item["intensity_source"] = "flux"


def _normalize_visual_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            normalized.append(action)
            continue
        item = dict(action)
        legacy_type = item.pop("type", None)
        if "action" not in item and isinstance(legacy_type, str):
            item["action"] = legacy_type
        if item.get("action") in {"set_prompt_destinations", "set_prompt"}:
            prompt_a = item.get("prompt_a") or item.get("a")
            prompt_b = item.get("prompt_b") or item.get("b")
            if isinstance(prompt_a, str) and prompt_a.strip():
                normalized.append(
                    {
                        "action": "set_destination",
                        "space": "prompt",
                        "slot": "a",
                        "destination_type": "prompt",
                        "prompt": prompt_a,
                    }
                )
            if isinstance(prompt_b, str) and prompt_b.strip():
                normalized.append(
                    {
                        "action": "set_destination",
                        "space": "prompt",
                        "slot": "b",
                        "destination_type": "prompt",
                        "prompt": prompt_b,
                    }
                )
            continue
        if item.get("action") == "set_composition":
            item["action"] = "set_composition_config"
        if item.get("action") == "set_destination":
            _normalize_destination_action(item)
        if item.get("action") == "update_block_config":
            _normalize_stage_object(item)
            if "rank" in item and "sae_rank" not in item:
                item["sae_rank"] = item.pop("rank")
            else:
                item.pop("rank", None)
            if item.get("enabled") is not False:
                item["sae_rank"] = 1
            if "gamma" in item and "intensity_gamma" not in item:
                item["intensity_gamma"] = item.pop("gamma")
            else:
                item.pop("gamma", None)
            _normalize_intensity_source_aliases(item, allow_position=False)
            _clamp_stage_fields(item)
        elif item.get("action") == "set_reactive_config":
            _normalize_common_reactive_aliases(item)
            _normalize_intensity_source_aliases(item, allow_position=True)
            _clamp_stage_fields(item)
        if item.get("action") == "set_composition_config":
            seed_a = item.pop("seed_a", None)
            seed_b = item.pop("seed_b", None)
            if isinstance(item.get("distance"), (int, float)):
                item["distance"] = max(0.0, min(4.0, float(item["distance"])))
            if isinstance(seed_a, int):
                normalized.append(
                    {
                        "action": "set_destination",
                        "space": "latent",
                        "slot": "a",
                        "destination_type": "seed",
                        "seed": seed_a,
                    }
                )
            if isinstance(seed_b, int):
                normalized.append(
                    {
                        "action": "set_destination",
                        "space": "latent",
                        "slot": "b",
                        "destination_type": "seed",
                        "seed": seed_b,
                    }
                )
        normalized.append(item)
    return normalized


def _normalize_destination_action(item: dict[str, Any]) -> None:
    destination_type = item.get("destination_type")
    if destination_type in {"prompt_a", "prompt_b", "seed_a", "seed_b"}:
        suffix = str(destination_type).rsplit("_", maxsplit=1)[-1]
        item.setdefault("slot", suffix)
        item["destination_type"] = "prompt" if str(destination_type).startswith("prompt") else "seed"

    if "space" not in item:
        if item.get("destination_type") == "seed" or "seed" in item:
            item["space"] = "latent"
        elif item.get("destination_type") == "prompt" or "prompt" in item:
            item["space"] = "prompt"

    if "destination_type" not in item:
        if item.get("space") == "latent" or "seed" in item:
            item["destination_type"] = "seed"
        elif item.get("space") == "prompt" or "prompt" in item:
            item["destination_type"] = "prompt"


def create_mcp_server(
    client: HambaBridgeClient | None = None,
    feature_catalog: FeatureCatalog | None = None,
) -> Any:
    if FastMCP is None:
        raise RuntimeError(
            "Hermes MCP server requires the `mcp` package. "
            "Install with `uv sync --extra hermes`."
        )

    agent_client = client or client_from_env()
    catalog = feature_catalog
    mcp = FastMCP("hambajuba2ba-hermes")
    fresh_entry_analysis_required = False
    fresh_entry_analysis_read = False
    fresh_entry_signature: tuple[Any, ...] | None = None
    feature_lookup_count = 0
    feature_lookup_budget = 8
    remembered_feature_candidates: dict[tuple[str, int], dict[str, Any]] = {}
    palette_epoch = 0
    palette_song_signature: tuple[Any, ...] | None = None
    feature_palette: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        block: {} for block in FEATURE_BLOCKS
    }
    used_palette_features: set[tuple[str, int]] = set()

    def active_catalog() -> FeatureCatalog:
        nonlocal catalog
        if catalog is None:
            catalog = get_default_catalog()
        return catalog

    def candidate_id(candidate: dict[str, Any]) -> int | None:
        value = candidate.get("id", candidate.get("feature_id"))
        if isinstance(value, int):
            return value
        return None

    def song_signature_from_state(
        result: dict[str, Any],
        summary: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        audio = _entry_context_value(summary, "audio")
        if not isinstance(audio, dict) or audio.get("audio_id_present") is not True:
            return None
        profile = result.get("song_profile")
        profile_audio_id = profile.get("audio_id") if isinstance(profile, dict) else None
        profile_duration = profile.get("duration") if isinstance(profile, dict) else None
        return (
            profile_audio_id,
            profile_duration or audio.get("duration"),
            audio.get("upload_phase"),
        )

    def reset_feature_palette(*, next_epoch: bool) -> None:
        nonlocal palette_epoch
        if next_epoch:
            palette_epoch += 1
        for block_candidates in feature_palette.values():
            block_candidates.clear()
        used_palette_features.clear()

    def palette_candidates_for_block(
        block: str,
        *,
        limit: int,
        prefer_unused: bool,
    ) -> list[dict[str, Any]]:
        candidates = list(feature_palette.get(block, {}).values())
        bucket_order = {
            "orthogonal": 0,
            "wildcard": 1,
            "anti_semantic": 2,
            "adjacent": 3,
            "anchor": 4,
        }
        candidates.sort(
            key=lambda candidate: (
                (block, candidate_id(candidate) or -1) in used_palette_features
                if prefer_unused
                else False,
                bucket_order.get(str(candidate.get("palette_bucket")), 9),
                -float(candidate.get("score") or candidate.get("mean_activation") or 0),
                candidate_id(candidate) or 0,
            )
        )
        return candidates[: max(1, min(int(limit), 24))]

    def compact_palette(
        *,
        limit_per_block: int = 8,
        prefer_unused: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            block: palette_candidates_for_block(
                block,
                limit=limit_per_block,
                prefer_unused=prefer_unused,
            )
            for block in FEATURE_BLOCKS
        }

    def palette_candidate_count() -> int:
        return sum(len(items) for items in feature_palette.values())

    def remember_palette_candidates(
        candidates: list[dict[str, Any]],
        *,
        block: str | None,
        bucket: str,
        source: str,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        remembered = remember_feature_candidates(
            candidates,
            block,
            palette_bucket=bucket,
            palette_source=source,
            palette_query=query,
        )
        return remembered

    def selected_palette_candidates_for_actions(
        actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for action in actions:
            if action.get("action") != "update_block_config":
                continue
            block = action.get("block")
            feature_id = action.get("feature_id")
            if not isinstance(block, str) or not isinstance(feature_id, int):
                continue
            candidate = feature_palette.get(block, {}).get((block, feature_id))
            if candidate is not None:
                selected.append(candidate)
        return selected

    def mark_used_palette_features(actions: list[dict[str, Any]]) -> None:
        for action in actions:
            if action.get("action") != "update_block_config":
                continue
            block = action.get("block")
            feature_id = action.get("feature_id")
            if isinstance(block, str) and isinstance(feature_id, int):
                used_palette_features.add((block, feature_id))

    def remember_feature_candidates(
        candidates: list[dict[str, Any]],
        block: str | None = None,
        *,
        palette_bucket: str | None = None,
        palette_source: str | None = None,
        palette_query: str | None = None,
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for candidate in candidates:
            item = _candidate_with_block(candidate, block)
            item_id = candidate_id(item)
            item_block = item.get("block")
            if not isinstance(item_block, str) or item_id is None:
                continue
            if palette_bucket is not None:
                item["palette_bucket"] = palette_bucket
                item["palette_epoch"] = palette_epoch
            if palette_source is not None:
                item["palette_source"] = palette_source
            if palette_query:
                item["palette_query"] = palette_query
            remembered_feature_candidates[(item_block, item_id)] = item
            if palette_bucket is not None and item_block in feature_palette:
                feature_palette[item_block][(item_block, item_id)] = item
            compact.append(item)
        return compact

    def remembered_candidates() -> list[dict[str, Any]]:
        return list(remembered_feature_candidates.values())[:48]

    def merge_feature_candidates(
        candidates: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for candidate in remembered_candidates() + (candidates or []):
            item_id = candidate_id(candidate)
            item_block = candidate.get("block")
            if isinstance(item_block, str) and item_id is not None:
                merged[(item_block, item_id)] = candidate
        return list(merged.values())

    def reset_feature_lookup_state() -> None:
        nonlocal feature_lookup_count
        feature_lookup_count = 0
        remembered_feature_candidates.clear()

    def feature_lookup_budget_result(
        *,
        block: str,
        query: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        return {
            "block": block,
            "query": query,
            "category": category,
            "candidate_count": 0,
            "candidates": [],
            "remembered_feature_candidates": remembered_candidates(),
            "search_budget": {
                "used": feature_lookup_count,
                "limit": feature_lookup_budget,
                "exhausted": True,
                "required_next_step": (
                    "Stop feature lookup and call hamba_apply_visual_plan now "
                    "using remembered candidates."
                ),
            },
        }

    def divergence_bucket_mix(divergence: float) -> dict[str, float]:
        if divergence < 0.35:
            return {
                "anchor": 0.55,
                "adjacent": 0.30,
                "orthogonal": 0.10,
                "wildcard": 0.05,
                "anti_semantic": 0.0,
            }
        if divergence < 0.70:
            return {
                "anchor": 0.35,
                "adjacent": 0.30,
                "orthogonal": 0.25,
                "wildcard": 0.10,
                "anti_semantic": 0.0,
            }
        if divergence < 0.90:
            return {
                "anchor": 0.25,
                "adjacent": 0.25,
                "orthogonal": 0.30,
                "wildcard": 0.15,
                "anti_semantic": 0.05,
            }
        return {
            "anchor": 0.15,
            "adjacent": 0.20,
            "orthogonal": 0.35,
            "wildcard": 0.25,
            "anti_semantic": 0.05,
        }

    def bucket_counts(per_block: int, divergence: float) -> dict[str, int]:
        count = max(8, min(int(per_block), 40))
        mix = divergence_bucket_mix(max(0.0, min(1.0, float(divergence))))
        counts = {
            bucket: max(0, int(round(count * weight)))
            for bucket, weight in mix.items()
        }
        while sum(counts.values()) < count:
            counts["orthogonal"] += 1
        while sum(counts.values()) > count and counts["anchor"] > 1:
            counts["anchor"] -= 1
        return counts

    def palette_query_for_block(block: str, theme: str, *, adjacent: bool) -> str:
        terms = {
            "down.2.1": "scene architecture landscape figure object environment",
            "mid.0": "geometry symmetry structure density rhythm depth pattern",
            "up.0.0": "detail edge face texture accessory small repeating marks",
            "up.0.1": "style material texture lighting color atmosphere surface",
        }
        if adjacent:
            return f"{theme} {terms.get(block, '')} lateral contrast".strip()
        return theme.strip() or terms.get(block, "visual feature")

    def palette_avoid_ids(block: str) -> list[int]:
        ids = []
        for candidate in feature_palette.get(block, {}).values():
            item_id = candidate_id(candidate)
            if item_id is not None:
                ids.append(item_id)
        return ids

    async def report_tool_event(
        *,
        name: str,
        status: str,
        arguments: dict[str, Any],
        phase: str,
        summary: str,
        result_summary: dict[str, Any] | None = None,
        feature_candidates: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "phase": phase,
            "summary": summary,
            "tool": {
                "name": name,
                "status": status,
                "arguments": arguments,
                "result_summary": result_summary or {},
            },
            "feature_candidates": feature_candidates or [],
            "error": error,
        }
        with suppress(Exception):
            await agent_client.report_phase(payload)

    @mcp.tool()
    async def hamba_get_state() -> dict[str, Any]:
        """Read the active Hamba visualizer state from the local bridge."""
        nonlocal fresh_entry_analysis_required, fresh_entry_analysis_read, fresh_entry_signature, palette_song_signature, palette_epoch
        result = await agent_client.get_state()
        summary = _state_result_summary(result)
        song_signature = song_signature_from_state(result, summary)
        if song_signature is None:
            if palette_song_signature is not None or palette_candidate_count() > 0:
                reset_feature_palette(next_epoch=False)
            palette_song_signature = None
            palette_epoch = 0
        elif song_signature != palette_song_signature:
            reset_feature_palette(next_epoch=False)
            palette_song_signature = song_signature
            palette_epoch = 0
        fresh_required = _requires_fresh_entry_song_analysis(summary)
        if fresh_required:
            signature = _fresh_entry_signature(summary)
            if signature != fresh_entry_signature:
                fresh_entry_analysis_read = False
                reset_feature_lookup_state()
            fresh_entry_signature = signature
        else:
            fresh_entry_analysis_read = False
            fresh_entry_signature = None
        fresh_entry_analysis_required = fresh_required
        await report_tool_event(
            name="hamba_get_state",
            status="completed",
            arguments={},
            phase="thinking",
            summary="Read visualizer state",
            result_summary=summary,
        )
        return result

    @mcp.tool()
    async def hamba_get_control_surface() -> dict[str, Any]:
        """Read static Hamba control vocabulary, ranges, modes, and runtime notes."""
        result = await agent_client.get_control_surface()
        await report_tool_event(
            name="hamba_get_control_surface",
            status="completed",
            arguments={},
            phase="thinking",
            summary="Read Hamba control surface",
            result_summary={
                "version": result.get("version"),
                "has_blocks": bool(result.get("blocks")),
                "has_link_targets": bool(result.get("link_targets")),
                "has_prompt": bool(result.get("prompt")),
                "has_composition": bool(result.get("composition")),
            },
        )
        return result

    @mcp.tool()
    async def hamba_get_music_window(
        lookback: float = 8.0,
        lookahead: float = 16.0,
    ) -> dict[str, Any]:
        """Read compact recent/current/upcoming DSP windows, including near-future per-target evidence."""
        result = await agent_client.get_music_window(
            lookback=lookback,
            lookahead=lookahead,
        )
        await report_tool_event(
            name="hamba_get_music_window",
            status="completed",
            arguments={"lookback": lookback, "lookahead": lookahead},
            phase="thinking",
            summary="Read local music window",
            result_summary={
                "active_session": result.get("active_session"),
                "sampled_at_audio_time": result.get("sampled_at_audio_time"),
                "bpm": result.get("bpm"),
                "section": result.get("section"),
            },
        )
        return result

    @mcp.tool()
    async def hamba_get_song_analysis() -> dict[str, Any]:
        """Required first planning tool for fresh loaded-idle blank songs; reads whole-song DSP affordances plus filename metadata when available."""
        nonlocal fresh_entry_analysis_read
        result = await agent_client.get_song_analysis()
        fresh_entry_analysis_read = True
        analysis = result.get("analysis") if isinstance(result.get("analysis"), dict) else {}
        ranked = analysis.get("ranked_drivers") if isinstance(analysis, dict) else {}
        primary = ranked.get("primary_driver", []) if isinstance(ranked, dict) else []
        await report_tool_event(
            name="hamba_get_song_analysis",
            status="completed",
            arguments={},
            phase="thinking",
            summary="Read whole-song DSP analysis",
            result_summary={
                "available": result.get("available"),
                "target_count": analysis.get("target_count"),
                "top_primary_targets": primary[:3],
                "metadata": analysis.get("metadata"),
                "metadata_policy": analysis.get("metadata_policy"),
            },
        )
        return result

    @mcp.tool()
    async def hamba_prepare_feature_palette(
        theme: str | None = None,
        divergence: float = 0.75,
        reset: bool = False,
        per_block: int = 24,
    ) -> dict[str, Any]:
        """Prepare a durable per-song feature palette for Auto Dance checkpoints.

        Use this during first setup or after a major vibe change. It deliberately
        mixes relevant anchors with adjacent, orthogonal, and wildcard browse
        candidates so live Auto Dance can swap features without live searching.
        """
        nonlocal palette_epoch
        arguments = {
            "theme": theme,
            "divergence": divergence,
            "reset": reset,
            "per_block": per_block,
        }
        if fresh_entry_analysis_required and not fresh_entry_analysis_read:
            message = (
                "Call hamba_get_song_analysis now, then prepare the feature palette with the whole-song analysis in hand."
            )
            await report_tool_event(
                name="hamba_prepare_feature_palette",
                status="failed",
                arguments=arguments,
                phase="error",
                summary="Feature palette preparation blocked until whole-song analysis is read",
                error=message,
            )
            raise RuntimeError(message)

        if reset or palette_epoch == 0:
            reset_feature_lookup_state()
            reset_feature_palette(next_epoch=True)

        safe_divergence = max(0.0, min(1.0, float(divergence)))
        counts = bucket_counts(per_block, safe_divergence)
        theme_text = (theme or "").strip()
        prepared: dict[str, dict[str, int]] = {}

        for block in active_catalog().blocks:
            prepared[block] = {}
            if counts.get("anchor", 0) > 0:
                query = palette_query_for_block(block, theme_text, adjacent=False)
                result = active_catalog().search_details(
                    block=block,
                    query=query,
                    limit=counts["anchor"],
                    seed=f"palette-{palette_epoch}-{block}-anchor-{theme_text}",
                    temperature=0.30,
                    semantic=True,
                    avoid_feature_ids=palette_avoid_ids(block),
                )
                candidates = remember_palette_candidates(
                    _compact_candidates(result.get("candidates", [])),
                    block=block,
                    bucket="anchor",
                    source="palette_search",
                    query=query,
                )
                prepared[block]["anchor"] = len(candidates)

            if counts.get("adjacent", 0) > 0:
                query = palette_query_for_block(block, theme_text, adjacent=True)
                result = active_catalog().search_details(
                    block=block,
                    query=query,
                    limit=counts["adjacent"],
                    seed=f"palette-{palette_epoch}-{block}-adjacent-{theme_text}",
                    temperature=0.55,
                    semantic=True,
                    avoid_feature_ids=palette_avoid_ids(block),
                )
                candidates = remember_palette_candidates(
                    _compact_candidates(result.get("candidates", [])),
                    block=block,
                    bucket="adjacent",
                    source="palette_search",
                    query=query,
                )
                prepared[block]["adjacent"] = len(candidates)

            for bucket, temperature in (
                ("orthogonal", 0.85),
                ("wildcard", 0.95),
                ("anti_semantic", 1.0),
            ):
                requested = counts.get(bucket, 0)
                if requested <= 0:
                    continue
                result = active_catalog().browse(
                    block=block,
                    category=None,
                    sample_count=requested,
                    seed=f"palette-{palette_epoch}-{block}-{bucket}-{theme_text}",
                    temperature=temperature,
                    avoid_feature_ids=palette_avoid_ids(block),
                )
                candidates = remember_palette_candidates(
                    _browse_candidates(result),
                    block=block,
                    bucket=bucket,
                    source="palette_browse",
                    query=None,
                )
                prepared[block][bucket] = len(candidates)

        response = {
            "available": palette_candidate_count() > 0,
            "epoch": palette_epoch,
            "divergence": safe_divergence,
            "bucket_mix": divergence_bucket_mix(safe_divergence),
            "prepared_counts": prepared,
            "total_candidate_count": palette_candidate_count(),
            "anti_semantic_available": False,
            "palette": compact_palette(limit_per_block=8, prefer_unused=True),
            "usage": (
                "During Auto Dance, call hamba_get_feature_palette and pick from "
                "this palette before doing any live feature search."
            ),
        }
        await report_tool_event(
            name="hamba_prepare_feature_palette",
            status="completed",
            arguments=arguments,
            phase="searching_features",
            summary="Prepared durable Auto Dance feature palette",
            result_summary={
                "epoch": response["epoch"],
                "divergence": response["divergence"],
                "prepared_counts": prepared,
                "total_candidate_count": response["total_candidate_count"],
                "anti_semantic_available": False,
            },
            feature_candidates=[
                candidate
                for block_candidates in response["palette"].values()
                for candidate in block_candidates
            ],
        )
        return response

    @mcp.tool()
    async def hamba_get_feature_palette(
        limit_per_block: int = 8,
        prefer_unused: bool = True,
    ) -> dict[str, Any]:
        """Read the prepared per-song feature palette for fast Auto Dance swaps."""
        palette = compact_palette(
            limit_per_block=limit_per_block,
            prefer_unused=prefer_unused,
        )
        candidates = [
            candidate for block_candidates in palette.values() for candidate in block_candidates
        ]
        remember_feature_candidates(candidates)
        response = {
            "available": palette_candidate_count() > 0,
            "epoch": palette_epoch,
            "total_candidate_count": palette_candidate_count(),
            "used_candidate_count": len(used_palette_features),
            "limit_per_block": limit_per_block,
            "prefer_unused": prefer_unused,
            "palette": palette,
            "recommended_next_step": (
                "Use one or more unused palette candidates for a visible Auto Dance "
                "evolution. If available=false, call hamba_prepare_feature_palette."
            ),
        }
        await report_tool_event(
            name="hamba_get_feature_palette",
            status="completed",
            arguments={
                "limit_per_block": limit_per_block,
                "prefer_unused": prefer_unused,
            },
            phase="thinking",
            summary="Read prepared Auto Dance feature palette",
            result_summary={
                "available": response["available"],
                "epoch": response["epoch"],
                "total_candidate_count": response["total_candidate_count"],
                "used_candidate_count": response["used_candidate_count"],
            },
            feature_candidates=candidates,
        )
        return response

    @mcp.tool()
    async def hamba_search_features(
        block: str,
        query: str,
        category: str | None = None,
        limit: int = 12,
        seed: str | None = None,
        temperature: float = 0.35,
        semantic: bool = True,
        avoid_feature_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Search relevant SAE feature candidates for one visual clause.

        Search is intentionally relevance-biased. For orthogonal surprises,
        broad first rigs, or high-divergence channels, use hamba_browse_catalog
        to pull lateral candidates from another category or kingdom.
        """
        nonlocal feature_lookup_count
        arguments = {
            "block": block,
            "query": query,
            "category": category,
            "limit": limit,
            "seed": seed,
            "temperature": temperature,
            "semantic": semantic,
            "avoid_feature_ids": avoid_feature_ids or [],
        }
        if fresh_entry_analysis_required and not fresh_entry_analysis_read:
            message = (
                "Call hamba_get_song_analysis now, then retry feature search with the whole-song analysis in hand."
            )
            await report_tool_event(
                name="hamba_search_features",
                status="failed",
                arguments=arguments,
                phase="error",
                summary="Feature search blocked until whole-song analysis is read",
                error=message,
            )
            raise RuntimeError(message)
        if feature_lookup_count >= feature_lookup_budget:
            result = feature_lookup_budget_result(
                block=block,
                query=query,
                category=category,
            )
            await report_tool_event(
                name="hamba_search_features",
                status="completed",
                arguments=arguments,
                phase="planning",
                summary="Feature lookup budget exhausted; apply the remembered candidates now",
                result_summary={
                    "search_budget": result["search_budget"],
                    "remembered_candidate_count": len(result["remembered_feature_candidates"]),
                },
                feature_candidates=result["remembered_feature_candidates"],
            )
            return result
        feature_lookup_count += 1
        try:
            result = active_catalog().search_details(
                block=block,
                query=query,
                category=category,
                limit=limit,
                seed=seed,
                temperature=temperature,
                semantic=semantic,
                avoid_feature_ids=avoid_feature_ids,
            )
        except Exception as exc:
            await report_tool_event(
                name="hamba_search_features",
                status="failed",
                arguments=arguments,
                phase="error",
                summary=f"Feature search failed: {query}",
                error=str(exc),
            )
            raise

        candidates = remember_feature_candidates(
            _compact_candidates(result.get("candidates", [])),
            block,
            palette_bucket="anchor",
            palette_source="search",
            palette_query=query,
        )
        result["search_budget"] = {
            "used": feature_lookup_count,
            "limit": feature_lookup_budget,
            "exhausted": False,
        }
        await report_tool_event(
            name="hamba_search_features",
            status="completed",
            arguments=arguments,
            phase="searching_features",
            summary=f"Searched {block}: {query}",
            result_summary={
                "candidate_count": len(result.get("candidates", [])),
                "retrieval": result.get("retrieval"),
                "top_candidates": candidates[:5],
                "search_budget": result["search_budget"],
            },
            feature_candidates=candidates,
        )
        return result

    @mcp.tool()
    async def hamba_browse_catalog(
        block: str,
        category: str | None = None,
        sample_count: int = 8,
        seed: str | None = None,
        temperature: float = 0.6,
        avoid_feature_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Explore SAE labels for one block, including lateral surprise candidates.

        Browse is the orthogonal-surprise tool. Use it at the beginning of a
        visualization or at high divergence to cast a wide net across categories,
        then converge with search or apply from remembered candidates.
        """
        nonlocal feature_lookup_count
        arguments = {
            "block": block,
            "category": category,
            "sample_count": sample_count,
            "seed": seed,
            "temperature": temperature,
            "avoid_feature_ids": avoid_feature_ids or [],
        }
        if fresh_entry_analysis_required and not fresh_entry_analysis_read:
            message = (
                "Call hamba_get_song_analysis now, then retry catalog browse with the whole-song analysis in hand."
            )
            await report_tool_event(
                name="hamba_browse_catalog",
                status="failed",
                arguments=arguments,
                phase="error",
                summary="Feature browse blocked until whole-song analysis is read",
                error=message,
            )
            raise RuntimeError(message)
        if feature_lookup_count >= feature_lookup_budget:
            result = feature_lookup_budget_result(block=block, category=category)
            await report_tool_event(
                name="hamba_browse_catalog",
                status="completed",
                arguments=arguments,
                phase="planning",
                summary="Feature lookup budget exhausted; apply the remembered candidates now",
                result_summary={
                    "search_budget": result["search_budget"],
                    "remembered_candidate_count": len(result["remembered_feature_candidates"]),
                },
                feature_candidates=result["remembered_feature_candidates"],
            )
            return result
        feature_lookup_count += 1
        try:
            result = active_catalog().browse(
                block=block,
                category=category,
                sample_count=sample_count,
                seed=seed,
                temperature=temperature,
                avoid_feature_ids=avoid_feature_ids,
            )
        except Exception as exc:
            await report_tool_event(
                name="hamba_browse_catalog",
                status="failed",
                arguments=arguments,
                phase="error",
                summary=f"Feature browse failed: {block}",
                error=str(exc),
            )
            raise

        candidates = remember_feature_candidates(
            _browse_candidates(result),
            block,
            palette_bucket="orthogonal",
            palette_source="browse",
            palette_query=category,
        )
        result["search_budget"] = {
            "used": feature_lookup_count,
            "limit": feature_lookup_budget,
            "exhausted": False,
        }
        summary = _browse_summary(result)
        summary["search_budget"] = result["search_budget"]
        await report_tool_event(
            name="hamba_browse_catalog",
            status="completed",
            arguments=arguments,
            phase="searching_features",
            summary=f"Browsed feature catalog: {block}",
            result_summary=summary,
            feature_candidates=candidates,
        )
        return result

    @mcp.tool()
    async def hamba_report_phase(
        mode: str,
        phase: str,
        transcript: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Report local voice/agent state to the Hamba frontend."""
        return await agent_client.report_phase(
            {
                "mode": mode,
                "phase": phase,
                "transcript": transcript,
                "provider": provider,
                "model": model,
                "summary": summary,
                "error": error,
            }
        )

    @mcp.tool()
    async def hamba_apply_visual_plan(
        actions: list[dict[str, Any]],
        based_on_audio_time: Any | None = None,
        based_on_wall_time_ms: Any | None = None,
        max_staleness_sec: Any | None = None,
        transcript: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        reason: str | None = None,
        feature_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Apply a sparse Hamba visual control plan through the frontend bridge.

        For update_block_config actions, use exact keys `sae_rank` and
        `intensity_gamma`; do not send shorthand `rank` or `gamma`.
        Leave timing fields unset; live steering changes are applied when ready,
        not rejected as stale beat-perfect cues.
        """
        normalized_actions = _normalize_visual_actions(actions)
        merged_feature_candidates = merge_feature_candidates(
            selected_palette_candidates_for_actions(normalized_actions)
            + (feature_candidates or [])
        )
        timing_ignored = any(
            value is not None
            for value in (
                based_on_audio_time,
                based_on_wall_time_ms,
                max_staleness_sec,
            )
        )
        arguments = {
            "action_count": len(normalized_actions),
            "actions": normalized_actions,
        }
        if timing_ignored:
            arguments["timing_ignored"] = True
        if fresh_entry_analysis_required and not fresh_entry_analysis_read:
            message = (
                "Call hamba_get_song_analysis now, then retry visual plan apply with the whole-song analysis in hand."
            )
            await report_tool_event(
                name="hamba_apply_visual_plan",
                status="failed",
                arguments=arguments,
                phase="error",
                summary="Visual plan apply blocked until whole-song analysis is read",
                feature_candidates=merged_feature_candidates,
                error=message,
            )
            raise RuntimeError(message)
        await report_tool_event(
            name="hamba_apply_visual_plan",
            status="started",
            arguments=arguments,
            phase="planning",
            summary=reason or "Applying visual plan",
            feature_candidates=merged_feature_candidates,
        )
        try:
            result = await agent_client.apply_visual_plan(
                actions=normalized_actions,
                transcript=transcript,
                provider=provider,
                model=model,
                reason=reason,
                feature_candidates=merged_feature_candidates,
            )
            mark_used_palette_features(normalized_actions)
            reset_feature_lookup_state()
            return result
        except Exception as exc:
            await report_tool_event(
                name="hamba_apply_visual_plan",
                status="failed",
                arguments=arguments,
                phase="error",
                summary="Visual plan apply failed",
                feature_candidates=merged_feature_candidates,
                error=str(exc),
            )
            raise

    return mcp


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
