"""Typed contracts shared by the Hermes integration boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AgentMode = Literal["off", "directive", "dj"]
AgentPhase = Literal[
    "off",
    "armed",
    "listening",
    "transcribing",
    "thinking",
    "searching_features",
    "planning",
    "applying",
    "watching",
    "dj_deciding",
    "cooldown",
    "error",
]
BlockCode = Literal["down.2.1", "mid.0", "up.0.0", "up.0.1"]
LinkTarget = Literal[
    "bass",
    "drums",
    "vocals",
    "other",
    "drums_harmonic",
    "drums_percussive",
    "other_harmonic",
    "other_percussive",
    "bass_harmonic",
    "bass_percussive",
    "vocals_harmonic",
    "vocals_percussive",
    "drums_low",
    "drums_mid",
    "drums_high",
    "other_mid",
    "other_high",
    "tension",
    "tonal_distance",
    "global",
]
DestinationSpace = Literal["latent", "prompt"]
DestinationSlot = Literal["a", "b"]
DestinationType = Literal["seed", "prompt"]
DestinationMode = Literal["slider", "reactive", "linked"]
ReplaceMode = Literal["direct", "from_blend"]
SpatialMode = Literal["draw", "pitch_aligned"]
Rank = Literal[1, 2, 3, 4]
PositionSource = Literal[
    "auto", "pitch", "brightness", "chroma", "tension", "tension_global"
]
IntensitySource = Literal["energy_smooth", "transient", "flux", "envelope"]
SilenceBehavior = Literal["drift_center", "hold_last"]
IntensityCurve = Literal["linear", "gamma", "clip"]
CompositionMode = Literal["auto", "pulse", "continuous"]
IntentClauseKind = Literal[
    "subject",
    "transformation",
    "effect",
    "driver",
    "target",
    "timing",
    "strength",
    "style",
    "composition",
]
IntentTiming = Literal[
    "persistent",
    "section",
    "on_hits",
    "on_transients",
    "on_tension",
    "on_release",
    "ambient",
]
IntentStrength = Literal["subtle", "medium", "strong", "extreme"]

FEATURE_ID_MIN = 0
FEATURE_ID_MAX = 5119
STAGE_MIN = -50.0
STAGE_MAX = 50.0
SPATIAL_MASK_SIZE = 256


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_stage_order(
    *,
    left: float | None,
    home: float | None,
    right: float | None,
) -> None:
    if left is not None and home is not None and left > home:
        raise ValueError("stage_left must be <= stage_home")
    if home is not None and right is not None and home > right:
        raise ValueError("stage_home must be <= stage_right")
    if left is not None and right is not None and left > right:
        raise ValueError("stage_left must be <= stage_right")


def _validate_prompt_space(*, action: str, space: str) -> None:
    if space != "prompt":
        raise ValueError(
            f"{action} only applies to prompt space; use latent seeds and "
            "set_composition_config for composition"
        )


def _clamp_stage(value: Any) -> Any:
    if not isinstance(value, (int, float)):
        return value
    return max(STAGE_MIN, min(STAGE_MAX, float(value)))


def _clamp_stage_fields(value: dict[str, Any]) -> None:
    for key in (
        "stage_left",
        "stage_home",
        "stage_right",
        "strength_min",
        "strength_max",
    ):
        if key in value:
            value[key] = _clamp_stage(value[key])


def _normalize_stage_object(value: dict[str, Any]) -> None:
    stage = value.pop("stage", None)
    if not isinstance(stage, dict):
        return
    aliases = {
        "left": "stage_left",
        "home": "stage_home",
        "center": "stage_home",
        "right": "stage_right",
    }
    for source, target in aliases.items():
        if target not in value and source in stage:
            value[target] = stage[source]


def _normalize_destination_aliases(value: dict[str, Any]) -> None:
    destination_type = value.get("destination_type")
    if destination_type in {"prompt_a", "prompt_b", "seed_a", "seed_b"}:
        suffix = str(destination_type).rsplit("_", maxsplit=1)[-1]
        value.setdefault("slot", suffix)
        value["destination_type"] = (
            "prompt" if str(destination_type).startswith("prompt") else "seed"
        )

    if "space" not in value:
        if value.get("destination_type") == "seed" or "seed" in value:
            value["space"] = "latent"
        elif value.get("destination_type") == "prompt" or "prompt" in value:
            value["space"] = "prompt"

    if "destination_type" not in value:
        if value.get("space") == "latent" or "seed" in value:
            value["destination_type"] = "seed"
        elif value.get("space") == "prompt" or "prompt" in value:
            value["destination_type"] = "prompt"


def _normalize_rank_weights(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    normalized = {
        str(key): float(weight)
        for key, weight in value.items()
        if isinstance(weight, (int, float)) and not isinstance(weight, bool)
    }
    return normalized or None


def _normalize_visual_action_aliases(actions: Any) -> Any:
    if not isinstance(actions, list):
        return actions
    normalized_actions: list[Any] = []
    for action in actions:
        if not isinstance(action, dict):
            normalized_actions.append(action)
            continue
        value = dict(action)
        legacy_type = value.pop("type", None)
        if "action" not in value and isinstance(legacy_type, str):
            value["action"] = legacy_type

        if value.get("action") in {"set_prompt_destinations", "set_prompt"}:
            prompt_a = value.get("prompt_a") or value.get("a")
            prompt_b = value.get("prompt_b") or value.get("b")
            if isinstance(prompt_a, str) and prompt_a.strip():
                normalized_actions.append(
                    {
                        "action": "set_destination",
                        "space": "prompt",
                        "slot": "a",
                        "destination_type": "prompt",
                        "prompt": prompt_a,
                    }
                )
            if isinstance(prompt_b, str) and prompt_b.strip():
                normalized_actions.append(
                    {
                        "action": "set_destination",
                        "space": "prompt",
                        "slot": "b",
                        "destination_type": "prompt",
                        "prompt": prompt_b,
                    }
                )
            continue

        if value.get("action") == "set_composition":
            value["action"] = "set_composition_config"

        if value.get("action") == "set_destination":
            _normalize_destination_aliases(value)

        if value.get("action") in {"set_composition_config", "set_composition"}:
            seed_a = value.pop("seed_a", None)
            seed_b = value.pop("seed_b", None)
            if isinstance(value.get("distance"), (int, float)):
                value["distance"] = max(0.0, min(4.0, float(value["distance"])))
            if isinstance(seed_a, int):
                normalized_actions.append(
                    {
                        "action": "set_destination",
                        "space": "latent",
                        "slot": "a",
                        "destination_type": "seed",
                        "seed": seed_a,
                    }
                )
            if isinstance(seed_b, int):
                normalized_actions.append(
                    {
                        "action": "set_destination",
                        "space": "latent",
                        "slot": "b",
                        "destination_type": "seed",
                        "seed": seed_b,
                    }
                )

        if value.get("action") in {"update_block_config", "set_reactive_config"}:
            _normalize_stage_object(value)
            _clamp_stage_fields(value)

        normalized_actions.append(value)
    return normalized_actions


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentArmRequest(BaseModel):
    armed: bool
    mode: AgentMode = "directive"


class IntentDriver(BaseModel):
    link_target: LinkTarget | None = None
    intensity_source: IntensitySource | None = None
    position_source: PositionSource | None = None
    aliases: list[str] = Field(default_factory=list)


class DirectiveClause(BaseModel):
    text: str
    kind: IntentClauseKind
    subject: str | None = None
    transformation: str | None = None
    effect: str | None = None
    target_blocks: list[BlockCode] = Field(default_factory=list)
    drivers: list[IntentDriver] = Field(default_factory=list)
    timing: IntentTiming = "persistent"
    strength: IntentStrength = "medium"
    confidence: float = Field(ge=0.0, le=1.0)


class DirectiveIntentIR(BaseModel):
    directive: str
    clauses: list[DirectiveClause] = Field(min_length=1)
    summary: str | None = None
    source: Literal["user", "dj"] = "user"


class AgentVisualPlanTiming(BaseModel):
    based_on_audio_time: float | None = Field(default=None, ge=0.0)
    based_on_wall_time_ms: int | None = Field(default=None, ge=0)
    max_staleness_sec: float | None = Field(default=None, gt=0.0)


class AgentActionModel(StrictContractModel):
    pass


class AgentUpdateBlockConfigAction(AgentActionModel):
    action: Literal["update_block_config"] = "update_block_config"
    block: BlockCode
    link_target: LinkTarget | None = None
    feature_label: str | None = None
    feature_id: int | None = Field(default=None, ge=FEATURE_ID_MIN, le=FEATURE_ID_MAX)
    enabled: bool | None = None
    auto_config: bool | None = None
    sae_rank: Rank | None = None
    spatial_mode: SpatialMode | None = None
    spatial_mask: list[float] | None = None
    intensity_source: IntensitySource | None = None
    intensity_curve: IntensityCurve | None = None
    intensity_gamma: float | None = Field(default=None, gt=0.0)
    strength_min: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    strength_max: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    stage_left: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    stage_home: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    stage_right: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        _normalize_stage_object(normalized)
        if "rank" in normalized:
            normalized.setdefault("sae_rank", normalized.pop("rank"))
        if "gamma" in normalized:
            normalized.setdefault("intensity_gamma", normalized.pop("gamma"))
        if normalized.get("enabled") is not False:
            normalized["sae_rank"] = 1
        if normalized.get("intensity_source") == "brightness":
            normalized["intensity_source"] = "energy_smooth"
        elif normalized.get("intensity_source") in {"sustain", "sustained", "body", "smooth", "energy"}:
            normalized["intensity_source"] = "energy_smooth"
        elif normalized.get("intensity_source") in {"attack", "hit", "hits", "percussive", "onset", "onsets"}:
            normalized["intensity_source"] = "transient"
        elif normalized.get("intensity_source") in {"motion", "change", "texture_motion"}:
            normalized["intensity_source"] = "flux"
        return normalized

    @field_validator("spatial_mask")
    @classmethod
    def validate_spatial_mask(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != SPATIAL_MASK_SIZE:
            raise ValueError("spatial_mask must contain 256 values")
        if any(not isfinite(item) or item < 0.0 or item > 1.0 for item in value):
            raise ValueError("spatial_mask values must be finite numbers in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_stage_bounds(self) -> "AgentUpdateBlockConfigAction":
        left = self.stage_left if self.stage_left is not None else self.strength_min
        right = self.stage_right if self.stage_right is not None else self.strength_max
        _validate_stage_order(left=left, home=self.stage_home, right=right)
        return self


class AgentSetDestinationAction(AgentActionModel):
    action: Literal["set_destination"] = "set_destination"
    space: DestinationSpace
    slot: DestinationSlot
    destination_type: DestinationType
    seed: int | None = None
    prompt: str | None = None
    replace_mode: ReplaceMode = "direct"

    @model_validator(mode="after")
    def validate_destination_value(self) -> "AgentSetDestinationAction":
        if self.space == "latent" and self.destination_type != "seed":
            raise ValueError("latent destinations require destination_type='seed'")
        if self.space == "prompt" and self.destination_type != "prompt":
            raise ValueError("prompt destinations require destination_type='prompt'")
        if self.destination_type == "seed" and self.seed is None:
            raise ValueError("seed destinations require seed")
        if self.destination_type == "seed" and self.prompt is not None:
            raise ValueError("seed destinations must not include prompt")
        if self.destination_type == "prompt" and not (
            self.prompt and self.prompt.strip()
        ):
            raise ValueError("prompt destinations require prompt")
        if self.destination_type == "prompt" and self.seed is not None:
            raise ValueError("prompt destinations must not include seed")
        return self


class AgentClearDestinationAction(AgentActionModel):
    action: Literal["clear_destination"] = "clear_destination"
    space: DestinationSpace
    slot: DestinationSlot


class AgentFreezeBlendAction(AgentActionModel):
    action: Literal["freeze_blend"] = "freeze_blend"
    space: DestinationSpace
    target_slot: DestinationSlot

    @model_validator(mode="after")
    def validate_space(self) -> "AgentFreezeBlendAction":
        _validate_prompt_space(action=self.action, space=self.space)
        return self


class AgentSetDestinationModeAction(AgentActionModel):
    action: Literal["set_destination_mode"] = "set_destination_mode"
    space: DestinationSpace
    mode: DestinationMode

    @model_validator(mode="after")
    def validate_mode(self) -> "AgentSetDestinationModeAction":
        _validate_prompt_space(action=self.action, space=self.space)
        if self.mode == "linked":
            raise ValueError("use set_destination_link for linked mode")
        return self


class AgentSetDestinationLinkAction(AgentActionModel):
    action: Literal["set_destination_link"] = "set_destination_link"
    space: DestinationSpace
    link_target: LinkTarget

    @model_validator(mode="after")
    def validate_space(self) -> "AgentSetDestinationLinkAction":
        _validate_prompt_space(action=self.action, space=self.space)
        return self


class AgentSetReactiveConfigAction(AgentActionModel):
    action: Literal["set_reactive_config"] = "set_reactive_config"
    space: DestinationSpace
    stage_left: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    stage_home: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    stage_right: float | None = Field(default=None, ge=STAGE_MIN, le=STAGE_MAX)
    position_source: PositionSource | None = None
    intensity_source: IntensitySource | None = None
    position_smoothing_ms: float | None = Field(default=None, ge=0.0)
    silence_behavior: SilenceBehavior | None = None
    drift_ms: float | None = Field(default=None, ge=0.0)
    intensity_curve: IntensityCurve | None = None
    intensity_gamma: float | None = Field(default=None, gt=0.0)
    stem_rankings: (
        dict[Literal["drums", "bass", "vocals", "other"], Rank | None] | None
    ) = None
    rank_weights: dict[str, float] | None = None
    blend_slew_rate: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        _normalize_stage_object(normalized)
        normalized.pop("target", None)
        if "blend_slew" in normalized:
            normalized.setdefault("blend_slew_rate", normalized.pop("blend_slew"))
        if "smoothing" in normalized:
            smoothing = normalized.pop("smoothing")
            if "position_smoothing_ms" not in normalized:
                if isinstance(smoothing, (int, float)) and 0 <= smoothing <= 10:
                    normalized["position_smoothing_ms"] = float(smoothing) * 1000
                else:
                    normalized["position_smoothing_ms"] = smoothing
        if normalized.get("silence_behavior") in {"return_home", "return_center", "center"}:
            normalized["silence_behavior"] = "drift_center"
        elif normalized.get("silence_behavior") in {"hold", "hold_latest"}:
            normalized["silence_behavior"] = "hold_last"
        if "rank_weights" in normalized:
            normalized["rank_weights"] = _normalize_rank_weights(
                normalized["rank_weights"]
            )
        if normalized.get("intensity_source") == "brightness":
            normalized.setdefault("position_source", "brightness")
            normalized["intensity_source"] = "energy_smooth"
        elif normalized.get("intensity_source") in {"sustain", "sustained", "body", "smooth", "energy"}:
            normalized["intensity_source"] = "energy_smooth"
        elif normalized.get("intensity_source") in {"attack", "hit", "hits", "percussive", "onset", "onsets"}:
            normalized["intensity_source"] = "transient"
        elif normalized.get("intensity_source") in {"motion", "change", "texture_motion"}:
            normalized["intensity_source"] = "flux"
        return normalized

    @model_validator(mode="after")
    def validate_stage_bounds(self) -> "AgentSetReactiveConfigAction":
        _validate_prompt_space(action=self.action, space=self.space)
        _validate_stage_order(
            left=self.stage_left,
            home=self.stage_home,
            right=self.stage_right,
        )
        return self


class AgentSetBlendPositionAction(AgentActionModel):
    action: Literal["set_blend_position"] = "set_blend_position"
    space: DestinationSpace
    position: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_space(self) -> "AgentSetBlendPositionAction":
        _validate_prompt_space(action=self.action, space=self.space)
        return self


class AgentSetCompositionConfigAction(AgentActionModel):
    action: Literal["set_composition_config"] = "set_composition_config"
    distance: float | None = Field(default=None, ge=0.0, le=4.0)
    mode: CompositionMode | None = None


AgentVisualAction = Annotated[
    Union[
        AgentUpdateBlockConfigAction,
        AgentSetDestinationAction,
        AgentClearDestinationAction,
        AgentFreezeBlendAction,
        AgentSetDestinationModeAction,
        AgentSetDestinationLinkAction,
        AgentSetReactiveConfigAction,
        AgentSetBlendPositionAction,
        AgentSetCompositionConfigAction,
    ],
    Field(discriminator="action"),
]


class AgentVisualPlan(AgentVisualPlanTiming):
    actions: list[AgentVisualAction] = Field(min_length=1)
    transcript: str | None = None
    provider: str | None = None
    model: str | None = None
    reason: str | None = None
    feature_candidates: list[dict[str, Any]] = Field(default_factory=list)
    intent: DirectiveIntentIR | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_action_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized["actions"] = _normalize_visual_action_aliases(normalized.get("actions"))
        return normalized


class AgentToolEvent(BaseModel):
    name: str
    status: Literal["started", "completed", "failed"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)


class AgentEvent(BaseModel):
    type: Literal["agent_event"] = "agent_event"
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str = Field(default_factory=utc_now_iso)
    mode: AgentMode
    phase: AgentPhase
    provider: str | None = None
    model: str | None = None
    transcript: str | None = None
    summary: str | None = None
    tool: AgentToolEvent | None = None
    feature_candidates: list[dict[str, Any]] = Field(default_factory=list)
    changes: list[dict[str, Any]] = Field(default_factory=list)
    intent: DirectiveIntentIR | None = None
    error: str | None = None


class AgentApplyRequest(AgentVisualPlan):
    pass


class AgentApplyResponse(BaseModel):
    accepted: bool
    event: AgentEvent
    responses: list[dict[str, Any]] = Field(default_factory=list)


class FeatureEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    label: str
    category: str = ""
    confidence: str | None = None
    mean_activation: float | None = Field(default=None, alias="meanActivation")


class FeatureCandidate(BaseModel):
    block: BlockCode
    id: int
    label: str
    category: str
    confidence: str | None = None
    mean_activation: float | None = None
    score: float


class FeatureCandidateScores(BaseModel):
    lexical: float = 0.0
    semantic: float = 0.0
    quality: float = 0.0
    diversity: float = 0.0


class FeatureCandidateDetail(FeatureCandidate):
    scores: FeatureCandidateScores | None = None


class FeatureRetrievalMetadata(BaseModel):
    lexical: bool = True
    semantic: bool = False
    embedding_model: str | None = None
    fallback: str | None = None


class FeatureSearchResponse(BaseModel):
    block: BlockCode
    query: str
    category: str | None = None
    candidates: list[FeatureCandidate]


class FeatureSearchDetailsResponse(BaseModel):
    block: BlockCode
    query: str
    category: str | None = None
    seed: str | None = None
    temperature: float
    retrieval: FeatureRetrievalMetadata
    candidates: list[FeatureCandidateDetail]


class FeatureBrowseSample(BaseModel):
    id: int
    label: str
    category: str
    confidence: str | None = None
    mean_activation: float | None = None


class FeatureBrowseCategory(BaseModel):
    count: int
    samples: list[FeatureBrowseSample]


class FeatureBrowseResponse(BaseModel):
    block: BlockCode
    category: str | None = None
    count: int | None = None
    total_features: int | None = None
    seed: str | None = None
    temperature: float
    samples: list[FeatureBrowseSample] | None = None
    categories: dict[str, FeatureBrowseCategory] | None = None


class AgentStateResponse(BaseModel):
    armed: bool
    mode: AgentMode
    active_session: bool
    block_configs: dict[str, Any] = Field(default_factory=dict)
    control_state: dict[str, Any] = Field(default_factory=dict)
    destinations: dict[str, Any] = Field(default_factory=dict)
    composition: dict[str, Any] = Field(default_factory=dict)
    song_profile: dict[str, Any] | None = None
    latest_event: AgentEvent | None = None


class MusicWindowResponse(BaseModel):
    active_session: bool
    current_time: float
    sampled_at_audio_time: float
    sampled_at_wall_time_ms: int
    duration: float | None = None
    bpm: float | None = None
    is_playing: bool
    lookback: float
    lookahead: float
    song_intelligence_available: bool = False
    song_profile: dict[str, Any] | None = None
    section: dict[str, Any] | None = None
    at_current_time: dict[str, Any] | None = None
    lookahead_context: dict[str, Any] | None = None
    window_summary: dict[str, Any] | None = None
    aggregate_windows: dict[str, Any] | None = None
    target_windows: dict[str, Any] | None = None
    ranked_window_targets: dict[str, Any] | None = None
    auto_dance_hints: dict[str, Any] | None = None
    dominant_targets: dict[str, Any] | None = None
    stems: dict[str, Any] | None = None
    prominence: dict[str, Any] | None = None
    block_configs: dict[str, Any] | None = None
    snapshots: list[dict[str, Any]]


class SongAnalysisResponse(BaseModel):
    available: bool
    audio_id: str | None = None
    sampled_at_wall_time_ms: int | None = None
    analysis: dict[str, Any] | None = None
