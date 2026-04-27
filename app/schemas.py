"""WebSocket message schemas for type-safe communication.

All messages exchanged over WebSocket are validated against these Pydantic models.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


# ============================================================================
# Client → Server Messages
# ============================================================================


class AudioTimeUpdate(BaseModel):
    """Audio playback time update (for sync correction)."""

    action: Literal["audio_timeupdate"] = "audio_timeupdate"
    time: float


class AudioPlay(BaseModel):
    """Audio playback started/resumed."""

    action: Literal["audio_play"] = "audio_play"
    time: float


class AudioPause(BaseModel):
    """Audio playback paused."""

    action: Literal["audio_pause"] = "audio_pause"


class AudioSeek(BaseModel):
    """Audio playback seeked to new position."""

    action: Literal["audio_seek"] = "audio_seek"
    time: float


class Stop(BaseModel):
    """Stop generation and fully disconnect."""

    action: Literal["stop"] = "stop"


class StopGeneration(BaseModel):
    """Stop generation but keep session alive for new songs."""

    action: Literal["stop_generation"] = "stop_generation"


# ============================================================================
# SAE Steering Messages (Client → Server)
# ============================================================================


class StartGeneration(BaseModel):
    """Base for all start messages. Subclasses add backend-specific fields."""

    audio_id: str


class StartSAESteering(StartGeneration):
    """Start SAE steering mode with audio-driven features."""

    action: Literal["start_sae_steering"] = "start_sae_steering"
    prompt: Optional[str] = None


# Audio feature channels
ChannelType = Literal[
    "envelope", "energy_smooth", "transient", "flux", "brightness",
]

# Dual-layer response options
LayerType = Literal["flash", "sustain", "combined"]

# Position/intensity sources for dance model
PositionSourceType = Literal[
    "auto", "pitch", "brightness", "chroma", "tension", "tension_global",
]
IntensitySourceType = Literal[
    "energy_smooth", "transient", "flux", "envelope",
]
SilenceBehaviorType = Literal["drift_center", "hold_last"]
IntensityCurveType = Literal["linear", "gamma", "impulse", "clip"]

# Physics presets
PhysicsPresetType = Literal[
    "kick", "drums", "drums_low", "drums_mid", "drums_high",
    "bass", "vocals", "other", "other_mid", "other_high",
    "ambient", "linear",
]

# Spatial modes: draw (user-painted 16x16 grid) or pitch_aligned (auto from audio)
SpatialModeType = Literal["draw", "pitch_aligned"]


# ============================================================================
# Audio v2: LinkTarget types
# ============================================================================

# All possible link targets for v2 audio-reactive system
LinkTargetType = Literal[
    # Physical stems (Demucs output)
    "bass", "drums", "vocals", "other",
    # HPSS components (harmonic/percussive separation)
    "drums_harmonic", "drums_percussive",
    "other_harmonic", "other_percussive",
    "bass_harmonic", "bass_percussive",
    "vocals_harmonic", "vocals_percussive",
    # Sub-bands (frequency separation)
    "drums_low", "drums_mid", "drums_high",
    "other_mid", "other_high",
    # Derived
    "tension", "tonal_distance", "global",
]


class SetSteeringMode(BaseModel):
    """Set steering mode (AUTO applies prominence weighting, MANUAL uses equal weights)."""

    action: Literal["set_steering_mode"] = "set_steering_mode"
    mode: Literal["manual", "auto"]


# ============================================================================
# Audio v2: Block Configuration Messages
# ============================================================================


# Rank type for Dancer Ensemble Architecture
# 1-4 are explicit ranks, None means auto/available for surprise
RankType = Literal[1, 2, 3, 4]


class UpdateBlockConfig(BaseModel):
    """Update configuration for an SAE block.

    Uses LinkTarget for audio source, StrengthRange for intuitive bounds,
    and ranking for the Dancer Ensemble architecture.

    Ranking System:
        - Rank 1: Main dancer(s) - primary visual focus
        - Rank 2: Backup dancer(s) - supporting, visible
        - Rank 3: Background - ambient presence
        - Rank 4: Barely there - subtle texture
        - None/null: Auto - available for surprise promotion

    Note: SLERP rankings are configured per-destination via SetReactiveConfig,
    not per-block. This keeps SAE steering separate from destination control.
    """

    action: Literal["update_block_config"] = "update_block_config"
    block: str  # SAE block name (down.2.1, mid.0, up.0.0, up.0.1)
    link_target: Optional[LinkTargetType] = None  # Audio source
    strength_min: Optional[float] = None
    strength_max: Optional[float] = None
    feature_id: Optional[int] = None
    enabled: Optional[bool] = None
    auto_config: Optional[bool] = None  # If True, derive channel/layer/spatial from classification

    # Dancer Ensemble ranking (1-4 or None for auto)
    sae_rank: Optional[RankType] = None  # Rank for SAE feature steering

    # Spatial: draw (user-painted grid) or pitch_aligned (auto from audio)
    spatial_mode: Optional[SpatialModeType] = None
    spatial_mask: Optional[list[float]] = None  # 256 floats (16x16 grid)

    # Manual overrides (when auto_config=False)
    channel: Optional[ChannelType] = None
    layer: Optional[LayerType] = None
    physics_preset: Optional[PhysicsPresetType] = None

    # v4 dance model overrides
    stage_left: Optional[float] = None
    stage_home: Optional[float] = None
    stage_right: Optional[float] = None
    position_source: Optional[PositionSourceType] = None
    intensity_source: Optional[IntensitySourceType] = None
    position_smoothing_ms: Optional[float] = None
    silence_behavior: Optional[SilenceBehaviorType] = None
    drift_ms: Optional[float] = None
    intensity_curve: Optional[IntensityCurveType] = None
    intensity_gamma: Optional[float] = None


class SetDestinationLink(BaseModel):
    """Set link target for destination reactive mode (v2 system).

    Allows tension-driven SLERP: high tension → destination B (e.g., darker visual).
    """

    action: Literal["set_destination_link"] = "set_destination_link"
    space: Literal["latent", "prompt"]
    link_target: LinkTargetType  # e.g., "tension", "bass", "drums_percussive"


# Composition mode for noise circular walk
CompositionModeType = Literal["auto", "pulse", "continuous"]


class SetCompositionConfig(BaseModel):
    """Configure composition engine (noise circular walk).

    distance: How far around the circle each beat moves.
              1.0 = full 2*pi per energetic beat, 0.5 = half circle.
    mode: "auto" (adaptive beat/drift), "pulse" (beats only),
          "continuous" (drift only).
    """

    action: Literal["set_composition_config"] = "set_composition_config"
    distance: Optional[float] = None
    mode: Optional[CompositionModeType] = None


# ============================================================================
# Destination Modulation Messages (Client → Server)
# ============================================================================

# Destination driver type for reactive mode
DestinationDriverType = Literal["global", "stem"]


class SetDestination(BaseModel):
    """Set a destination (A or B) in latent or prompt space.

    For latent space: only seed type is supported (loads noise into CompositionEngine).
    For prompt space: only prompt type is supported (loads into prompt SLERP).
    """

    action: Literal["set_destination"] = "set_destination"
    space: Literal["latent", "prompt"]
    slot: Literal["a", "b"]
    destination_type: Literal["seed", "prompt"]
    seed: Optional[int] = None
    prompt: Optional[str] = None
    replace_mode: Literal["direct", "from_blend"] = "direct"


class FreezeBlend(BaseModel):
    """Freeze current blend position into a specific slot.

    Captures the current SLERP result at the current blend position
    and loads it into the target slot. Example: at 50% blend of A and B,
    freeze to A makes A = the current blended state.
    """

    action: Literal["freeze_blend"] = "freeze_blend"
    space: Literal["latent", "prompt"]
    target_slot: Literal["a", "b"]


class SetBlendPosition(BaseModel):
    """Set blend position manually (slider mode)."""

    action: Literal["set_blend_position"] = "set_blend_position"
    space: Literal["latent", "prompt"]
    position: Annotated[float, Field(ge=0.0, le=1.0)]


class SetDestinationMode(BaseModel):
    """Set destination mode (slider, reactive/global, or linked)."""

    action: Literal["set_destination_mode"] = "set_destination_mode"
    space: Literal["latent", "prompt"]
    mode: Literal["slider", "reactive", "linked"]


class SetReactiveConfig(BaseModel):
    """Configure global/reactive mode for a destination modulator (prompt space)."""

    action: Literal["set_reactive_config"] = "set_reactive_config"
    space: Literal["latent", "prompt"]
    driver: DestinationDriverType = "global"
    stem: Optional[str] = None  # Required if driver="stem"
    blend_min: float = 0.0
    blend_max: float = 1.0

    # v4 dance model
    stage_left: Optional[float] = None
    stage_home: Optional[float] = None
    stage_right: Optional[float] = None
    position_source: Optional[PositionSourceType] = None
    intensity_source: Optional[IntensitySourceType] = None
    position_smoothing_ms: Optional[float] = None
    silence_behavior: Optional[SilenceBehaviorType] = None
    drift_ms: Optional[float] = None
    intensity_curve: Optional[IntensityCurveType] = None
    intensity_gamma: Optional[float] = None
    stem_rankings: Optional[dict[str, Optional[int]]] = None
    rank_weights: Optional[dict[str, float]] = None
    blend_slew_rate: Optional[float] = None  # Max blend change per second (0 = no limit)


# Discriminated union of all client messages
ClientMessage = Annotated[
    Union[
        # Playback control
        AudioTimeUpdate,
        AudioPlay,
        AudioPause,
        AudioSeek,
        Stop,
        StopGeneration,
        # SAE steering
        StartSAESteering,
        SetSteeringMode,
        UpdateBlockConfig,
        # Composition
        SetCompositionConfig,
        # Destination modulation
        SetDestination,
        FreezeBlend,
        SetBlendPosition,
        SetDestinationMode,
        SetReactiveConfig,
        SetDestinationLink,
    ],
    Field(discriminator="action"),
]


# ============================================================================
# Server → Client Messages (JSON only, binary frames sent separately)
# ============================================================================


class ErrorMessage(BaseModel):
    """Error occurred during processing."""

    type: Literal["error"] = "error"
    message: str


class TrackInfo(BaseModel):
    """Track metadata sent after audio upload/analysis."""

    type: Literal["track_info"] = "track_info"
    audio_id: str
    duration: float  # Total duration in seconds
    bpm: float  # Detected tempo
    stems: list[str]  # Available stems (including virtual)


class StemProminence(BaseModel):
    """Computed prominence for a single stem (from ProminenceEngine)."""

    prominence: float = Field(..., ge=0.0, le=1.0)  # Current computed prominence
    surprise_active: bool = False  # True when stem is temporarily promoted
    rank: Optional[RankType] = None  # User-assigned rank (for UI display)


class ExtendedStemActivity(BaseModel):
    """Extended activity data for all channels (for advanced UI).

    v3 (Dancer Ensemble): Now includes computed prominence values from
    ProminenceEngine, allowing the frontend to visualize which stems
    the system is currently focusing on.
    """

    type: Literal["extended_activity"] = "extended_activity"
    audio_time: float
    # Per-stem data with all channels
    stems: dict[str, dict[str, float]]  # {"drums": {"envelope": 0.5, "flash": 0.8, ...}}
    # Dancer Ensemble: computed prominence per stem (optional for backwards compat)
    prominence: Optional[dict[str, StemProminence]] = None  # {"drums": {prominence: 0.85, ...}}
    # Block-level activity (optional; used for UI physics sync)
    blocks: Optional[dict[str, dict[str, float]]] = None  # {"down.2.1": {"raw": 0.2, "physics": 0.3}}


class DestinationStatus(BaseModel):
    """Current state of destination modulator (for UI sync)."""

    type: Literal["destination_status"] = "destination_status"
    space: Literal["latent", "prompt"]
    destination_a: Optional[str] = None  # Label
    destination_b: Optional[str] = None  # Label
    blend_position: Annotated[float, Field(ge=0.0, le=1.0)]
    mode: Literal["slider", "reactive", "linked"]


class BlockConfigSnapshot(BaseModel):
    """Snapshot of a block's configuration for UI sync."""

    block: str
    link_target: LinkTargetType
    strength_min: float
    strength_max: float
    feature_id: int
    enabled: bool
    auto_config: bool
    sae_rank: Optional[RankType] = None
    spatial_mode: SpatialModeType
    spatial_mask: Optional[list[float]] = None  # 256 floats (16x16 grid)
    channel: ChannelType
    layer: LayerType
    physics_preset: PhysicsPresetType
    stage_left: Optional[float] = None
    stage_home: Optional[float] = None
    stage_right: Optional[float] = None
    position_source: Optional[PositionSourceType] = None
    intensity_source: Optional[IntensitySourceType] = None
    position_smoothing_ms: Optional[float] = None
    silence_behavior: Optional[SilenceBehaviorType] = None
    drift_ms: Optional[float] = None
    intensity_curve: Optional[IntensityCurveType] = None
    intensity_gamma: Optional[float] = None


class BlockConfigs(BaseModel):
    """Block config snapshot for frontend state sync."""

    type: Literal["block_configs"] = "block_configs"
    configs: dict[str, BlockConfigSnapshot]


