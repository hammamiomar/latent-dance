"""The per-slot steering contract — shared by every generation backend.

A slot binds one audio-derived signal to one backend control input:
SAE blocks today, RA-SAE concept slots and dinoDreamer objective terms
tomorrow (see app/backends.py for the control-input manifest side).
Historically this lived in audio/focus_config.py with SAE naming; the
class is still called BlockLinkConfig and its id field `block`, while
the wire speaks `slot` (with `block` accepted as the legacy alias).

When auto_config=True on a BlockLinkConfig, the system derives:
- Channel: transient vs energy_smooth based on percussive confidence
- Layer: flash vs sustain vs combined based on role
- Spatial: floor/ceiling/center based on frequency range
- Physics: Already handled by BlendedPhysics

This reduces user burden by providing sensible defaults based on
what kind of audio source they've linked.

Ranking System (Dancer Ensemble Architecture):
Instead of static focus_weight, users assign ranks (1-4 or None for auto).
The ProminenceEngine then computes dynamic prominence based on musical context.

Rank Semantics:
    Rank 1: Main dancer(s) - primary visual focus
    Rank 2: Backup dancer(s) - supporting, visible
    Rank 3: Background - ambient presence
    Rank 4: Barely there - subtle texture
    None:   Auto/available - can be promoted on surprise moments

Per-Axis Rankings:
Each stem can have different rankings for different control axes:
    - sae_rank: Controls SAE feature steering
    - slerp_latent_rank: Controls latent space SLERP blend
    - slerp_prompt_rank: Controls prompt space SLERP blend

Design:
- FocusConfig is the computed output from classification
- derive_focus_config() takes a classification and returns FocusConfig
- Manual overrides are still possible by setting auto_config=False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from hambajuba2ba.audio.classification import ComponentClassification

# Spatial mode: draw (user-painted 16x16 grid) or pitch_aligned (auto from audio)
SpatialModeType = Literal["draw", "pitch_aligned"]

# Shared dance model defaults (used by both BlockLinkConfig and ReactiveConfig)
DANCE_MODEL_DEFAULTS: dict[str, float | str] = {
    "stage_left": -30.0,
    "stage_home": 0.0,
    "stage_right": 30.0,
    "position_source": "auto",
    "intensity_source": "energy_smooth",
    "position_smoothing_ms": 50.0,
    "silence_behavior": "hold_last",
    "drift_ms": 1500.0,
    "intensity_curve": "linear",
    "intensity_gamma": 1.0,
}


@dataclass
class FocusConfig:
    """Derived configuration from ComponentClassification.

    All fields have sensible defaults that can be used even without
    a classification available.

    Attributes:
        channel: Which feature channel to sample (energy_smooth, transient, etc.)
        layer: Which response layer (flash, sustain, combined)
        spatial_preset: Which spatial mask to use (floor, ceiling, center, global)
    """

    channel: str = "energy_smooth"
    layer: str = "combined"
    spatial_preset: str = "global"


def derive_focus_config(classification: "ComponentClassification | None") -> FocusConfig:
    """Derive focus configuration from component classification.

    Logic:
        Channel:
            - percussive > 0.6 → transient (captures attack transients)
            - else → energy_smooth (captures sustained energy)

        Layer:
            - rhythm > 0.6 → flash (fast, punchy response)
            - harmony > 0.5 → sustain (slow, trailing glow)
            - else → combined (balanced)

        Spatial:
            - is_bass → floor (low frequency feels grounded)
            - is_high → ceiling (high frequency feels airy)
            - melody > 0.4 → center (melodic content is focal)
            - else → global (applies everywhere)

    Args:
        classification: ComponentClassification or None

    Returns:
        FocusConfig with derived settings
    """
    if classification is None:
        return FocusConfig()

    # Channel selection: percussive → transient, else → energy_smooth
    if classification.percussive_confidence > 0.6:
        channel = "transient"
    else:
        channel = "energy_smooth"

    # Layer selection: rhythm → flash, harmony → sustain, else → combined
    if classification.rhythm_confidence > 0.6:
        layer = "flash"
    elif classification.harmony_confidence > 0.5:
        layer = "sustain"
    else:
        layer = "combined"

    # Spatial selection based on frequency range and melodic content
    if classification.is_bass:
        spatial = "floor"
    elif classification.is_high:
        spatial = "ceiling"
    elif classification.melody_confidence > 0.4:
        spatial = "center"
    else:
        spatial = "global"

    return FocusConfig(
        channel=channel,
        layer=layer,
        spatial_preset=spatial,
    )


# Valid rank values for type hints and validation
VALID_RANKS = (1, 2, 3, 4, None)


@dataclass
class BlockLinkConfig:
    """Configuration for linking an SAE block to an audio source.

    Replaces the old StemFeatureMapping with:
    - LinkTarget instead of stem name
    - StrengthRange instead of sensitivity
    - Ranking system instead of static focus_weight

    The ranking system (Dancer Ensemble Architecture):
    - User picks WHAT to focus on (ranks 1-4 or None for auto)
    - System computes HOW MUCH via ProminenceEngine
    - Enables surprise moments where unranked stems get promoted

    Note: SLERP rankings are configured per-destination in ReactiveConfig,
    not per-block. This keeps SAE steering separate from destination control.

    Field groups by subsystem:

    -- Core identity (used by all subsystems) --
        block: SAE block name (down.2.1, mid.0, up.0.0, etc.)
        feature_id: SAE feature index to steer
        link_target: Which audio source drives this block
        enabled: Whether this block is actively steered
        auto_config: If True, derive channel/layer/spatial from classification

    -- SAE steering (SteeringComputation, ProminenceEngine) --
        strength_min: Floor strength when physics = 0
        strength_max: Ceiling strength when physics = 1
        sae_rank: Rank for SAE steering (1-4 or None for auto)
        channel: Feature channel to sample (when auto_config=False)
        layer: Response layer flash/sustain/combined (when auto_config=False)

    -- Spatial masks (SpatialManager) --
        spatial_mode: draw (user-painted 16x16 grid) or pitch_aligned (auto from audio)
        spatial_mask: 256 floats representing a 16x16 binary mask (None = uniform)

    -- Physics simulation (PhysicsManager) --
        physics_preset: Physics simulation preset (ambient, punchy, etc.)

    -- Dance model (LEGACY — no runtime reader on this class) --
        The stage/position machinery runs on the destination system's
        ReactiveConfig (bridge/destinations.py). These fields are still
        accepted and stored for protocol compatibility (Hermes plans and
        older clients send them) but nothing samples them per-frame.
        stage_left: Left bound of position range (degrees)
        stage_home: Center/rest position (degrees)
        stage_right: Right bound of position range (degrees)
        position_source: Which audio feature drives position ("auto" or explicit)
        intensity_source: Which audio feature drives intensity
        position_smoothing_ms: Smoothing window for position updates
        silence_behavior: What to do when audio is silent ("hold_last", etc.)
        drift_ms: Time to drift back to home position during silence
        intensity_curve: Mapping curve for intensity ("linear", "exponential", etc.)
        intensity_gamma: Gamma exponent for intensity curve
    """

    # ── Core identity (used by all subsystems) ──────────────────────────
    block: str
    feature_id: int
    link_target: str = "bass"
    enabled: bool = False
    auto_config: bool = True

    # ── SAE steering (SteeringComputation, ProminenceEngine) ────────────
    strength_min: float = 0.0
    strength_max: float = 15.0
    sae_rank: int | None = None  # Rank 1-4, or None for auto/surprise promotion

    # Manual overrides for SAE (used when auto_config=False)
    channel: str = "energy_smooth"
    layer: str = "combined"

    # ── Spatial masks (SpatialManager) ──────────────────────────────────
    spatial_mode: SpatialModeType = "draw"
    spatial_mask: list[float] | None = None  # 256 floats (16x16 grid), None = uniform (all 1s)

    # ── Physics simulation (PhysicsManager) ─────────────────────────────
    physics_preset: str = "ambient"

    # ── Destination / position control (PositionFollower, DestinationHandlers) ──
    # Defaults sourced from DANCE_MODEL_DEFAULTS for consistency with ReactiveConfig
    stage_left: float = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["stage_left"])
    stage_home: float = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["stage_home"])
    stage_right: float = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["stage_right"])
    position_source: str = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["position_source"])
    intensity_source: str = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["intensity_source"])
    position_smoothing_ms: float = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["position_smoothing_ms"])
    silence_behavior: str = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["silence_behavior"])
    drift_ms: float = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["drift_ms"])
    intensity_curve: str = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["intensity_curve"])
    intensity_gamma: float = field(default_factory=lambda: DANCE_MODEL_DEFAULTS["intensity_gamma"])

    def __post_init__(self):
        """Validate rank values."""
        if self.sae_rank is not None and self.sae_rank not in (1, 2, 3, 4):
            raise ValueError(
                f"sae_rank must be 1, 2, 3, 4, or None (got {self.sae_rank})"
            )

    def get_effective_config(
        self,
        classification: "ComponentClassification | None" = None,
    ) -> FocusConfig:
        """Get the effective focus configuration.

        If auto_config is True, derives from classification.
        Otherwise, uses the manually-set values.

        Args:
            classification: ComponentClassification for the linked stem

        Returns:
            FocusConfig with effective settings
        """
        if self.auto_config and classification is not None:
            return derive_focus_config(classification)
        else:
            return FocusConfig(
                channel=self.channel,
                layer=self.layer,
            )


# Default block configurations (used when no user config exists)
# Default ranking: drums and bass are main dancers (1), vocals backup (2), other auto
DEFAULT_BLOCK_CONFIGS = {
    "down.2.1": BlockLinkConfig(
        block="down.2.1",
        feature_id=0,
        link_target="bass",
        strength_min=-30.0,
        strength_max=30.0,
        sae_rank=1,  # Main dancer - bass drives visual intensity
    ),
    "mid.0": BlockLinkConfig(
        block="mid.0",
        feature_id=0,
        link_target="vocals",
        strength_min=-30.0,
        strength_max=30.0,
        sae_rank=2,  # Backup dancer
    ),
    "up.0.0": BlockLinkConfig(
        block="up.0.0",
        feature_id=0,
        link_target="drums",
        strength_min=-30.0,
        strength_max=30.0,
        sae_rank=1,  # Main dancer - drums drive punchy response
    ),
    "up.0.1": BlockLinkConfig(
        block="up.0.1",
        feature_id=0,
        link_target="other_high",
        strength_min=-30.0,
        strength_max=30.0,
        sae_rank=None,  # Auto - available for surprise moments
    ),
}


def get_base_stem(link_target: str) -> str | None:
    """Extract base stem name from a link target string.

    Maps compound link targets (e.g. "drums_low", "bass_harmonic")
    back to their physical stem ("drums", "bass"). Returns None for
    derived targets ("tension", "global"); unrecognized targets pass
    through unchanged.
    """
    if link_target in ("tension", "global"):
        return None
    for base in ("drums", "vocals", "bass", "other"):
        if link_target.startswith(base):
            return base
    return link_target
