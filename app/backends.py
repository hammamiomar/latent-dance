"""Backend registry — one place where generation backends are declared.

A backend is a thing that produces images from signals. The audio half
(sampler → physics → prominence → steering) is a model-agnostic signal
factory; slots bind its outputs to the control inputs a backend declares
here. Adding a backend = one pipeline class + one strategy subclass + one
register_backend() call. Nothing else in the app should know mode names.

Capabilities are a control-input manifest first, UI hints second: the
frontend renders whatever the manifest declares (Phase 4), and the routing
layer only binds signals to inputs that exist.

Design doc: notes/design_docs/MULTI_BACKEND_SPEC.md (Phase 2) plus the
signal-contract note in notes/design_docs/PREFLIGHT_PLAN.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from app.strategies.base import GenerationStrategy
    from hambajuba2ba.config import PipelineConfig


# ---------------------------------------------------------------------------
# Capability manifest types
# ---------------------------------------------------------------------------

# Per-frame generators produce a fresh frame from signals each tick.
# Evolving-canvas backends advance a stateful optimization K steps per
# displayed frame; signals modulate the ongoing process (e.g. dinoDreamer).
TemporalContract = Literal["per_frame", "evolving_canvas"]

# What shape of value a control input accepts from the routing layer.
ControlKind = Literal["scalar", "id", "mask2d", "text", "event"]


@dataclass(frozen=True)
class ControlInput:
    """One named input a backend accepts from the signal-routing layer.

    Attributes:
        name: Dotted identifier, stable across releases (e.g. "slot.strength").
        kind: Value shape — scalar signal, discrete id, 2D mask, text, event.
        count: How many independent instances exist (e.g. one per slot).
        id_range: Inclusive (lo, hi) for kind="id" when the id space is bounded.
        shape: (H, W) for kind="mask2d".
        description: One line of human-readable semantics.
    """

    name: str
    kind: ControlKind
    count: int = 1
    id_range: Optional[tuple[int, int]] = None
    shape: Optional[tuple[int, int]] = None
    description: str = ""


@dataclass(frozen=True)
class SlotInfo:
    """Display metadata for one steering slot (mirrors frontend data/features.ts)."""

    name: str            # canonical id ("down.2.1", "slot_0", ...)
    display_name: str    # "Composition"
    short_name: str      # "COMP"
    color: str           # hex, muted earthy palette
    description: str = ""


@dataclass(frozen=True)
class BackendCapabilities:
    """What the active backend accepts and produces.

    Served at GET /api/capabilities and sent as the first WebSocket message,
    so the frontend can render slots/panels without hardcoding a mode.
    """

    mode: str
    temporal: TemporalContract
    slots: tuple[SlotInfo, ...]
    feature_id_range: tuple[int, int]   # inclusive
    feature_label: str                  # "Feature" | "Concept" | "Unit"
    spatial_mask_shape: tuple[int, int]
    has_prompts: bool
    has_destinations: bool
    output_resolution: tuple[int, int]  # (width, height); corrected from live config at startup
    control_inputs: tuple[ControlInput, ...] = ()

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload for the HTTP endpoint and WS hello message."""
        payload = asdict(self)
        payload["slot_count"] = self.slot_count
        return payload


@dataclass(frozen=True)
class BackendSpec:
    """Everything the app needs to boot and serve one backend."""

    mode: str
    mode_label: str                                        # human-readable, for logs
    pipeline_factory: Callable[["PipelineConfig"], Any]    # constructs (does not load)
    strategy_class: type["GenerationStrategy"]
    capabilities: BackendCapabilities = field(repr=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKENDS: dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    """Register a backend under its mode name. Duplicate modes are a bug."""
    if spec.mode in BACKENDS:
        raise ValueError(f"Backend mode {spec.mode!r} is already registered")
    if spec.mode != spec.capabilities.mode:
        raise ValueError(
            f"BackendSpec mode {spec.mode!r} != capabilities.mode "
            f"{spec.capabilities.mode!r}"
        )
    BACKENDS[spec.mode] = spec


def get_backend(mode: str) -> BackendSpec:
    """Look up a registered backend, failing loudly with the known modes."""
    spec = BACKENDS.get(mode)
    if spec is None:
        raise ValueError(
            f"Unknown mode: {mode!r}. Registered backends: "
            + ", ".join(sorted(BACKENDS))
        )
    return spec


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------

# SAE slot display data mirrors frontend/src/data/features.ts (BLOCKS,
# BLOCK_NAMES, BLOCK_COLORS) — one source of truth once Phase 4 reads this.
_SAE_SLOTS = (
    SlotInfo("down.2.1", "Composition", "COMP", "#c45a2a", "Scene structure, mood, intensity"),
    SlotInfo("mid.0", "Abstract", "ABS", "#7a5090", "Global effects, distortion"),
    SlotInfo("up.0.0", "Details", "DET", "#b85a7a", "Expressions, objects"),
    SlotInfo("up.0.1", "Style", "STY", "#5a8a4a", "Patterns, textures"),
)

_SAE_CONTROL_INPUTS = (
    ControlInput(
        "slot.feature", "id", count=4, id_range=(0, 5119),
        description="SAE feature steered by each slot (decoder column index)",
    ),
    ControlInput(
        "slot.strength", "scalar", count=4,
        description="Audio-driven steering amplitude per slot",
    ),
    ControlInput(
        "slot.spatial_mask", "mask2d", count=4, shape=(16, 16),
        description="Where in the image each slot's feature applies",
    ),
    ControlInput(
        "prompt.destination", "text", count=2,
        description="Prompt destinations A/B for prompt-space SLERP",
    ),
    ControlInput(
        "prompt.blend", "scalar",
        description="SLERP position between prompt destinations",
    ),
    ControlInput(
        "composition.seed", "id", count=2,
        description="Noise seeds A/B for the composition circular walk",
    ),
    ControlInput(
        "composition.distance", "scalar",
        description="Radius of the noise circular walk",
    ),
)

SAE_CAPABILITIES = BackendCapabilities(
    mode="sae_steering",
    temporal="per_frame",
    slots=_SAE_SLOTS,
    feature_id_range=(0, 5119),
    feature_label="Feature",
    spatial_mask_shape=(16, 16),
    has_prompts=True,
    has_destinations=True,
    output_resolution=(512, 512),
    control_inputs=_SAE_CONTROL_INPUTS,
)


def _register_builtin_backends() -> None:
    # Imports are function-local: the strategy/pipeline chain pulls in torch
    # and diffusers, and keeping it here avoids import cycles with
    # app.strategies (which looks backends up lazily inside create_strategy).
    from app.strategies.sae_steering_strategy import SAESteeringStrategy
    from hambajuba2ba.generation.pipeline import SAESteerablePipeline

    register_backend(
        BackendSpec(
            mode="sae_steering",
            mode_label="SDXL-Turbo + SAE steering",
            pipeline_factory=SAESteerablePipeline,
            strategy_class=SAESteeringStrategy,
            capabilities=SAE_CAPABILITIES,
        )
    )


_register_builtin_backends()
