"""Mock backend — synthetic frames, six slots, zero GPU and zero weights.

`HAMBA_MODE=mock uv run hambajuba` boots the full app (manifest, WebSocket,
config ACKs) on any machine: the frontend renders six orbs purely from this
manifest, with no frontend edits — rendering is capability-driven. It is
also the permanent frontend-dev backend.

Frames are drawn on the CPU: one horizontal band per slot, lit by that
slot's physics-smoothed activity, plus a time sweep so motion is always
visible. With a song uploaded, the entire real signal chain runs (sampler →
physics → prominence → telemetry) — only the image model is fake, so what
you see IS the steering signal the real backends receive.

The mock declares prompts/destinations for UI parity but has no prompt
space; destination messages are accepted and ignored.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import torch
from pydantic import BaseModel

from app.backends import BackendCapabilities, ControlInput, SlotInfo
from app.strategies.base import GenerationStrategy
from app.strategies.handlers import (
    handle_audio_message,
    handle_modulation_message,
    handle_slot_message,
)
from app.strategies.handlers.audio_handlers import is_audio_message
from app.strategies.handlers.destination_handlers import is_destination_message
from app.strategies.handlers.modulation_handlers import is_modulation_message
from app.strategies.handlers.slot_handlers import is_slot_message
from hambajuba2ba.config.slots import BlockLinkConfig

logger = logging.getLogger("uvicorn")


# ---------------------------------------------------------------------------
# Capabilities — six slots so N-slot rendering is exercised beyond SAE's four
# ---------------------------------------------------------------------------

_MOCK_SLOTS = (
    SlotInfo("slot_0", "Pulse", "PLS", "#b8863a", "Synthetic band 0"),
    SlotInfo("slot_1", "Drift", "DRF", "#4a8a6a", "Synthetic band 1"),
    SlotInfo("slot_2", "Strike", "STK", "#b85a5a", "Synthetic band 2"),
    SlotInfo("slot_3", "Shimmer", "SHM", "#6a7aaa", "Synthetic band 3"),
    SlotInfo("slot_4", "Ground", "GND", "#8a6a4a", "Synthetic band 4"),
    SlotInfo("slot_5", "Air", "AIR", "#6a9098", "Synthetic band 5"),
)

# Deliberately NOT the SAE range: exercises manifest-driven feature bounds
# in the frontend (FeaturePicker degrades to a 0–999 ID spinner: no labels).
_MOCK_FEATURE_RANGE = (0, 999)

MOCK_CAPABILITIES = BackendCapabilities(
    mode="mock",
    temporal="per_frame",
    slots=_MOCK_SLOTS,
    feature_id_range=_MOCK_FEATURE_RANGE,
    feature_label="Feature",
    spatial_mask_shape=(16, 16),
    has_prompts=True,
    has_destinations=True,
    output_resolution=(512, 512),
    control_inputs=(
        ControlInput(
            "slot.feature", "id", count=6, id_range=_MOCK_FEATURE_RANGE,
            description="Synthetic feature id per slot (tints nothing; echoed in ACKs)",
        ),
        ControlInput(
            "slot.strength", "scalar", count=6,
            description="Audio-driven amplitude per slot (lights its band)",
        ),
        ControlInput(
            "slot.spatial_mask", "mask2d", count=6, shape=(16, 16),
            description="Accepted for UI parity; the mock renders bands, not masks",
        ),
        ControlInput(
            "prompt.destination", "text", count=2,
            description="Accepted for UI parity; the mock has no prompt space",
        ),
    ),
)

# Default link targets cycle the same musical spread the frontend seeds:
# foundation, voice, hits, air — then wrap for slots 5 and 6.
_SLOT_SEEDS = (
    ("bass", 1),
    ("vocals", 2),
    ("drums", 1),
    ("other_high", None),
)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = int(hex_color.lstrip("#"), 16)
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


_SLOT_RGB = tuple(_hex_to_rgb(slot.color) for slot in _MOCK_SLOTS)


# ---------------------------------------------------------------------------
# Frame synthesis — pure function so tests can pin it without a strategy
# ---------------------------------------------------------------------------

def render_mock_frame(
    width: int,
    height: int,
    colors: tuple[tuple[int, int, int], ...],
    activities: np.ndarray,
    enabled: np.ndarray,
    t: float,
) -> np.ndarray:
    """Draw one synthetic frame: a horizontal band per slot plus a time sweep.

    Enabled slots glow between 25% and 100% of their color with activity;
    disabled slots stay a faint 5% outline so the layout is always legible.

    Args:
        colors: (N, 3) slot colors, 0–255, in slot order.
        activities: (N,) physics activity per slot, clipped to [0, 1].
        enabled: (N,) bool mask.
        t: wall-clock seconds; drives the sweep column.

    Returns:
        (height, width, 3) uint8 RGB frame.
    """
    color_arr = np.asarray(colors, dtype=np.float32)
    act = np.clip(np.asarray(activities, dtype=np.float32), 0.0, 1.0)
    brightness = np.where(np.asarray(enabled, dtype=bool), 0.25 + 0.75 * act, 0.05)

    # Each row belongs to one slot band; index rows → slots, then broadcast
    # across the width. No per-pixel Python.
    n = len(color_arr)
    row_slot = np.linspace(0, n, height, endpoint=False).astype(np.intp)
    rows = color_arr[row_slot] * brightness[row_slot, None]          # (H, 3)
    frame = np.broadcast_to(rows[:, None, :], (height, width, 3)).copy()

    # Sweep column crossing every 4s — visible motion even with all slots off
    sweep = int((t % 4.0) / 4.0 * width)
    frame[:, sweep : min(sweep + 3, width), :] += 90.0

    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pipeline + strategy
# ---------------------------------------------------------------------------

class MockPipeline:
    """No-op pipeline satisfying the lifespan/strategy contract on CPU.

    engine=None deliberately skips CompositionEngine init (the mock draws
    its own frames, there is no latent space to walk).
    """

    def __init__(self, config: Any):
        self.config = config
        self.device = "cpu"
        self.engine = None
        self.steering_manager = None

    def load(self) -> None:
        logger.info("Mock backend: nothing to load (synthetic frames)")

    def cleanup(self) -> None:
        pass


class MockStrategy(GenerationStrategy):
    """Synthetic-frame strategy: real signal chain, fake image model.

    Reuses the shared audio/slot/modulation handlers and the base frame
    loop; only _gpu_forward differs (CPU band rendering). Starts with the
    standard start message — the manager dispatches any StartGeneration.
    """

    def _init_default_slot_configs(self) -> None:
        rng = np.random.default_rng()
        lo, hi = _MOCK_FEATURE_RANGE
        for i, slot in enumerate(_MOCK_SLOTS):
            link_target, sae_rank = _SLOT_SEEDS[i % len(_SLOT_SEEDS)]
            self.slot_configs[slot.name] = BlockLinkConfig(
                block=slot.name,
                feature_id=int(rng.integers(lo, hi + 1)),
                link_target=link_target,
                sae_rank=sae_rank,
            )

    async def _setup_backend(self, params: BaseModel, cached: dict) -> None:
        await self._send_config_snapshots()

    def _is_ready(self) -> bool:
        return True

    def _pre_generate(self, audio_time: float, dt: float) -> None:
        return None

    def _gpu_forward(
        self, steering: dict, pre_ctx: Any, audio_time: float = 0.0
    ) -> Optional[tuple]:
        """Draw the band frame from the physics activity the base computed."""
        t_infer = time.perf_counter()

        configs = self._get_slot_configs()
        activities = np.array(
            [
                self._last_block_activity.get(name, {}).get("physics", 0.0)
                for name in configs
            ],
            dtype=np.float32,
        )
        enabled = np.array([cfg.enabled for cfg in configs.values()], dtype=bool)

        frame = render_mock_frame(
            self.config.width, self.config.height,
            _SLOT_RGB, activities, enabled, t_infer,
        )
        infer_ms = (time.perf_counter() - t_infer) * 1000

        # Already (H, W, 3) uint8 on CPU — no D2H, no prep
        return torch.from_numpy(frame), 0.0, infer_ms, 0.0

    async def handle_message(self, message: BaseModel) -> Optional[dict]:
        if is_audio_message(message):
            return handle_audio_message(self, message)

        if is_slot_message(message):
            response = handle_slot_message(self, message)
            if response is not None:
                return response
            await self._send_config_snapshots()
            return None

        if is_modulation_message(message):
            return handle_modulation_message(self, message)

        if is_destination_message(message):
            # Accepted for UI parity (the manifest declares destinations so
            # the frontend renders identically); nothing to modulate here.
            logger.info(f"Mock backend: ignoring {type(message).__name__}")
            return None

        logger.warning(f"Unhandled message type: {type(message).__name__}")
        return None
