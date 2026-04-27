"""Destination modulation: SLERP travel between prompt embedding points.

Driven by tonal distance for semantic scene transitions.
CompositionEngine (noise circular walk) lives in bridge/composition.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import logging
from typing import Dict, Literal, Optional, Tuple

import torch

from hambajuba2ba.audio.focus_config import DANCE_MODEL_DEFAULTS
from hambajuba2ba.bridge.physics import SteeringPhysics, get_physics_preset

logger = logging.getLogger(__name__)


def slerp_inplace(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float,
    out: torch.Tensor,
    v0_norm_buf: torch.Tensor,
    v1_norm_buf: torch.Tensor,
    temp_buf: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SLERP with pre-allocated buffers for torch.compile compatibility.

    Uses in-place operations to avoid tensor allocations in the hot path.
    All buffers must be pre-allocated flat tensors of size v0.numel().

    Args:
        v0: Start point (any shape)
        v1: End point (same shape as v0)
        t: Interpolation factor (0.0 = v0, 1.0 = v1)
        out: Pre-allocated output buffer (same shape as v0/v1)
        v0_norm_buf: Pre-allocated buffer for normalized v0 (flat)
        v1_norm_buf: Pre-allocated buffer for normalized v1 (flat)
        temp_buf: Pre-allocated temp buffer for intermediate results (flat)
        eps: Small value for numerical stability

    Returns:
        out tensor with interpolated result
    """
    # Fast path for edge cases
    if t <= 0.0:
        out.copy_(v0)
        return out
    if t >= 1.0:
        out.copy_(v1)
        return out

    # Views for computation (no allocation)
    v0_flat = v0.view(-1)
    v1_flat = v1.view(-1)
    out_flat = out.view(-1)

    # Normalize v0 into buffer
    v0_norm_buf.copy_(v0_flat)
    v0_norm_buf.div_(v0_norm_buf.norm() + eps)

    # Normalize v1 into buffer
    v1_norm_buf.copy_(v1_flat)
    v1_norm_buf.div_(v1_norm_buf.norm() + eps)

    # Dot product (scalar tensor - unavoidable but tiny)
    dot = torch.clamp(torch.dot(v0_norm_buf, v1_norm_buf), -1.0, 1.0)

    # Compute angle and sin (scalar tensors)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega) + eps

    # SLERP coefficients
    s0 = torch.sin((1.0 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega

    # Compute SLERP result directly into out: out = s0*v0 + s1*v1
    # No LERP fallback needed - sin_omega has eps for numerical stability,
    # and at dot≈1.0 SLERP converges to LERP anyway.
    out_flat.copy_(v0_flat)
    out_flat.mul_(s0)
    # Can't use add_(alpha=s1) because s1 is a tensor; use addcmul pattern
    out_flat.addcmul_(v1_flat, s1.expand_as(v1_flat))

    return out


@dataclass
class Destination:
    """A point in the space to travel toward.

    Represents a meaningful destination in latent or embedding space.
    Contains the actual tensor plus metadata for display/debugging.

    For prompt space, SDXL requires both prompt_embeds and pooled_prompt_embeds.
    Store both using tensor (main) and tensor_pooled (secondary).
    """

    tensor: torch.Tensor  # The actual embedding/latent (or prompt_embeds for prompt space)
    label: str  # User-facing name ("album art", "seed 42", "cyberpunk")

    # For SDXL prompt space: stores pooled_prompt_embeds
    tensor_pooled: Optional[torch.Tensor] = None

    # For latent space destinations:
    seed: Optional[int] = None

    # For prompt space destinations:
    prompt: Optional[str] = None


@dataclass(frozen=True)
class ReactiveConfig:
    """Configuration for reactive mode.

    Controls how audio drives the blend position.
    """

    driver: Literal["global", "stem"] = "global"
    stem: Optional[str] = None  # If driver="stem"
    blend_min: float = 0.0
    blend_max: float = 1.0

    # v4 dance model (defaults from shared DANCE_MODEL_DEFAULTS)
    stage_left: float = DANCE_MODEL_DEFAULTS["stage_left"]
    stage_home: float = DANCE_MODEL_DEFAULTS["stage_home"]
    stage_right: float = DANCE_MODEL_DEFAULTS["stage_right"]
    position_source: str = DANCE_MODEL_DEFAULTS["position_source"]
    intensity_source: str = DANCE_MODEL_DEFAULTS["intensity_source"]
    position_smoothing_ms: float = DANCE_MODEL_DEFAULTS["position_smoothing_ms"]
    silence_behavior: Literal["drift_center", "hold_last"] = DANCE_MODEL_DEFAULTS["silence_behavior"]
    drift_ms: float = DANCE_MODEL_DEFAULTS["drift_ms"]
    intensity_curve: Literal["linear", "gamma", "impulse", "clip"] = DANCE_MODEL_DEFAULTS["intensity_curve"]
    intensity_gamma: float = DANCE_MODEL_DEFAULTS["intensity_gamma"]

    # v4 global weights
    stem_rankings: dict[str, Optional[int]] | None = None
    rank_weights: dict[str, float] | None = None


class PositionFollower:
    """Smoothly follows target position, with optional drift on silence."""

    def __init__(
        self,
        position: float = 0.5,
        smoothing_ms: float = 50.0,
        drift_ms: float = 1500.0,
    ) -> None:
        self.position = position
        self.smoothing_ms = smoothing_ms
        self.drift_ms = drift_ms

    def _alpha(self, dt: float, ms: float) -> float:
        if ms <= 0:
            return 1.0
        tau = ms / 1000.0
        return 1.0 - math.exp(-dt / tau)

    def step(self, target: Optional[float], dt: float, silence_behavior: str) -> float:
        if target is None:
            if silence_behavior == "drift_center":
                alpha = self._alpha(dt, self.drift_ms)
                self.position += alpha * (0.5 - self.position)
            return self.position

        alpha = self._alpha(dt, self.smoothing_ms)
        self.position += alpha * (target - self.position)
        return self.position


def apply_intensity_curve(
    value: float,
    curve: str,
    gamma: float,
    dt: float,
    physics: Optional[SteeringPhysics],
) -> Tuple[float, Optional[SteeringPhysics]]:
    """Apply intensity curve, optionally using physics for impulse behavior."""
    if curve == "gamma":
        return max(0.0, min(1.0, value)) ** max(0.01, gamma), physics
    if curve == "clip":
        return max(0.0, min(1.0, value * 1.5)), physics
    if curve == "impulse":
        if physics is None:
            physics = SteeringPhysics(get_physics_preset("drums"))
        return max(0.0, min(1.0, physics.step(value, dt))), physics
    # linear (default)
    return max(0.0, min(1.0, value)), physics


class DestinationModulator:
    """SLERP-based modulation between two destinations.

    Works identically for latent space and prompt embedding space.
    Supports two modes:
    - slider: User controls crossfader directly
    - reactive: Audio drives blend position via physics

    Usage:
        modulator = DestinationModulator(device, dtype, bpm=120, fps=60)
        modulator.load_a(Destination(tensor=latent_a, label="Album Art"))
        modulator.load_b(Destination(tensor=latent_b, label="Seed 42"))

        # Slider mode
        modulator.set_blend(0.5)
        result = modulator.step(blend_position=0.5)

        # Reactive mode (blend position computed externally by strategy)
        modulator.set_mode("reactive", ReactiveConfig(physics_preset="bass"))
        result = modulator.step(blend_position=computed_blend)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        bpm: float = 120.0,
        fps: float = 60.0,
        blend_slew_rate: float = 1.5,
    ):
        """Initialize destination modulator.

        Args:
            device: Torch device for tensors
            dtype: Torch dtype for tensors
            bpm: Beats per minute for physics scaling
            fps: Frames per second
            blend_slew_rate: Max blend position change per second (units/s).
                Forces the blend to travel through intermediate SLERP positions
                so you see the morph between prompts. 1.5 = full 0→1 in ~0.67s.
                0 = no limit (raw, instant jumps).
        """
        self.device = device
        self.dtype = dtype
        self.bpm = bpm
        self.fps = fps
        self.blend_slew_rate = blend_slew_rate

        # Destinations
        self.destination_a: Optional[Destination] = None
        self.destination_b: Optional[Destination] = None

        # Blend state
        self.blend_position: float = 0.0  # 0.0 = A, 1.0 = B

        # Mode (default to reactive to match frontend)
        self.mode: Literal["slider", "reactive", "linked"] = "reactive"
        self.reactive_config: ReactiveConfig = ReactiveConfig()

        # Link target for "linked" mode (audio-driven SLERP)
        self.link_target: Optional[str] = None

        # Physics (for reactive mode, created lazily)
        self._physics: Optional[SteeringPhysics] = None

        # Position follower for dance motion
        self._position_follower: Optional[PositionFollower] = None

        # Pre-allocated buffers for SLERP results (torch.compile compatibility)
        # These ensure same tensor address every frame, enabling CUDA graph optimization
        self._result_buffer: Optional[torch.Tensor] = None  # For latent space
        self._embeds_buffer: Optional[torch.Tensor] = None  # For prompt embeds
        self._pooled_buffer: Optional[torch.Tensor] = None  # For pooled embeds

        # SLERP intermediate buffers (allocated lazily on first destination load)
        self._slerp_v0_norm: Optional[torch.Tensor] = None
        self._slerp_v1_norm: Optional[torch.Tensor] = None
        self._slerp_temp: Optional[torch.Tensor] = None
        self._slerp_pooled_v0: Optional[torch.Tensor] = None
        self._slerp_pooled_v1: Optional[torch.Tensor] = None
        self._slerp_pooled_temp: Optional[torch.Tensor] = None

    def load_a(self, dest: Destination) -> None:
        """Load destination A.

        Args:
            dest: Destination to load
        """
        self.destination_a = dest
        self._ensure_buffer()
        self._log_latent_stats("A", dest)

    def load_b(self, dest: Destination) -> None:
        """Load destination B.

        Args:
            dest: Destination to load
        """
        self.destination_b = dest
        self._ensure_buffer()
        self._log_latent_stats("B", dest)

        if (
            self.destination_a is not None
            and self.destination_a.tensor_pooled is None
            and self.destination_b.tensor_pooled is None
        ):
            try:
                diff = (self.destination_b.tensor - self.destination_a.tensor).float()
                diff_std = float(diff.std().item())
                logger.info("Destination latents diff std=%.5f", diff_std)
            except Exception:
                logger.debug("Failed to compute latent diff stats", exc_info=True)

    def _ensure_buffer(self) -> None:
        """Ensure result buffers are allocated with correct shapes.

        For torch.compile compatibility, we pre-allocate buffers so that
        SLERP always returns the same tensor address (enabling CUDA graphs).
        """
        dest = self.destination_a or self.destination_b
        if dest is None:
            return

        # Main tensor buffer (latent or prompt_embeds)
        if self._result_buffer is None:
            self._result_buffer = torch.zeros_like(dest.tensor)

        # SLERP intermediate buffers for main tensor (flat shape)
        # Used by both latent space and prompt space
        flat_size = dest.tensor.numel()
        if self._slerp_v0_norm is None:
            device, dtype = dest.tensor.device, dest.tensor.dtype
            self._slerp_v0_norm = torch.zeros(flat_size, device=device, dtype=dtype)
            self._slerp_v1_norm = torch.zeros(flat_size, device=device, dtype=dtype)
            self._slerp_temp = torch.zeros(flat_size, device=device, dtype=dtype)

        # For prompt space: also need embeds and pooled buffers
        if dest.tensor_pooled is not None:
            if self._embeds_buffer is None:
                self._embeds_buffer = torch.zeros_like(dest.tensor)
            if self._pooled_buffer is None:
                self._pooled_buffer = torch.zeros_like(dest.tensor_pooled)

            # SLERP intermediate buffers for pooled_embeds (different size)
            pooled_flat = dest.tensor_pooled.numel()
            if self._slerp_pooled_v0 is None:
                device, dtype = dest.tensor_pooled.device, dest.tensor_pooled.dtype
                self._slerp_pooled_v0 = torch.zeros(pooled_flat, device=device, dtype=dtype)
                self._slerp_pooled_v1 = torch.zeros(pooled_flat, device=device, dtype=dtype)
                self._slerp_pooled_temp = torch.zeros(pooled_flat, device=device, dtype=dtype)

    def _log_latent_stats(self, slot: str, dest: Destination) -> None:
        if dest.tensor_pooled is not None:
            return
        try:
            mean = float(dest.tensor.mean().item())
            std = float(dest.tensor.std().item())
            seed = dest.seed if dest.seed is not None else "n/a"
            logger.info("Destination %s latent: seed=%s mean=%.4f std=%.4f", slot, seed, mean, std)
        except Exception:
            logger.debug("Failed to log latent stats", exc_info=True)

    def set_blend(self, position: float) -> None:
        """Set blend position (slider mode).

        Args:
            position: Blend position (0.0 = A, 1.0 = B)
        """
        self.blend_position = max(0.0, min(1.0, position))

    def set_mode(
        self,
        mode: Literal["slider", "reactive", "linked"],
        config: Optional[ReactiveConfig] = None,
    ) -> None:
        """Set modulation mode.

        Args:
            mode: "slider", "reactive", or "linked"
            config: ReactiveConfig for reactive mode
        """
        self.mode = mode
        if config is not None:
            self.reactive_config = config
        self._ensure_followers()

    def set_link_target(self, target: str) -> None:
        """Set link target and switch to linked mode.

        Args:
            target: LinkTarget value (e.g., "tension", "bass", "drums_percussive")
        """
        self.link_target = target
        self.mode = "linked"
        self._ensure_followers()

    def clear_link_target(self) -> None:
        """Clear link target and switch back to slider mode."""
        self.link_target = None
        self.mode = "slider"

    def _slew_blend(self, raw_blend: float, dt: float) -> float:
        """Slew rate limiter on blend position.

        Limits how fast the blend can change per frame, forcing it to travel
        through intermediate SLERP positions. The blend WILL reach the target —
        it just can't jump there instantly. You see the morph.
        """
        if self.blend_slew_rate <= 0:
            return raw_blend
        max_delta = self.blend_slew_rate * dt
        delta = raw_blend - self.blend_position
        clamped = max(-max_delta, min(max_delta, delta))
        return self.blend_position + clamped

    def update_from_link(
        self,
        position: Optional[float],
        intensity: float,
        dt: float,
        is_silent: bool,
    ) -> None:
        """Update blend position from linked audio values (dance model)."""
        if self.mode != "linked" or self.link_target is None:
            return

        cfg = self.reactive_config
        self._ensure_followers()

        target = None if is_silent else position
        pos = self._position_follower.step(target, dt, cfg.silence_behavior)

        intensity_value, self._physics = apply_intensity_curve(
            intensity, cfg.intensity_curve, cfg.intensity_gamma, dt, self._physics
        )

        raw_blend = self._apply_stage(pos, intensity_value, cfg)
        self.blend_position = self._slew_blend(raw_blend, dt)

    def update_from_reactive(
        self,
        position: Optional[float],
        intensity: float,
        dt: float,
        is_silent: bool,
    ) -> None:
        """Update blend position from global/reactive values."""
        if self.mode != "reactive":
            return

        cfg = self.reactive_config
        self._ensure_followers()

        target = None if is_silent else position
        pos = self._position_follower.step(target, dt, cfg.silence_behavior)

        intensity_value, self._physics = apply_intensity_curve(
            intensity, cfg.intensity_curve, cfg.intensity_gamma, dt, self._physics
        )

        raw_blend = self._apply_stage(pos, intensity_value, cfg)
        self.blend_position = self._slew_blend(raw_blend, dt)

    def _ensure_followers(self) -> None:
        cfg = self.reactive_config
        if self._position_follower is None:
            self._position_follower = PositionFollower(
                smoothing_ms=cfg.position_smoothing_ms,
                drift_ms=cfg.drift_ms,
            )
        else:
            self._position_follower.smoothing_ms = cfg.position_smoothing_ms
            self._position_follower.drift_ms = cfg.drift_ms

    def _apply_stage(self, position: float, intensity: float, cfg: ReactiveConfig) -> float:
        left = cfg.stage_left
        right = cfg.stage_right
        home = cfg.stage_home

        pos_value = left + position * (right - left)
        output = home + intensity * (pos_value - home)

        denom = (right - left)
        if abs(denom) < 1e-6:
            return 0.5

        blend = (output - left) / denom
        return max(0.0, min(1.0, blend))

    def step(
        self,
        blend_position: Optional[float] = None,
    ) -> torch.Tensor:
        """Advance one frame, return blended result.

        Args:
            blend_position: Override blend position directly (for slider mode).
                           If provided, sets blend_position and skips reactive logic.

        Returns:
            Blended tensor (reference to internal buffer - do not modify)

        Raises:
            ValueError: If destination A not loaded
        """
        if self.destination_a is None:
            raise ValueError("Destination A not loaded")

        # If only A is loaded, return A
        if self.destination_b is None:
            return self.destination_a.tensor

        # Update blend position based on mode
        if blend_position is not None:
            # Direct override (slider mode from strategy)
            self.blend_position = max(0.0, min(1.0, blend_position))
        elif self.mode == "reactive":
            # Reactive/global blend position is computed externally in strategy.
            pass

        # SLERP with pre-allocated buffers for torch.compile compatibility
        return slerp_inplace(
            self.destination_a.tensor,
            self.destination_b.tensor,
            self.blend_position,
            out=self._result_buffer,
            v0_norm_buf=self._slerp_v0_norm,
            v1_norm_buf=self._slerp_v1_norm,
            temp_buf=self._slerp_temp,
        )

    def step_dual(
        self,
        blend_position: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step for prompt destinations - SLERP both prompt_embeds AND pooled_prompt_embeds.

        SDXL requires both prompt_embeds and pooled_prompt_embeds. This method
        SLERPs both tensors with the same blend position.

        Args:
            blend_position: Override blend position directly (for slider mode).
                           If provided, sets blend_position and skips reactive logic.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)

        Raises:
            ValueError: If destination A not loaded
        """
        if self.destination_a is None:
            raise ValueError("Destination A not loaded")

        # If only A is loaded, return A's tensors
        if self.destination_b is None:
            pooled_a = self.destination_a.tensor_pooled
            if pooled_a is None:
                pooled_a = self.destination_a.tensor
            return self.destination_a.tensor, pooled_a

        # Update blend position (same logic as step())
        if blend_position is not None:
            self.blend_position = max(0.0, min(1.0, blend_position))
        elif self.mode == "reactive":
            # Reactive/global blend position is computed externally in strategy.
            pass

        # SLERP prompt_embeds with pre-allocated intermediate buffers
        embeds = slerp_inplace(
            self.destination_a.tensor,
            self.destination_b.tensor,
            self.blend_position,
            out=self._embeds_buffer,
            v0_norm_buf=self._slerp_v0_norm,
            v1_norm_buf=self._slerp_v1_norm,
            temp_buf=self._slerp_temp,
        )

        # SLERP pooled_prompt_embeds (both must exist for prompt space)
        pooled_a = self.destination_a.tensor_pooled
        pooled_b = self.destination_b.tensor_pooled
        if pooled_a is None or pooled_b is None:
            raise ValueError(
                "step_dual requires both destinations to have tensor_pooled. "
                f"Got A.tensor_pooled={pooled_a is not None}, B.tensor_pooled={pooled_b is not None}"
            )

        # SLERP pooled with pre-allocated intermediate buffers
        pooled = slerp_inplace(
            pooled_a,
            pooled_b,
            self.blend_position,
            out=self._pooled_buffer,
            v0_norm_buf=self._slerp_pooled_v0,
            v1_norm_buf=self._slerp_pooled_v1,
            temp_buf=self._slerp_pooled_temp,
        )

        return embeds, pooled

    def replace_destination(
        self,
        which: Literal["a", "b"],
        new_dest: Destination,
    ) -> None:
        """Replace a destination with node-to-node transition.

        When both destinations exist:
        1. Compute current blended result
        2. Freeze that result as new A
        3. Set new destination as B
        4. Reset blend position to 0 (we're now AT the frozen state)

        This creates smooth node-to-node travel - you're always traveling
        FROM where you are TO somewhere new.

        Args:
            which: Which slot to replace ("a" or "b")
            new_dest: New destination to load
        """
        if self.destination_a is None or self.destination_b is None:
            # Simple load, no transition needed
            if which == "a":
                self.load_a(new_dest)
            else:
                self.load_b(new_dest)
            return

        # Freeze current blended state as new A
        # For prompt space, need to freeze both embeds and pooled_embeds
        if self.destination_a.tensor_pooled is not None:
            # Prompt space: use step_dual to get both tensors
            embeds, pooled = self.step_dual(blend_position=self.blend_position)
            frozen = Destination(
                tensor=embeds.clone(),
                tensor_pooled=pooled.clone(),
                label="[from blend]",
            )
        else:
            # Latent space: just the main tensor
            current = self.step(blend_position=self.blend_position)
            frozen = Destination(
                tensor=current.clone(),
                label="[from blend]",
            )

        # Set up new A→B
        self.destination_a = frozen
        self.destination_b = new_dest
        self.blend_position = 0.0  # We're now AT the frozen state

        # Reset physics if in reactive mode (start fresh toward new target)
        if self._physics is not None:
            self._physics.reset(position=0.0)

    def freeze_blend_to(self, target: Literal["a", "b"]) -> None:
        """Capture current blended result into target slot.

        Computes SLERP at the current blend position and loads the result
        into the specified slot. The other slot stays as-is.

        Example: at 50% blend of Seed 42 and Seed 999, freeze to A
        makes A = the blended tensor, B stays Seed 999, blend resets to 0.

        Args:
            target: Which slot to freeze into ("a" or "b")
        """
        if self.destination_a is None or self.destination_b is None:
            return  # Nothing to freeze

        # Compute current blend
        if self.destination_a.tensor_pooled is not None:
            # Prompt space: freeze both embeds and pooled
            embeds, pooled = self.step_dual(blend_position=self.blend_position)
            frozen = Destination(
                tensor=embeds.clone(),
                tensor_pooled=pooled.clone(),
                label="[frozen blend]",
            )
        else:
            # Latent space: just the main tensor
            current = self.step(blend_position=self.blend_position)
            frozen = Destination(
                tensor=current.clone(),
                label="[frozen blend]",
            )

        if target == "a":
            self.destination_a = frozen
            # At position 0 we get the frozen blend, position 1 is still B
            self.blend_position = 0.0
        else:
            self.destination_b = frozen
            # At position 0 is still A, position 1 is the frozen blend
            self.blend_position = 1.0

        if self._physics is not None:
            self._physics.reset(position=self.blend_position)

    def reset(self) -> None:
        """Reset blend position and physics to initial state."""
        self.blend_position = 0.0
        if self._physics is not None:
            self._physics.reset()

    def has_destinations(self) -> bool:
        """Check if at least one destination is loaded.

        Returns:
            True if destination A is loaded, False otherwise
        """
        return self.destination_a is not None

    def warmup(self) -> None:
        """Pre-warm SLERP kernels to avoid first-time compilation spike.

        Runs full SLERP path once at t=0.5 to compile/optimize CUDA kernels.
        Call after loading both destinations A and B.
        """
        if self.destination_a is None or self.destination_b is None:
            return  # Need both destinations to warmup

        # Save current position
        old_pos = self.blend_position

        # Run full SLERP path (not edge case) to warm up kernels
        if self.destination_a.tensor_pooled is not None:
            # Prompt space: warmup step_dual
            self.step_dual(blend_position=0.5)
        else:
            # Latent space: warmup step
            self.step(blend_position=0.5)

        # Restore position
        self.blend_position = old_pos

    def get_status(self) -> Dict:
        """Get current status for UI display.

        Returns:
            Dict with labels, position, mode info
        """
        return {
            "destination_a": self.destination_a.label if self.destination_a else None,
            "destination_b": self.destination_b.label if self.destination_b else None,
            "blend_position": self.blend_position,
            "mode": self.mode,
        }


