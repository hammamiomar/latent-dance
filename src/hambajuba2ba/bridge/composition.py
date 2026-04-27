"""Unified composition engine: noise circular walk + motion signals.

Combines the old CompositionEngine (GPU noise buffers) with MotionEngine
(beat grid + tonal drift) into a single module with one entry point:

    noise = composition.get_noise(audio_time)

Motion signals are pre-computed as cumulative angles at distance=1.0:
- Beat signal: angular increments per beat, cubic-eased within interval
- Drift signal: tonal distance mapped to smooth angular displacement
- Beat weight: transient/energy ratio blends beat vs drift

At runtime: theta = precomputed_angle(audio_time) * distance
Then: noise(theta) = cos(theta) * noise_a + sin(theta) * noise_b

Unlike the old MotionEngine (blend [0,1] → quarter circle), this uses
unbounded cumulative angles for full circular walks. The `distance`
property controls how far around the circle each beat moves.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Motion signal builders (offline, at upload) ─────────────────────────


def _cubic_ease_in_out(t: np.ndarray) -> np.ndarray:
    """Cubic ease-in-out for smooth beat pulse transitions.

    t in [0,1] -> smooth [0,1]
    """
    return np.where(
        t < 0.5,
        4.0 * t * t * t,
        1.0 - (-2.0 * t + 2.0) ** 3 / 2.0,
    )


def _build_beat_angles(
    beat_times: np.ndarray,
    energy_at_beats: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Pre-compute cumulative beat angles at distance=1.0.

    Between beats, angle increments by energy * 2*pi with cubic easing.
    The cumulative sum gives continuous position on the circle.

    Args:
        beat_times: (N,) beat timestamps in seconds
        energy_at_beats: (N,) energy [0,1] at each beat
        timestamps: (T,) output time axis

    Returns:
        (T,) cumulative angle in radians (at distance=1.0)
    """
    angles = np.zeros(len(timestamps), dtype=np.float32)

    if len(beat_times) < 2:
        return angles

    running_angle = 0.0

    for i in range(len(beat_times) - 1):
        t_start = beat_times[i]
        t_end = beat_times[i + 1]
        energy = energy_at_beats[i]

        total_angle = energy * 2 * np.pi

        mask = (timestamps >= t_start) & (timestamps < t_end)
        if not np.any(mask):
            running_angle += total_angle
            continue

        t_local = (timestamps[mask] - t_start) / max(t_end - t_start, 1e-6)
        eased = _cubic_ease_in_out(t_local)

        angles[mask] = running_angle + eased * total_angle
        running_angle += total_angle

    # After last beat: hold at final angle
    if len(beat_times) > 0 and beat_times[-1] < timestamps[-1]:
        mask = timestamps >= beat_times[-1]
        angles[mask] = running_angle

    return angles


def _build_drift_angles(
    tonal_distance: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Pre-compute tonal drift as angular displacement.

    Tonal distance [0,1] maps to angular displacement [0, pi/2].
    Home key → 0, maximum departure → pi/2 radians.

    Args:
        tonal_distance: (T,) tonal distance [0, 1]
        timestamps: (T,) time axis

    Returns:
        (T,) angular displacement in radians
    """
    if tonal_distance is None or len(tonal_distance) == 0:
        return np.zeros(len(timestamps), dtype=np.float32)

    n = min(len(tonal_distance), len(timestamps))
    angles = tonal_distance[:n].astype(np.float32) * (np.pi / 2)

    if len(angles) < len(timestamps):
        angles = np.pad(angles, (0, len(timestamps) - len(angles)), mode="edge")

    return angles


def _build_beat_weight(
    transient: np.ndarray,
    energy_smooth: np.ndarray,
    timestamps: np.ndarray,
    smoothing_sigma: float = 1.0,
) -> np.ndarray:
    """Pre-compute adaptive beat weight from transient-to-energy ratio.

    High ratio → percussive → beats dominate (weight → 1).
    Low ratio → sustained → drift dominates (weight → 0).

    Args:
        transient: (T,) transient energy
        energy_smooth: (T,) smoothed energy envelope
        timestamps: (T,) time axis
        smoothing_sigma: Gaussian smoothing for weight curve (seconds)

    Returns:
        (T,) beat weight in [0, 1]
    """
    from scipy.ndimage import gaussian_filter1d

    n = min(len(transient), len(energy_smooth), len(timestamps))
    transient = transient[:n]
    energy = energy_smooth[:n]

    ratio = transient / np.maximum(energy, 0.05)

    p5, p95 = np.percentile(ratio, [5, 95])
    if p95 - p5 > 1e-6:
        ratio = np.clip((ratio - p5) / (p95 - p5), 0.0, 1.0)
    else:
        ratio = np.full_like(ratio, 0.5)

    fps = len(timestamps) / max(timestamps[-1] - timestamps[0], 1e-6) if len(timestamps) > 1 else 60.0
    sigma_frames = max(1, int(smoothing_sigma * fps))
    ratio = gaussian_filter1d(ratio, sigma=sigma_frames)

    return ratio.astype(np.float32)


# ─── CompositionEngine ───────────────────────────────────────────────────


class CompositionEngine:
    """Unified noise circular walk engine. CUDA-graph safe.

    Combines GPU noise buffers with pre-computed angular motion signals.
    One call produces the noise tensor that dominates composition (~95%).

        noise(theta) = cos(theta) * noise_a + sin(theta) * noise_b

    Properties:
        distance: Circle radius (how much angular change per beat).
                  1.0 = full 2*pi per energetic beat, 0.5 = half circle.
        mode: "auto" (adaptive beat/drift), "pulse" (beat_weight=1),
              "continuous" (beat_weight=0).

    Usage:
        comp = CompositionEngine(shape, device, dtype)
        comp.load_noise("a", engine.make_noise(seed_a), seed_a)
        comp.load_noise("b", engine.make_noise(seed_b), seed_b)
        comp.load_motion(beat_times, energy_at_beats, tonal_distance,
                         transient, energy_smooth, timestamps)

        # In frame loop (~0.01ms):
        noise = comp.get_noise(audio_time)
    """

    def __init__(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize with pre-allocated GPU buffers.

        Args:
            shape: (1, 4, H/8, W/8) noise tensor shape
            device: GPU device
            dtype: Tensor dtype (float16 typically)
        """
        # GPU noise buffers (addresses never change for CUDA graph safety)
        self._noise_a = torch.zeros(shape, device=device, dtype=dtype)
        self._noise_b = torch.zeros(shape, device=device, dtype=dtype)
        self._out = torch.zeros(shape, device=device, dtype=dtype)

        self._has_a = False
        self._has_b = False
        self._seed_a: int | None = None
        self._seed_b: int | None = None

        # Pre-computed motion arrays (set by load_motion)
        self._timestamps: np.ndarray | None = None
        self._beat_angles: np.ndarray | None = None
        self._drift_angles: np.ndarray | None = None
        self._combined_angles: np.ndarray | None = None

        # Runtime properties
        self.distance: float = 1.0
        self.mode: str = "auto"

    def load_noise(self, slot: str, tensor: torch.Tensor, seed: int | None = None) -> None:
        """Load a noise tensor into slot A or B.

        Uses copy_() for CUDA graph safety — buffer address never changes.

        Args:
            slot: "a" or "b"
            tensor: Pre-computed noise tensor from engine.make_noise()
            seed: Optional seed for logging
        """
        if slot == "a":
            self._noise_a.copy_(tensor)
            self._has_a = True
            self._seed_a = seed
        else:
            self._noise_b.copy_(tensor)
            self._has_b = True
            self._seed_b = seed

    def has_both(self) -> bool:
        """Check if both noise buffers are loaded."""
        return self._has_a and self._has_b

    def load_motion(
        self,
        beat_times: np.ndarray,
        energy_at_beats: np.ndarray,
        tonal_distance: Optional[np.ndarray],
        transient: np.ndarray,
        energy_smooth: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """Pre-compute all motion signals from audio features.

        Call once during setup (not per-frame). Builds cumulative angle
        arrays that get_noise() samples from at runtime.

        Args:
            beat_times: (N,) beat timestamps in seconds
            energy_at_beats: (N,) energy [0,1] at each beat
            tonal_distance: (T,) JSD tonal distance [0,1] per frame (or None)
            transient: (T,) transient/peak energy per frame
            energy_smooth: (T,) smoothed energy envelope per frame
            timestamps: (T,) time axis in seconds
        """
        self._timestamps = timestamps

        self._beat_angles = _build_beat_angles(beat_times, energy_at_beats, timestamps)
        self._drift_angles = _build_drift_angles(tonal_distance, timestamps)
        beat_weight = _build_beat_weight(transient, energy_smooth, timestamps)

        # Ensure drift_angles matches timestamps length
        if len(self._drift_angles) < len(timestamps):
            self._drift_angles = np.pad(
                self._drift_angles,
                (0, len(timestamps) - len(self._drift_angles)),
                mode="edge",
            )

        # Combined: beat weight blends between beat-driven and drift-driven angles
        self._combined_angles = beat_weight * self._beat_angles + (1.0 - beat_weight) * self._drift_angles

        logger.info(
            "CompositionEngine motion: %d beats, %d frames, "
            "angle range [%.2f, %.2f] rad",
            len(beat_times),
            len(timestamps),
            float(self._combined_angles.min()),
            float(self._combined_angles.max()),
        )

    def step(self, theta: float) -> torch.Tensor:
        """Walk the noise circle at angle theta, return blended result.

        noise(theta) = cos(theta) * noise_a + sin(theta) * noise_b

        All ops are in-place on pre-allocated _out for CUDA graph safety.

        Args:
            theta: Angle in radians

        Returns:
            Reference to internal output buffer (do not modify)
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        self._out.copy_(self._noise_a).mul_(cos_t).add_(self._noise_b, alpha=sin_t)
        return self._out

    def get_noise(self, audio_time: float) -> torch.Tensor:
        """Get noise tensor for the given audio time.

        Looks up precomputed angle, multiplies by distance, and steps.
        This is the single entry point for the frame loop.

        Args:
            audio_time: Current playback time in seconds

        Returns:
            Interpolated noise tensor (reference to internal buffer)
        """
        if self._combined_angles is not None and self._timestamps is not None:
            # Select angle source based on mode
            if self.mode == "pulse" and self._beat_angles is not None:
                angles = self._beat_angles
            elif self.mode == "continuous" and self._drift_angles is not None:
                angles = self._drift_angles
            else:
                angles = self._combined_angles

            angle = float(np.interp(audio_time, self._timestamps, angles))
            theta = angle * self.distance
        else:
            theta = 0.0  # No motion data: static noise_a

        return self.step(theta)
