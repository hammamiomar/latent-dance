"""Pitch-indexed spatial masks for SAE activation map steering.

12 chromatic Gaussian bands map pitch → Y-axis position:
  pitch_level 0  → band near bottom (low notes)
  pitch_level 11 → band near top (high notes)

Pre-generated at warmup for O(1) runtime indexing.
"""

from __future__ import annotations

import numpy as np
import torch

N_PITCH_LEVELS = 12


def _generate_pitch_mask(
    pitch_level: int,
    height: int,
    width: int,
    spread: float = 0.25,
) -> np.ndarray:
    """Generate spatial mask for a specific pitch level.

    Args:
        pitch_level: Pitch index 0-11 (0=lowest, 11=highest)
        height: Mask height
        width: Mask width
        spread: Vertical spread of the Gaussian

    Returns:
        Mask array, shape (height, width), values in [0, 1]
    """
    y_center = 1.0 - (pitch_level / (N_PITCH_LEVELS - 1))
    y = np.linspace(0, 1, height)[:, np.newaxis]
    mask = np.exp(-((y - y_center) ** 2) / (2 * spread ** 2))
    return np.broadcast_to(mask, (height, width)).astype(np.float32).copy()


def generate_pitch_indexed_masks(
    height: int,
    width: int,
) -> np.ndarray:
    """Pre-generate all 12 chromatic pitch masks.

    Called once during warmup. Returns stacked array for O(1) runtime indexing.

    Returns:
        Stacked masks, shape (12, height, width)
    """
    masks = [
        _generate_pitch_mask(level, height, width)
        for level in range(N_PITCH_LEVELS)
    ]
    return np.stack(masks, axis=0)


def generate_pitch_indexed_masks_gpu(
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Pre-generate all 12 chromatic pitch masks on GPU.

    Called once during warmup. Zero CPU→GPU transfers at runtime.

    Returns:
        GPU tensor, shape (12, height, width)
    """
    masks_np = generate_pitch_indexed_masks(height, width)
    return torch.from_numpy(masks_np).to(device=device, dtype=dtype)


def pitch_to_mask_index(pitch_normalized: float) -> int:
    """Convert normalized pitch [0, 1] to mask index [0, 11]."""
    clamped = max(0.0, min(1.0, pitch_normalized))
    return min(int(clamped * N_PITCH_LEVELS), N_PITCH_LEVELS - 1)
