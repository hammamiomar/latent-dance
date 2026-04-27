"""Shared audio utilities.

Small helpers used across the audio subpackage.
Extracted here to avoid duplication between modules.
"""

from __future__ import annotations

import numpy as np


def align_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Align 1D array to target length (librosa outputs can vary by +/-1).

    Truncates if longer, edge-pads if shorter.

    Args:
        arr: 1D array to align
        target_len: Desired length

    Returns:
        Array of exactly target_len elements
    """
    if len(arr) == target_len:
        return arr
    elif len(arr) > target_len:
        return arr[:target_len]
    else:
        return np.pad(arr, (0, target_len - len(arr)), mode="edge")
