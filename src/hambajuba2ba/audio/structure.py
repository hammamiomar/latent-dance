"""Structural analysis for audio-reactive visualization.

Implements §8 of the audio v2 spec: structural awareness features that
drive SAE strength bursts, SLERP movement, and visual complexity.

Features:
- Multi-timescale novelty: detect transients (short), phrases (medium), sections (long)
- Layer detection: identify when instruments enter/exit
- Layer count: estimate active layers per frame for visual complexity

Design:
- Purely offline (runs at upload time)
- BPM-scaled: timescales adapt to music tempo
- Complements existing perceptual features (flux, flatness, energy)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_multi_timescale_novelty(
    flux_normalized: np.ndarray,
    fps: int,
    bpm: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute novelty at three musically-meaningful timescales.

    Per spec §8: Uses amplitude-normalized spectral flux with Gaussian smoothing
    at different scales. This is the MIR-standard approach.

    Timescales (BPM-scaled):
    - Short: ~0.5 second (transients, fills, ornaments) - for SAE bursts
    - Medium: ~4 seconds (phrase boundaries, 4-bar changes) - for SLERP movement
    - Long: ~16 seconds (section changes, drops) - for destination jumps

    Args:
        flux_normalized: Amplitude-normalized spectral flux, shape (n_frames,)
        fps: Frame rate
        bpm: Beats per minute (for tempo-scaling)

    Returns:
        Tuple of (novelty_short, novelty_medium, novelty_long), each shape (n_frames,)
    """
    # Tempo-scale factor: 120 BPM is reference
    # Faster tempo = shorter windows (events happen faster)
    tempo_scale = 120.0 / max(60.0, min(180.0, bpm))

    # Gaussian smoothing sigmas in frames (BPM-scaled)
    # Short: ~0.5 second window → captures individual transients
    sigma_short = int(0.5 * fps * tempo_scale)
    # Medium: ~4 second window → captures phrase-level changes
    sigma_medium = int(4.0 * fps * tempo_scale)
    # Long: ~16 second window → captures section-level changes
    sigma_long = int(16.0 * fps * tempo_scale)

    # Ensure minimum sigmas (at least 1 frame)
    sigma_short = max(1, sigma_short)
    sigma_medium = max(2, sigma_medium)
    sigma_long = max(4, sigma_long)

    # Apply Gaussian smoothing at each timescale
    # This is the spec-compliant approach: flux-based with different smoothing
    novelty_short = gaussian_filter1d(flux_normalized, sigma=sigma_short)
    novelty_medium = gaussian_filter1d(flux_normalized, sigma=sigma_medium)
    novelty_long = gaussian_filter1d(flux_normalized, sigma=sigma_long)

    # Normalize each to [0, 1]
    for arr in (novelty_short, novelty_medium, novelty_long):
        max_val = arr.max()
        if max_val > 1e-10:
            arr /= max_val

    return novelty_short, novelty_medium, novelty_long


def detect_layer_entries(
    flux_normalized: np.ndarray,
    flatness: np.ndarray,
    fps: float,
    threshold_flux: float = 0.4,
    threshold_flatness_change: float = 0.15,
) -> np.ndarray:
    """Detect frames where new instrumental layers enter.

    A layer entry is characterized by:
    1. High normalized flux (spectral change independent of volume)
    2. Significant flatness change (timbral texture shift)

    This detects when an instrument starts playing, even if quietly.
    More robust than simple energy-based onset detection.

    Args:
        flux_normalized: Amplitude-normalized spectral flux, shape (T,)
        flatness: Spectral flatness (0=tonal, 1=noisy), shape (T,)
        fps: Frames per second
        threshold_flux: Normalized flux threshold for detection
        threshold_flatness_change: Flatness delta threshold

    Returns:
        Boolean mask, shape (T,), True at layer entry frames
    """
    n_frames = len(flux_normalized)

    # Compute flatness derivative (texture changes)
    flatness_diff = np.abs(np.diff(flatness, prepend=flatness[0]))

    # Smooth flatness_diff to reduce noise
    flatness_diff = gaussian_filter1d(flatness_diff, sigma=fps / 10)

    # Detect layer entries: high flux AND texture change (vectorized)
    mask = (flux_normalized > threshold_flux) & (flatness_diff > threshold_flatness_change)

    # Debounce: require minimum spacing between detections (500ms)
    min_spacing = int(0.5 * fps)
    last_entry = -min_spacing - 1
    debounced = np.zeros_like(mask)

    for i in range(n_frames):
        if mask[i] and (i - last_entry) >= min_spacing:
            debounced[i] = True
            last_entry = i

    return debounced


def estimate_layer_count(
    layer_entry_mask: np.ndarray,
    energy_db: np.ndarray,
    fps: float,
    decay_time: float = 8.0,
    silence_threshold_db: float = -35.0,
) -> np.ndarray:
    """Estimate the number of active instrumental layers per frame.

    Tracks layer entries with gradual decay. More layers = more visual complexity.
    Layers "exit" when overall energy drops significantly.

    Args:
        layer_entry_mask: Boolean mask from detect_layer_entries
        energy_db: Energy in dB per frame (from HPSS)
        fps: Frames per second
        decay_time: Seconds for a layer to fully decay without reinforcement
        silence_threshold_db: dB below which layers reset

    Returns:
        Layer count per frame, shape (T,), values typically 0-6
    """
    n_frames = len(layer_entry_mask)
    layer_count = np.zeros(n_frames, dtype=np.float32)

    # Running layer estimate with decay
    current_layers = 0.0
    decay_per_frame = 1.0 / (decay_time * fps)

    for i in range(n_frames):
        # Layer entry: increment
        if layer_entry_mask[i]:
            current_layers += 1.0

        # Natural decay
        current_layers = max(0, current_layers - decay_per_frame)

        # Hard reset on silence
        if energy_db is not None and i < len(energy_db):
            if energy_db[i] < silence_threshold_db:
                current_layers = 0.0

        layer_count[i] = current_layers

    # Round to integers and cap at reasonable max
    layer_count = np.clip(np.round(layer_count), 0, 6).astype(np.int32)

    return layer_count


def compute_novelty_derivative(novelty: np.ndarray, fps: float) -> np.ndarray:
    """Compute rate of change of novelty (rising vs falling edges).

    Positive derivative = novelty increasing = event starting
    Negative derivative = novelty decreasing = event ending

    The derivative is useful for triggering visual responses:
    - High positive derivative → SAE strength burst trigger
    - Sustained positive derivative → SLERP movement

    Args:
        novelty: Novelty curve, shape (T,)
        fps: Frames per second (for scaling)

    Returns:
        Novelty derivative, shape (T,), values roughly in [-1, 1]
    """
    # Compute frame-to-frame difference
    derivative = np.diff(novelty, prepend=novelty[0])

    # Scale by fps to get per-second rate
    derivative = derivative * fps

    # Smooth to reduce jitter
    derivative = gaussian_filter1d(derivative, sigma=fps / 20)

    # Normalize to roughly [-1, 1]
    max_abs = np.abs(derivative).max()
    if max_abs > 1e-10:
        derivative = derivative / max_abs

    return derivative.astype(np.float32)


def compute_structure_features(
    flux_normalized: np.ndarray,
    flatness: np.ndarray,
    energy_db: np.ndarray | None,
    fps: int,
    bpm: float,
) -> dict:
    """Compute all structural features for a stem.

    Convenience function that runs all structure analysis in one call.

    Args:
        flux_normalized: Pre-computed amplitude-normalized flux
        flatness: Pre-computed spectral flatness
        energy_db: Pre-computed energy in dB (optional)
        fps: Frame rate
        bpm: Tempo in BPM (for BPM-scaled timescales)

    Returns:
        Dict with keys: novelty_short, novelty_medium, novelty_long,
                       novelty_short_deriv, novelty_medium_deriv,
                       layer_entry_mask
    """
    # Multi-timescale novelty (flux-based per spec §8)
    novelty_short, novelty_medium, novelty_long = compute_multi_timescale_novelty(
        flux_normalized, fps, bpm
    )

    # Novelty derivatives (for detecting rising edges / event triggers)
    novelty_short_deriv = compute_novelty_derivative(novelty_short, fps)
    novelty_medium_deriv = compute_novelty_derivative(novelty_medium, fps)

    # Layer detection
    layer_entry_mask = detect_layer_entries(flux_normalized, flatness, fps)

    return {
        "novelty_short": novelty_short,
        "novelty_medium": novelty_medium,
        "novelty_long": novelty_long,
        "novelty_short_deriv": novelty_short_deriv,
        "novelty_medium_deriv": novelty_medium_deriv,
        "layer_entry_mask": layer_entry_mask,
    }
