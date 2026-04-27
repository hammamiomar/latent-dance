"""Perceptual audio processing for natural-feeling reactivity.

This module implements perceptual audio processing that creates
visualizations which "feel right" - fast attacks that catch transients,
slow releases for organic decay, and proper cross-modal correspondence.

Core insight: Human perception is asymmetric. We notice sudden changes
immediately but perceive decay as gradual. The asymmetric envelope follower
models this by using different time constants for rising vs falling signals.

References:
- Spence (2011): Crossmodal correspondences
- Ciphrd: Beat-reactive visualization (asymmetric smoothing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


@dataclass(frozen=True)
class EnvelopeConfig:
    """Configuration for asymmetric envelope following.

    Time constants are in milliseconds. The ratio of release/attack
    determines how "punchy" (low ratio) vs "sustained" (high ratio)
    the response feels.

    Attributes:
        attack_ms: Time to reach ~63% of target when rising
        release_ms: Time to reach ~63% of target when falling
    """
    attack_ms: float = 5.0
    release_ms: float = 150.0

    def to_frame_coefficients(self, fps: float) -> tuple[float, float]:
        """Convert ms time constants to frame-rate coefficients.

        Returns one-pole IIR coefficients: coeff = 1 - exp(-1/tau_frames)

        Note: We use a time-based minimum (0.5ms) instead of frame-based (0.1 frames)
        to allow sub-frame response for fast transients and ensure FPS-independent behavior.
        """
        ms_per_frame = 1000.0 / fps

        # Minimum 0.5ms time constant - sub-frame response is fine for fast transients
        # This is FPS-independent (0.5ms stays 0.5ms at any frame rate)
        MIN_TIME_CONSTANT_MS = 0.5
        attack_frames = max(self.attack_ms / ms_per_frame, MIN_TIME_CONSTANT_MS / ms_per_frame)
        release_frames = max(self.release_ms / ms_per_frame, MIN_TIME_CONSTANT_MS / ms_per_frame)

        attack_coeff = 1.0 - np.exp(-1.0 / attack_frames)
        release_coeff = 1.0 - np.exp(-1.0 / release_frames)

        return attack_coeff, release_coeff


# Stem-specific presets based on perceptual requirements
# Presets loaded from presets/envelope.yaml
def _load_envelope_presets() -> dict[str, EnvelopeConfig]:
    """Load envelope presets from YAML file."""
    from hambajuba2ba.presets import load_envelope_presets

    json_presets = load_envelope_presets()
    return {
        name: EnvelopeConfig(
            attack_ms=params.get("attack_ms", 5.0),
            release_ms=params.get("release_ms", 150.0),
        )
        for name, params in json_presets.items()
    }


_ENVELOPE_PRESETS_CACHE: dict[str, EnvelopeConfig] | None = None


def _get_envelope_presets() -> dict[str, EnvelopeConfig]:
    """Get envelope presets (lazy load from JSON)."""
    global _ENVELOPE_PRESETS_CACHE
    if _ENVELOPE_PRESETS_CACHE is None:
        _ENVELOPE_PRESETS_CACHE = _load_envelope_presets()
    return _ENVELOPE_PRESETS_CACHE


def get_envelope_preset(stem: str) -> EnvelopeConfig:
    """Get envelope preset for a stem, with fallback to default."""
    presets = _get_envelope_presets()
    return presets.get(stem, presets.get("default", EnvelopeConfig()))


# Stem-specific brightness presets (for spectral centroid processing)
# Presets loaded from presets/brightness.yaml
def _load_brightness_presets() -> dict[str, EnvelopeConfig]:
    """Load brightness presets from YAML file."""
    from hambajuba2ba.presets import load_brightness_presets

    json_presets = load_brightness_presets()
    return {
        name: EnvelopeConfig(
            attack_ms=params.get("attack_ms", 15.0),
            release_ms=params.get("release_ms", 150.0),
        )
        for name, params in json_presets.items()
    }


_BRIGHTNESS_PRESETS_CACHE: dict[str, EnvelopeConfig] | None = None


def _get_brightness_presets() -> dict[str, EnvelopeConfig]:
    """Get brightness presets (lazy load from JSON)."""
    global _BRIGHTNESS_PRESETS_CACHE
    if _BRIGHTNESS_PRESETS_CACHE is None:
        _BRIGHTNESS_PRESETS_CACHE = _load_brightness_presets()
    return _BRIGHTNESS_PRESETS_CACHE


def get_brightness_preset(stem: str) -> EnvelopeConfig:
    """Get brightness preset for a stem, with fallback to default."""
    presets = _get_brightness_presets()
    return presets.get(stem, presets.get("default", EnvelopeConfig(attack_ms=15.0, release_ms=150.0)))


def asymmetric_envelope_follow(
    signal: np.ndarray,
    fps: float,
    config: Optional[EnvelopeConfig] = None,
    *,
    attack_ms: Optional[float] = None,
    release_ms: Optional[float] = None,
) -> np.ndarray:
    """Apply asymmetric attack/release envelope following.

    This is the core perceptual transformation: fast attack captures
    transients, slow release creates organic decay. The mathematical
    model is a one-pole IIR filter with state-dependent coefficients.

    Note: This operation is inherently sequential (each sample depends
    on the previous). At video frame rate (~5400 samples for 3 min),
    this is instantaneous.

    Args:
        signal: Input signal, shape (n_frames,)
        fps: Frame rate of the signal
        config: EnvelopeConfig with attack/release times
        attack_ms: Override attack time (alternative to config)
        release_ms: Override release time (alternative to config)

    Returns:
        Smoothed signal with asymmetric response, same shape as input
    """
    if config is None:
        config = EnvelopeConfig(
            attack_ms=attack_ms or 5.0,
            release_ms=release_ms or 150.0,
        )

    attack_coeff, release_coeff = config.to_frame_coefficients(fps)

    # IIR filter with conditional coefficient selection
    # y[n] = y[n-1] + coeff * (x[n] - y[n-1])
    # where coeff = attack_coeff if x[n] > y[n-1] else release_coeff
    output = np.empty_like(signal)
    state = 0.0

    for i, x in enumerate(signal):
        coeff = attack_coeff if x > state else release_coeff
        state += coeff * (x - state)
        output[i] = state

    return output


def compute_onset_strength(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """Compute spectral flux (onset strength) using librosa.

    Spectral flux measures the rate of spectral change - high values
    indicate attacks/transients, low values indicate sustains. This is
    more discriminative than RMS for detecting rhythmic events.

    librosa.onset.onset_strength:
    - Computes mel spectrogram
    - Takes first-order difference
    - Applies half-wave rectification (only increases count)
    - Aggregates across frequency bands

    Args:
        audio: Audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples

    Returns:
        Onset strength envelope, shape (n_frames,), normalized to [0, 1]
    """
    flux = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )

    # Normalize to [0, 1]
    flux_max = flux.max()
    if flux_max > 1e-8:
        flux = flux / flux_max

    return flux


def detect_peaks(
    signal: np.ndarray,
    fps: float,
    *,
    pre_max_ms: float = 30.0,
    post_max_ms: float = 30.0,
    pre_avg_ms: float = 100.0,
    post_avg_ms: float = 100.0,
    delta: float = 0.35,
    wait_ms: float = 80.0,
) -> np.ndarray:
    """Detect peaks in a signal using librosa's peak picker.

    Returns a binary mask at the same frame rate as input, with 1s
    at detected peak locations. This is computed offline at upload
    time, not per-frame at runtime.

    The parameters control sensitivity vs. false positive rate:
    - pre/post_max: Local maximum window (shorter = more peaks)
    - pre/post_avg: Comparison to local average (longer = fewer spurious)
    - delta: Threshold above local average (higher = fewer peaks)
      Research: 0.3-0.5 range. 0.35 = meaningful 35% rise required.
    - wait: Minimum time between peaks (prevents double-triggers)
      Research: 60-200ms. 80ms prevents double-triggers on transients.

    Args:
        signal: Input signal, shape (n_frames,)
        fps: Frame rate of the signal
        pre_max_ms/post_max_ms: Peak maximum window in ms
        pre_avg_ms/post_avg_ms: Average comparison window in ms
        delta: Threshold above average to count as peak
        wait_ms: Minimum inter-peak interval in ms

    Returns:
        Binary peak mask, shape (n_frames,), values in {0, 1}
    """
    # Convert ms to frames
    ms_per_frame = 1000.0 / fps
    pre_max = max(1, int(pre_max_ms / ms_per_frame))
    post_max = max(1, int(post_max_ms / ms_per_frame))
    pre_avg = max(1, int(pre_avg_ms / ms_per_frame))
    post_avg = max(1, int(post_avg_ms / ms_per_frame))
    wait = max(1, int(wait_ms / ms_per_frame))

    # librosa's peak_pick returns indices of peaks
    peak_indices = librosa.util.peak_pick(
        signal,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )

    # Convert to binary mask
    mask = np.zeros(len(signal), dtype=np.float32)
    mask[peak_indices] = 1.0

    return mask


def compute_spectral_flatness(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    mag: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spectral flatness (tonality measure).

    Spectral flatness is the ratio of geometric to arithmetic mean
    of power spectrum. Values near 1 indicate noise-like signals,
    values near 0 indicate tonal signals.

    Useful for distinguishing:
    - Tonal content (synths, vocals, pitched instruments): low flatness
    - Noise-like content (hi-hats, cymbals, breath): high flatness

    Args:
        audio: Audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples

    Returns:
        Spectral flatness, shape (n_frames,), values in [0, 1]
    """
    if mag is None:
        flatness = librosa.feature.spectral_flatness(
            y=audio,
            hop_length=hop_length,
        )[0]
    else:
        flatness = librosa.feature.spectral_flatness(S=mag)[0]

    return flatness


def normalize_feature(
    feature: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> np.ndarray:
    """Normalize feature to [0, 1] using percentile clipping.

    Using 5th/95th percentiles instead of min/max prevents outliers
    (single loud transients, silence) from compressing the useful
    dynamic range. Research-validated for audio visualization.

    Args:
        feature: Feature array, shape (n_frames,)
        percentile_low: Lower percentile for clipping (default: 5)
        percentile_high: Upper percentile for clipping (default: 95)

    Returns:
        Normalized feature, shape (n_frames,), values in [0, 1]
    """
    low = np.percentile(feature, percentile_low)
    high = np.percentile(feature, percentile_high)

    if high - low > 1e-8:
        normalized = (feature - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0)
    else:
        return np.zeros_like(feature)


# =============================================================================
# Dual-Layer Response (Flash + Sustain)
# =============================================================================


@dataclass(frozen=True)
class DualLayerConfig:
    """Configuration for dual-layer (flash + sustain) response.

    Flash captures immediate transient impact with very fast attack/release.
    Sustain captures the trailing "comet tail" with slower dynamics.

    Combined, they create a visual effect where:
    - Flash: bright initial "pop" that appears and fades quickly
    - Sustain: gentler trailing glow that lingers

    Attributes:
        flash_attack_ms: Flash attack time (very fast, ~2ms)
        flash_release_ms: Flash release time (fast, ~50ms)
        sustain_attack_ms: Sustain attack time (moderate, ~20ms)
        sustain_release_ms: Sustain release time (slow, ~200ms)
    """

    flash_attack_ms: float = 2.0
    flash_release_ms: float = 50.0
    sustain_attack_ms: float = 20.0
    sustain_release_ms: float = 200.0

    def flash_config(self) -> EnvelopeConfig:
        return EnvelopeConfig(self.flash_attack_ms, self.flash_release_ms)

    def sustain_config(self) -> EnvelopeConfig:
        return EnvelopeConfig(self.sustain_attack_ms, self.sustain_release_ms)


# Stem-specific dual-layer presets
# Presets loaded from presets/dual_layer.yaml
def _load_dual_layer_presets() -> dict[str, DualLayerConfig]:
    """Load dual-layer presets from YAML file."""
    from hambajuba2ba.presets import load_dual_layer_presets

    json_presets = load_dual_layer_presets()
    return {
        name: DualLayerConfig(
            flash_attack_ms=params.get("flash_attack_ms", 2.0),
            flash_release_ms=params.get("flash_release_ms", 50.0),
            sustain_attack_ms=params.get("sustain_attack_ms", 20.0),
            sustain_release_ms=params.get("sustain_release_ms", 200.0),
        )
        for name, params in json_presets.items()
    }


_DUAL_LAYER_PRESETS_CACHE: dict[str, DualLayerConfig] | None = None


def _get_dual_layer_presets() -> dict[str, DualLayerConfig]:
    """Get dual-layer presets (lazy load from JSON)."""
    global _DUAL_LAYER_PRESETS_CACHE
    if _DUAL_LAYER_PRESETS_CACHE is None:
        _DUAL_LAYER_PRESETS_CACHE = _load_dual_layer_presets()
    return _DUAL_LAYER_PRESETS_CACHE


def get_dual_layer_preset(stem: str) -> DualLayerConfig:
    """Get dual-layer preset for a stem, with fallback to default."""
    presets = _get_dual_layer_presets()
    return presets.get(stem, presets.get("default", DualLayerConfig()))


def compute_dual_layer(
    signal: np.ndarray,
    fps: float,
    config: Optional[DualLayerConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute flash and sustain layers for dual-layer response.

    Args:
        signal: Input signal (typically raw envelope or energy)
        fps: Frame rate
        config: Dual-layer configuration

    Returns:
        Tuple of (flash, sustain) arrays, both shape (n_frames,)
    """
    if config is None:
        config = get_dual_layer_preset("default")

    flash = asymmetric_envelope_follow(signal, fps, config.flash_config())
    sustain = asymmetric_envelope_follow(signal, fps, config.sustain_config())

    return flash, sustain
