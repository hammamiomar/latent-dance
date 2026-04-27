"""Virtual stems via bandpass filtering.

Demucs gives us 4 physical stems: bass, drums, vocals, other.
Virtual stems split these further by frequency to enable finer control:

- drums_low: kick drums, toms (20-200 Hz) → grounded, impact
- drums_mid: snare body + crack (200-5000 Hz) → rhythmic punch
- drums_high: hi-hats, cymbals (5-16 kHz) → elevated, sparkle
- other_mid: guitars, keys, synth body (200-4000 Hz) → harmony, melody
- other_high: brightness, air, shimmer (4-16 kHz) → atmosphere

This uses scipy's Butterworth filters in second-order sections (SOS)
form for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import signal


def _create_linkwitz_riley_sos(freq: float, sr: int, btype: str) -> np.ndarray:
    """Create Linkwitz-Riley filter (flat crossover summing).

    LR-4 = two cascaded 2nd-order Butterworth filters.
    Has -6dB at crossover frequency, sums to unity with complementary filter.
    This is the pro audio standard for crossover networks.

    Args:
        freq: Cutoff frequency in Hz
        sr: Sample rate
        btype: Filter type ('low' or 'high')

    Returns:
        Second-order sections (SOS) filter coefficients
    """
    nyquist = sr / 2
    normalized = freq / nyquist

    # Two cascaded 2nd-order Butterworth = 4th order Linkwitz-Riley
    sos1 = signal.butter(2, normalized, btype=btype, output='sos')
    sos2 = signal.butter(2, normalized, btype=btype, output='sos')
    return np.vstack([sos1, sos2])


@dataclass(frozen=True)
class BandConfig:
    """Configuration for a frequency band.

    Attributes:
        low_hz: Lower cutoff frequency (None = lowpass)
        high_hz: Upper cutoff frequency (None = highpass)
        order: Butterworth filter order (higher = steeper rolloff)
    """

    low_hz: float | None
    high_hz: float | None
    order: int = 4

    def to_sos(self, sr: int, use_linkwitz_riley: bool = True) -> np.ndarray:
        """Convert to second-order sections filter coefficients.

        Args:
            sr: Sample rate
            use_linkwitz_riley: If True, use Linkwitz-Riley filters (flat summed
                response at crossover). If False, use standard Butterworth.
        """
        nyquist = sr / 2

        if self.low_hz is None and self.high_hz is not None:
            # Lowpass
            if use_linkwitz_riley:
                return _create_linkwitz_riley_sos(self.high_hz, sr, 'low')
            return signal.butter(
                self.order,
                self.high_hz / nyquist,
                btype="low",
                output="sos",
            )
        elif self.low_hz is not None and self.high_hz is None:
            # Highpass
            if use_linkwitz_riley:
                return _create_linkwitz_riley_sos(self.low_hz, sr, 'high')
            return signal.butter(
                self.order,
                self.low_hz / nyquist,
                btype="high",
                output="sos",
            )
        elif self.low_hz is not None and self.high_hz is not None:
            # Bandpass: cascade highpass and lowpass
            if use_linkwitz_riley:
                hp_sos = _create_linkwitz_riley_sos(self.low_hz, sr, 'high')
                lp_sos = _create_linkwitz_riley_sos(self.high_hz, sr, 'low')
                return np.vstack([hp_sos, lp_sos])
            return signal.butter(
                self.order,
                [self.low_hz / nyquist, self.high_hz / nyquist],
                btype="band",
                output="sos",
            )
        else:
            raise ValueError("At least one of low_hz or high_hz must be set")


# Virtual stem band configurations
# These are tuned for typical music production frequency ranges
VIRTUAL_STEM_BANDS: Dict[str, Dict[str, BandConfig]] = {
    "drums": {
        # Kick: 20-200 Hz captures fundamental + punch
        "drums_low": BandConfig(low_hz=20, high_hz=200, order=4),
        # Snare: 200-5000 Hz captures body (200-400) + crack (2-5k)
        # Perfect for jungle breaks where snare is rhythmically distinct
        "drums_mid": BandConfig(low_hz=200, high_hz=5000, order=4),
        # Hi-hats: 5-16 kHz captures shimmer without snare bleed
        "drums_high": BandConfig(low_hz=5000, high_hz=16000, order=4),
    },
    "other": {
        # Body: 200-4000 Hz captures melodic content + presence frequencies (2-4 kHz)
        "other_mid": BandConfig(low_hz=200, high_hz=4000, order=4),
        # Air: 4-16 kHz captures brightness + shimmer
        "other_high": BandConfig(low_hz=4000, high_hz=16000, order=4),
    },
}


def apply_bandpass(
    audio: np.ndarray,
    sr: int,
    config: BandConfig,
) -> np.ndarray:
    """Apply bandpass filter to audio signal.

    Uses scipy.signal.sosfiltfilt for zero-phase filtering
    (no phase distortion, but not causal - fine for offline processing).

    Args:
        audio: Input audio, shape (n_samples,) or (n_samples, n_channels)
        sr: Sample rate in Hz
        config: Band configuration

    Returns:
        Filtered audio, same shape as input
    """
    sos = config.to_sos(sr)
    return signal.sosfiltfilt(sos, audio, axis=0)


def extract_virtual_stems(
    stems: Dict[str, np.ndarray],
    sr: int,
) -> Dict[str, np.ndarray]:
    """Extract virtual stems from physical stems via bandpass filtering.

    Takes the 4 Demucs stems and produces additional virtual stems:
    - drums → drums_low, drums_high
    - other → other_mid, other_high

    Args:
        stems: {"bass": array, "drums": array, "vocals": array, "other": array}
        sr: Sample rate in Hz

    Returns:
        Dict including both original and virtual stems:
        {"bass": ..., "drums": ..., "vocals": ..., "other": ...,
         "drums_low": ..., "drums_high": ..., "other_mid": ..., "other_high": ...}
    """
    result = dict(stems)  # Copy original stems

    for source_stem, bands in VIRTUAL_STEM_BANDS.items():
        if source_stem not in stems:
            continue

        source_audio = stems[source_stem]
        for virtual_name, band_config in bands.items():
            result[virtual_name] = apply_bandpass(source_audio, sr, band_config)

    return result


def get_virtual_stem_names() -> Tuple[str, ...]:
    """Get all virtual stem names (excluding physical stems)."""
    names = []
    for bands in VIRTUAL_STEM_BANDS.values():
        names.extend(bands.keys())
    return tuple(names)


def get_all_stem_names() -> Tuple[str, ...]:
    """Get all stem names (physical + virtual)."""
    physical = ("bass", "drums", "vocals", "other")
    virtual = get_virtual_stem_names()
    return physical + virtual


def get_virtual_parent_stem(virtual_name: str) -> str | None:
    """Get the parent physical stem for a virtual stem name."""
    for parent, bands in VIRTUAL_STEM_BANDS.items():
        if virtual_name in bands:
            return parent
    return None
