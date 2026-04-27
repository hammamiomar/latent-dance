"""Audio feature extraction with perceptual processing.

This module extracts time-series features from audio stems for
real-time audio-reactive visualization. All features are computed
offline at upload time, enabling O(1) sampling at runtime.

Design principles:
- Perceptually meaningful: asymmetric attack/release, spectral flux
- Frame-aligned: all channels share timestamps for direct indexing
- Normalized: all values in [0, 1] for consistent steering
- Offline-first: expensive DSP runs once, runtime is just array lookup
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import librosa
import numpy as np

try:  # Optional torch backend
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in tests
    torch = None

from .harmonic import (
    compute_chroma,
    compute_chroma_centroid,
    compute_tension,
    compute_tonal_distance,
)
from .hpss import compute_hpss_components, compute_normalized_flux, compute_energy_db
from .perceptual import (
    asymmetric_envelope_follow,
    compute_dual_layer,
    compute_onset_strength,
    compute_spectral_flatness,
    detect_peaks,
    get_brightness_preset,
    get_dual_layer_preset,
    get_envelope_preset,
    normalize_feature,
)
from .pitch import extract_pitch_pesto, extract_pitch_polyphonic, normalize_pitch
from .spectral import compute_spectrogram
from .structure import compute_structure_features
from .util import align_1d
from .virtual_stems import get_virtual_parent_stem, get_virtual_stem_names

if TYPE_CHECKING:
    from .coupling import CrossStemFeatures

logger = logging.getLogger(__name__)

# Bump when extraction logic changes to invalidate cached features
FEATURE_CACHE_VERSION = 5

# Field lists for NPZ serialization
_STEM_REQUIRED = [
    "envelope", "energy_smooth", "transient", "flux", "brightness",
    "flatness", "flash", "sustain", "onsets", "timestamps",
]
_STEM_SCALARS = ["duration", "fps", "hpss_ratio", "tempo"]
_STEM_OPTIONAL_1D = [
    "flux_normalized", "energy_db", "harmonic_energy", "percussive_energy",
    "tension", "tonal_distance",
    "chroma_centroid",
    "pitch_hz", "pitch_confidence", "pitch_normalized",
    "novelty_short", "novelty_medium", "novelty_long",
    "novelty_short_deriv", "novelty_medium_deriv",
    "beat_frames", "energy_at_beats",
]
_STEM_OPTIONAL_2D = ["chroma"]  # (12, T) chromagram
_STEM_OPTIONAL_BOOL = ["layer_entry_mask"]


@dataclass
class StemFeatures:
    """Perceptually-optimized features for a single audio stem.

    All time-series arrays are frame-aligned (same length, same timestamps).
    Values are normalized to [0, 1] for direct use as steering multipliers.

    Primary Channels:
        envelope: Raw RMS energy (symmetric, unsmoothed)
        energy_smooth: Asymmetrically smoothed energy (fast attack, slow release)
        transient: Binary peak mask (1.0 at detected peaks, 0.0 elsewhere)
        flux: Spectral flux / onset strength (measures spectral change)
        brightness: Smoothed spectral centroid (timbral brightness)
        flatness: Spectral flatness (0=tonal, 1=noise-like) for Bouba/Kiki mapping

    Dual-Layer Channels (for comet-tail effects):
        flash: Very fast attack/release - captures immediate transient "pop"
        sustain: Slower attack/release - captures trailing "glow"

    Metadata:
        onsets: Sparse onset times in seconds (for beat sync)
        timestamps: Time axis in seconds
        duration: Total track duration
        fps: Frame rate of feature arrays
    """

    envelope: np.ndarray
    energy_smooth: np.ndarray
    transient: np.ndarray
    flux: np.ndarray
    brightness: np.ndarray
    flatness: np.ndarray
    flash: np.ndarray
    sustain: np.ndarray
    onsets: np.ndarray
    timestamps: np.ndarray
    duration: float
    fps: float

    # Optional HPSS-derived fields (Phase 1 extension)
    hpss_ratio: float | None = None  # Percussive energy ratio (0=harmonic, 1=percussive)
    flux_normalized: np.ndarray | None = None  # Amplitude-normalized flux (layer detection)
    energy_db: np.ndarray | None = None  # Energy in dB (activity gating)

    # HPSS component time-series (v2: separate link targets)
    harmonic_energy: np.ndarray | None = None  # (T,) Harmonic component energy [0,1]
    percussive_energy: np.ndarray | None = None  # (T,) Percussive component energy [0,1]

    # Harmonic features (Phase 2)
    tension: np.ndarray | None = None  # (T,) Combined roughness/entropy tension [0,1]
    tonal_distance: np.ndarray | None = None  # (T,) JSD from track-average chroma [0,1]
    chroma: np.ndarray | None = None  # (12, T) Chromagram pitch class representation
    chroma_centroid: np.ndarray | None = None  # (T,) Circular mean of chroma bins [0,1]

    # Pitch tracking (Phase 3)
    pitch_hz: np.ndarray | None = None  # (T,) Hz, 0=unvoiced
    pitch_confidence: np.ndarray | None = None  # (T,) 0-1 voicing confidence
    pitch_normalized: np.ndarray | None = None  # (T,) 0-1 scaled for spatial mapping

    # Structural features (§8: structural awareness)
    novelty_short: np.ndarray | None = None  # (T,) 0-1, transients/fills (0.5-2s)
    novelty_medium: np.ndarray | None = None  # (T,) 0-1, phrase boundaries (4-8s)
    novelty_long: np.ndarray | None = None  # (T,) 0-1, section changes (16-32s)
    novelty_short_deriv: np.ndarray | None = None  # (T,) rate of change, burst trigger
    novelty_medium_deriv: np.ndarray | None = None  # (T,) rate of change, movement trigger
    layer_entry_mask: np.ndarray | None = None  # (T,) bool, new instrument entries

    # Beat tracking (for MotionEngine)
    beat_frames: np.ndarray | None = None  # (N,) beat frame indices
    tempo: float | None = None  # Detected BPM
    energy_at_beats: np.ndarray | None = None  # (N,) energy values at beat frames

    # Channel lookup for generic access
    _CHANNELS: dict = None

    def __post_init__(self):
        channels = {
            "envelope": self.envelope,
            "energy_smooth": self.energy_smooth,
            "transient": self.transient,
            "flux": self.flux,
            "brightness": self.brightness,
            "flatness": self.flatness,
            "flash": self.flash,
            "sustain": self.sustain,
        }
        # Add optional Phase 1 channels if present
        if self.flux_normalized is not None:
            channels["flux_normalized"] = self.flux_normalized
        if self.energy_db is not None:
            channels["energy_db"] = self.energy_db
        # HPSS component time-series (v2)
        if self.harmonic_energy is not None:
            channels["harmonic_energy"] = self.harmonic_energy
        if self.percussive_energy is not None:
            channels["percussive_energy"] = self.percussive_energy
        # Add optional Phase 2 (harmonic) channels if present
        if self.tension is not None:
            channels["tension"] = self.tension
        if self.tonal_distance is not None:
            channels["tonal_distance"] = self.tonal_distance
        if self.chroma_centroid is not None:
            channels["chroma_centroid"] = self.chroma_centroid
        # Add optional Phase 3 (pitch) channels if present
        if self.pitch_hz is not None:
            channels["pitch_hz"] = self.pitch_hz
        if self.pitch_confidence is not None:
            channels["pitch_confidence"] = self.pitch_confidence
        if self.pitch_normalized is not None:
            channels["pitch_normalized"] = self.pitch_normalized
        # Add structural features (§8) if present
        if self.novelty_short is not None:
            channels["novelty_short"] = self.novelty_short
        if self.novelty_medium is not None:
            channels["novelty_medium"] = self.novelty_medium
        if self.novelty_long is not None:
            channels["novelty_long"] = self.novelty_long
        if self.novelty_short_deriv is not None:
            channels["novelty_short_deriv"] = self.novelty_short_deriv
        if self.novelty_medium_deriv is not None:
            channels["novelty_medium_deriv"] = self.novelty_medium_deriv
        object.__setattr__(self, '_CHANNELS', channels)

    # Known optional channels that may not be populated
    _OPTIONAL_CHANNELS = frozenset({
        "flux_normalized", "energy_db",  # Phase 1
        "harmonic_energy", "percussive_energy",  # Phase 1 v2
        "tension", "tonal_distance", "chroma_centroid",  # Phase 2
        "pitch_hz", "pitch_confidence", "pitch_normalized",  # Phase 3
        "novelty_short", "novelty_medium", "novelty_long",  # §8 structural
        "novelty_short_deriv", "novelty_medium_deriv",  # §8 derivatives
    })

    def sample_at_time(self, t: float, channel: str = "energy_smooth") -> float:
        """Interpolate channel value at time t (seconds).

        Returns 0.0 for optional channels that weren't computed.
        """
        data = self._CHANNELS.get(channel)
        if data is None:
            # Check if it's a known optional channel that wasn't populated
            if channel in self._OPTIONAL_CHANNELS:
                return 0.0
            raise ValueError(f"Unknown channel: {channel}")
        t_clamped = np.clip(t, 0.0, self.duration)
        return float(np.interp(t_clamped, self.timestamps, data))

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    def to_npz(self, path: Path) -> None:
        """Save all arrays to compressed NPZ for disk caching."""
        arrays = {}

        # Required arrays (always present)
        for name in _STEM_REQUIRED:
            arrays[name] = getattr(self, name)

        # Scalars as 0-d arrays
        for name in _STEM_SCALARS:
            val = getattr(self, name)
            if val is not None:
                arrays[name] = np.array(val)

        # Optional 1D arrays (skip None)
        for name in _STEM_OPTIONAL_1D:
            val = getattr(self, name)
            if val is not None:
                arrays[name] = val

        # Optional 2D arrays (e.g., chroma)
        for name in _STEM_OPTIONAL_2D:
            val = getattr(self, name)
            if val is not None:
                arrays[name] = val

        # Optional bool arrays (stored as uint8 for NPZ compatibility)
        for name in _STEM_OPTIONAL_BOOL:
            val = getattr(self, name)
            if val is not None:
                arrays[name] = val.astype(np.uint8)

        np.savez_compressed(path, **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> "StemFeatures":
        """Load StemFeatures from compressed NPZ file.

        Uses allow_pickle=False for security - all data is stored as
        native NumPy arrays without Python object serialization.
        """
        data = np.load(path, allow_pickle=False)

        kwargs = {}

        # Required arrays
        for name in _STEM_REQUIRED:
            kwargs[name] = data[name]

        # Scalars (stored as 0-d arrays)
        for name in _STEM_SCALARS:
            if name in data:
                kwargs[name] = float(data[name])
            else:
                kwargs[name] = None

        # Optional 1D arrays
        for name in _STEM_OPTIONAL_1D:
            kwargs[name] = data[name] if name in data else None

        # Optional 2D arrays
        for name in _STEM_OPTIONAL_2D:
            kwargs[name] = data[name] if name in data else None

        # Optional bool arrays (stored as uint8)
        for name in _STEM_OPTIONAL_BOOL:
            if name in data:
                kwargs[name] = data[name].astype(bool)
            else:
                kwargs[name] = None

        return cls(**kwargs)


class StemAnalyzer:
    """Extract perceptual features from a mono audio signal.

    All heavy DSP (FFT, onset detection, envelope following) runs here
    at upload time. The resulting StemFeatures enables O(1) runtime access.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        fps: int = 60,
        stem_name: str = "default",
        hpss_ratio: float | None = None,
        bpm: float = 120.0,
        feature_level: str = "full",
        feature_backend: str = "auto",
        feature_device: str = "auto",
        chroma_mode: str = "stft",
        hpss_backend: str = "auto",
    ):
        """
        Args:
            audio: Mono audio signal, shape (n_samples,)
            sr: Sample rate in Hz
            fps: Target frame rate for feature extraction
            stem_name: Stem type for envelope preset selection
            hpss_ratio: Optional percussive ratio from HPSS (0=harmonic, 1=percussive).
                        If provided, harmonic features are skipped for purely percussive stems.
            bpm: Detected tempo (for BPM-scaled novelty/structure windows)
            feature_level: "core" for faster extraction, "full" for all features,
                "minimal" for lightweight virtual stems
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio (1D), got shape {audio.shape}")

        self.audio = audio.astype(np.float32)
        self.sr = sr
        self.fps = fps
        self.stem_name = stem_name
        self.hpss_ratio = hpss_ratio
        self.bpm = bpm
        self.feature_level = feature_level
        self.hop_length = sr // fps
        self.duration = len(audio) / sr
        self.feature_backend = feature_backend
        self.feature_device = feature_device
        self.chroma_mode = chroma_mode
        self.hpss_backend = hpss_backend

    def extract(self) -> StemFeatures:
        """Extract all perceptual features. This is the main entry point.

        Delegates to phase methods for clarity:
        - _extract_perceptual: envelope, energy, flux, brightness, dual-layer
        - _extract_harmonic: tension, tonal_distance, chroma_centroid
        - _extract_pitch: pitch_hz, confidence, normalized
        - _extract_structure: novelty derivatives, layer_entry_mask
        - _extract_beat: beat_frames, tempo, energy_at_beats
        """
        t0 = time.perf_counter()
        self._t_last = t0

        timestamps = self._compute_timestamps()
        n_frames = len(timestamps)

        # Shared spectrogram cache (reused by HPSS, tension, flux, flatness, centroid)
        mag, freqs, mag_np, freqs_np = None, None, None, None
        if self.feature_level != "minimal":
            spectrogram = compute_spectrogram(
                self.audio,
                self.sr,
                self.hop_length,
                backend=self.feature_backend,
                device=self.feature_device,
            )
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Features[%s]: spectral backend=%s device=%s",
                    self.stem_name,
                    spectrogram.backend,
                    spectrogram.device,
                )
            mag = spectrogram.mag
            freqs = spectrogram.freqs
            if torch is not None and isinstance(mag, torch.Tensor):
                mag_np = mag.detach().cpu().numpy()
                freqs_np = freqs.detach().cpu().numpy()
            else:
                mag_np = mag
                freqs_np = freqs

        is_full = self.feature_level == "full"
        is_minimal = self.feature_level == "minimal"

        # Phase 1: Perceptual features (envelope, flux, brightness, flatness, dual-layer)
        perceptual = self._extract_perceptual(n_frames, mag_np=mag_np, freqs_np=freqs_np)

        # HPSS component time-series (harmonic/percussive energy)
        harmonic_energy = None
        percussive_energy = None
        computed_hpss_ratio = self.hpss_ratio
        has_harmonic_content = False
        if not is_minimal:
            hpss_components = compute_hpss_components(
                self.audio,
                self.sr,
                self.hop_length,
                mag=mag,
                backend=self.hpss_backend,
            )
            harmonic_energy = self._align(hpss_components.harmonic_energy, n_frames)
            percussive_energy = self._align(hpss_components.percussive_energy, n_frames)
            computed_hpss_ratio = self.hpss_ratio if self.hpss_ratio is not None else hpss_components.hpss_ratio
            has_harmonic_content = computed_hpss_ratio < 0.7
            self._log_phase("hpss")

        # Phase 2: Harmonic features (tension, tonal_distance, chroma)
        harmonic = self._extract_harmonic(
            n_frames,
            has_harmonic_content=has_harmonic_content,
            is_full=is_full,
            mag=mag,
            freqs=freqs,
            mag_np=mag_np,
        )

        # Phase 3: Pitch tracking (melodic stems only)
        pitch = self._extract_pitch(
            n_frames,
            has_harmonic_content=has_harmonic_content,
            is_full=is_full,
        )

        # Phase 4: Structural features (novelty + layer detection)
        structure_result = self._extract_structure(
            n_frames,
            is_minimal=is_minimal,
            flatness=perceptual["flatness"],
            mag_np=mag_np,
        )

        # Phase 5: Beat tracking (drums only)
        beat = self._extract_beat(
            is_minimal=is_minimal,
            energy_smooth=perceptual["energy_smooth"],
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Features[%s]: total %.1fms",
                self.stem_name,
                (time.perf_counter() - t0) * 1000.0,
            )

        structure = structure_result.get("structure")
        return StemFeatures(
            # Perceptual
            envelope=perceptual["envelope"],
            energy_smooth=perceptual["energy_smooth"],
            transient=perceptual["transient"],
            flux=perceptual["flux"],
            brightness=perceptual["brightness"],
            flatness=perceptual["flatness"],
            flash=perceptual["flash"],
            sustain=perceptual["sustain"],
            onsets=perceptual["onsets"],
            timestamps=timestamps,
            duration=self.duration,
            fps=self.fps,
            # HPSS (Phase 1)
            hpss_ratio=computed_hpss_ratio,
            harmonic_energy=harmonic_energy,
            percussive_energy=percussive_energy,
            # Harmonic features (Phase 2)
            tension=harmonic["tension"],
            tonal_distance=harmonic["tonal_distance"],
            chroma=harmonic["chroma"],
            chroma_centroid=harmonic["chroma_centroid"],
            # Pitch tracking (Phase 3)
            pitch_hz=pitch["pitch_hz"],
            pitch_confidence=pitch["pitch_confidence"],
            pitch_normalized=pitch["pitch_normalized"],
            # Structural features (Phase 4)
            flux_normalized=structure_result.get("flux_normalized"),
            energy_db=structure_result.get("energy_db"),
            novelty_short=structure["novelty_short"] if structure else None,
            novelty_medium=structure["novelty_medium"] if structure else None,
            novelty_long=structure["novelty_long"] if structure else None,
            novelty_short_deriv=structure["novelty_short_deriv"] if structure else None,
            novelty_medium_deriv=structure["novelty_medium_deriv"] if structure else None,
            layer_entry_mask=structure["layer_entry_mask"] if structure else None,
            # Beat tracking (Phase 5)
            beat_frames=beat["beat_frames"],
            tempo=beat["tempo"],
            energy_at_beats=beat["energy_at_beats"],
        )

    # ── Phase extraction methods ──────────────────────────────────────────

    def _log_phase(self, label: str) -> None:
        """Log timing for a feature extraction phase."""
        if logger.isEnabledFor(logging.INFO):
            now = time.perf_counter()
            logger.info(
                "Features[%s]: %s %.1fms",
                self.stem_name,
                label,
                (now - self._t_last) * 1000.0,
            )
            self._t_last = now

    def _extract_perceptual(
        self,
        n_frames: int,
        *,
        mag_np: np.ndarray | None,
        freqs_np: np.ndarray | None,
    ) -> dict:
        """Extract envelope, energy, flux, brightness, flatness, dual-layer, onsets."""
        # Raw RMS envelope
        envelope = self._compute_rms()
        envelope = self._align(envelope, n_frames)

        # Asymmetric smoothing (stem-specific preset from JSON)
        config = get_envelope_preset(self.stem_name)
        energy_smooth = asymmetric_envelope_follow(envelope, self.fps, config)
        self._log_phase("envelope")

        # Spectral flux (onset strength) - more discriminative for transients
        flux = compute_onset_strength(self.audio, self.sr, self.hop_length)
        flux = self._align(flux, n_frames)
        flux = normalize_feature(flux)

        # Peak detection on flux (NOT energy_smooth) - flux better detects rhythmic events
        transient = detect_peaks(flux, self.fps)
        self._log_phase("flux+transient")

        # Spectral centroid -> brightness (stem-specific preset from JSON)
        centroid = self._compute_centroid(mag=mag_np, freqs=freqs_np)
        centroid = self._align(centroid, n_frames)
        brightness_config = get_brightness_preset(self.stem_name)
        brightness = asymmetric_envelope_follow(centroid, self.fps, brightness_config)

        # Spectral flatness (0=tonal, 1=noise-like) for Bouba/Kiki audio-visual mapping
        flatness = compute_spectral_flatness(
            self.audio, self.sr, self.hop_length, mag=mag_np
        )
        flatness = self._align(flatness, n_frames)

        # Dual-layer response (flash + sustain, preset from JSON)
        dual_config = get_dual_layer_preset(self.stem_name)
        flash, sustain = compute_dual_layer(envelope, self.fps, dual_config)

        # Onset times (sparse)
        onsets = self._detect_onsets()
        self._log_phase("centroid+flatness+dual+onsets")

        return {
            "envelope": envelope,
            "energy_smooth": energy_smooth,
            "transient": transient,
            "flux": flux,
            "brightness": brightness,
            "flatness": flatness,
            "flash": flash,
            "sustain": sustain,
            "onsets": onsets,
        }

    def _extract_harmonic(
        self,
        n_frames: int,
        *,
        has_harmonic_content: bool,
        is_full: bool,
        mag,
        freqs,
        mag_np: np.ndarray | None,
    ) -> dict:
        """Extract tension, tonal_distance, chroma, chroma_centroid."""
        result = {
            "chroma": None,
            "tension": None,
            "tonal_distance": None,
            "chroma_centroid": None,
        }

        if not has_harmonic_content:
            return result

        chroma = None
        if is_full:
            chroma = compute_chroma(
                self.audio,
                self.sr,
                self.hop_length,
                mag=mag_np,
                mode=self.chroma_mode,
            )
            chroma = self._align_2d(chroma, n_frames)
            self._log_phase("chroma")

            result["chroma"] = chroma

            chroma_centroid_arr = compute_chroma_centroid(chroma)
            result["chroma_centroid"] = self._align(chroma_centroid_arr, n_frames)

        # Tension: psychoacoustic roughness with entropy fallback
        tension = compute_tension(
            self.audio,
            self.sr,
            self.hop_length,
            mag=mag,
            freqs=freqs,
        )
        result["tension"] = self._align(tension, n_frames)

        # Tonal distance: JSD from track-average chroma for prompt SLERP
        if chroma is None:
            chroma_for_tonal = compute_chroma(
                self.audio,
                self.sr,
                self.hop_length,
                mag=mag_np,
                mode=self.chroma_mode,
            )
            chroma_for_tonal = self._align_2d(chroma_for_tonal, n_frames)
        else:
            chroma_for_tonal = chroma
        tonal_distance = compute_tonal_distance(chroma_for_tonal, fps=self.fps)
        result["tonal_distance"] = self._align(tonal_distance, n_frames)
        self._log_phase("harmonic")

        return result

    def _extract_pitch(
        self,
        n_frames: int,
        *,
        has_harmonic_content: bool,
        is_full: bool,
    ) -> dict:
        """Extract pitch_hz, confidence, normalized."""
        result = {
            "pitch_hz": None,
            "pitch_confidence": None,
            "pitch_normalized": None,
        }

        is_melodic_stem = self.stem_name in ("vocals", "bass", "other")
        if not (is_melodic_stem and has_harmonic_content):
            return result

        pitch_hz = None
        pitch_confidence = None

        if self.stem_name == "other":
            # Polyphonic: piano, synths, chords
            if is_full:
                pitch_hz, pitch_confidence, _ = extract_pitch_polyphonic(
                    self.audio, self.sr, self.hop_length
                )
        else:
            # Monophonic: vocals, bass
            pitch_hz, pitch_confidence = extract_pitch_pesto(
                self.audio, self.sr, self.hop_length
            )

        if pitch_hz is None or pitch_confidence is None:
            self._log_phase("pitch")
            return result

        # Derived pitch features
        pitch_normalized = normalize_pitch(pitch_hz, pitch_confidence)

        # Align all pitch arrays to frame count
        result["pitch_hz"] = self._align(pitch_hz, n_frames)
        result["pitch_confidence"] = self._align(pitch_confidence, n_frames)
        result["pitch_normalized"] = self._align(pitch_normalized, n_frames)

        self._log_phase("pitch")
        return result

    def _extract_structure(
        self,
        n_frames: int,
        *,
        is_minimal: bool,
        flatness: np.ndarray,
        mag_np: np.ndarray | None,
    ) -> dict:
        """Extract multi-timescale novelty, layer detection, flux_normalized, energy_db."""
        result: dict = {
            "flux_normalized": None,
            "energy_db": None,
            "structure": None,
        }

        if is_minimal:
            return result

        flux_normalized = compute_normalized_flux(
            self.audio, self.sr, self.hop_length, mag=mag_np
        )
        flux_normalized = self._align(flux_normalized, n_frames)
        energy_db = compute_energy_db(self.audio, self.sr, self.hop_length)
        energy_db = self._align(energy_db, n_frames)

        structure = compute_structure_features(
            flux_normalized=flux_normalized,
            flatness=flatness,
            energy_db=energy_db,
            fps=self.fps,
            bpm=self.bpm,
        )
        self._log_phase("structure")

        result["flux_normalized"] = flux_normalized
        result["energy_db"] = energy_db
        result["structure"] = structure
        return result

    def _extract_beat(
        self,
        *,
        is_minimal: bool,
        energy_smooth: np.ndarray,
    ) -> dict:
        """Extract beat_frames, tempo, energy_at_beats (drums stem only)."""
        result: dict = {
            "beat_frames": None,
            "tempo": None,
            "energy_at_beats": None,
        }

        if is_minimal or self.stem_name != "drums":
            return result

        tempo_val, beat_frame_indices = librosa.beat.beat_track(
            y=self.audio, sr=self.sr, hop_length=self.hop_length
        )
        tempo_val = float(tempo_val) if np.isscalar(tempo_val) else float(tempo_val[0])
        beat_frames_arr = beat_frame_indices.astype(np.int32)

        # Sample energy at each beat frame for MotionEngine travel distance
        energy_at_beats_arr = None
        if len(beat_frames_arr) > 0 and energy_smooth is not None:
            beat_indices_clamped = np.clip(beat_frames_arr, 0, len(energy_smooth) - 1)
            energy_at_beats_arr = energy_smooth[beat_indices_clamped].astype(np.float32)
        self._log_phase("beats")

        result["beat_frames"] = beat_frames_arr
        result["tempo"] = tempo_val
        result["energy_at_beats"] = energy_at_beats_arr
        return result

    def _compute_timestamps(self) -> np.ndarray:
        """Generate time axis at target FPS."""
        n_frames = 1 + len(self.audio) // self.hop_length
        return librosa.frames_to_time(
            np.arange(n_frames), sr=self.sr, hop_length=self.hop_length
        )

    def _compute_rms(self) -> np.ndarray:
        """Compute RMS envelope, normalized to [0, 1]."""
        rms = librosa.feature.rms(y=self.audio, hop_length=self.hop_length)[0]
        return normalize_feature(rms)

    def _compute_centroid(
        self,
        *,
        mag: np.ndarray | None = None,
        freqs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute spectral centroid, normalized to [0, 1]."""
        if mag is not None and freqs is not None:
            denom = np.maximum(mag.sum(axis=0), 1e-10)
            centroid = (freqs[:, None] * mag).sum(axis=0) / denom
        else:
            centroid = librosa.feature.spectral_centroid(
                y=self.audio, sr=self.sr, hop_length=self.hop_length
            )[0]
        return normalize_feature(centroid)

    def _detect_onsets(self) -> np.ndarray:
        """Detect onset times in seconds (sparse array)."""
        onset_frames = librosa.onset.onset_detect(
            y=self.audio, sr=self.sr, hop_length=self.hop_length, backtrack=True
        )
        return librosa.frames_to_time(
            onset_frames, sr=self.sr, hop_length=self.hop_length
        )

    def _align(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Align 1D array to target length (librosa outputs can vary by +-1)."""
        return align_1d(arr, target_len)

    def _align_2d(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Align 2D array (features, frames) to target frame count."""
        n_frames = arr.shape[1]
        if n_frames == target_len:
            return arr
        elif n_frames > target_len:
            return arr[:, :target_len]
        else:
            return np.pad(arr, ((0, 0), (0, target_len - n_frames)), mode="edge")


def extract_all_features(
    stems: Dict[str, np.ndarray],
    sr: int,
    fps: int = 60,
    bpm: float = 120.0,
    feature_level: str = "full",
    feature_backend: str = "auto",
    feature_device: str = "auto",
    chroma_mode: str = "stft",
    hpss_backend: str = "auto",
    coupling_stems: Optional[str] = None,
) -> Tuple[Dict[str, StemFeatures], "CrossStemFeatures | None"]:
    """Extract features for all stems in a track, with optional cross-stem coupling.

    Args:
        stems: {"bass": audio_array, "drums": audio_array, ...}
        sr: Sample rate (must match all stems)
        fps: Target frame rate
        bpm: Detected tempo (used for BPM-scaled novelty windows)
        feature_level: "core" (faster), "full" (all features). Virtual stems
            are automatically extracted in "minimal" mode for performance.
        coupling_stems: Which stems to include in cross-stem coupling.
            "physical" = only bass/drums/vocals/other.
            "all" = all stems including virtual.
            None = skip cross-stem coupling (backwards-compatible default).

    Returns:
        Tuple of (stem_features_dict, cross_stem_features_or_none).
    """
    from .coupling import extract_cross_stem_features

    virtual_stems = set(get_virtual_stem_names())
    features: Dict[str, StemFeatures] = {}
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Feature extraction: stems=%d, level=%s, sr=%d, fps=%d",
            len(stems),
            feature_level,
            sr,
            fps,
        )
    t0 = time.perf_counter()
    for name, audio in stems.items():
        level = "minimal" if name in virtual_stems else feature_level
        parent = get_virtual_parent_stem(name) if name in virtual_stems else None
        stem_t0 = time.perf_counter()
        features[name] = StemAnalyzer(
            audio,
            sr=sr,
            fps=fps,
            stem_name=name,
            bpm=bpm,
            feature_level=level,
            feature_backend=feature_backend,
            feature_device=feature_device,
            chroma_mode=chroma_mode,
            hpss_backend=hpss_backend,
            hpss_ratio=features[parent].hpss_ratio if parent and parent in features else None,
        ).extract()
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Feature extraction: %s level=%s %.1fms",
                name,
                level,
                (time.perf_counter() - stem_t0) * 1000.0,
            )

    # Cross-stem coupling (PLV, lock index, spectral overlap, call-response)
    cross_stem: "CrossStemFeatures | None" = None
    if coupling_stems is not None:
        physical_names = {"bass", "drums", "vocals", "other"}
        if coupling_stems == "physical":
            coupling_audio = {k: v for k, v in stems.items() if k in physical_names}
            coupling_feats = {k: v for k, v in features.items() if k in physical_names}
        else:
            coupling_audio = stems
            coupling_feats = features

        coupling_t0 = time.perf_counter()
        cross_stem = extract_cross_stem_features(
            stem_audio=coupling_audio,
            stem_features=coupling_feats,
            sr=sr,
            fps=float(fps),
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Cross-stem coupling: %.1fms",
                (time.perf_counter() - coupling_t0) * 1000.0,
            )

    if logger.isEnabledFor(logging.INFO):
        logger.info("Feature extraction: total %.1fms", (time.perf_counter() - t0) * 1000.0)
    return features, cross_stem
