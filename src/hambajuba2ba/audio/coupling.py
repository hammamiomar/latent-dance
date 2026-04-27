"""Cross-stem coupling detection.

Detects rhythmic relationships between stems:
- Phase Locking Value (PLV): how synchronized envelope phases are
- Lock Index: combined metric (envelope correlation + PLV + onset sync)
- Spectral Overlap: frequency competition for side-chaining
- Call-Response: alternating activity patterns

All functions are vectorized (no Python loops in hot paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import hilbert

if TYPE_CHECKING:
    from .features import StemFeatures


def compute_plv(
    env1: np.ndarray,
    env2: np.ndarray,
    window_samples: int = 100,
) -> np.ndarray:
    """Compute Phase Locking Value between two envelope signals.

    PLV measures phase synchrony: 0 = independent, 1 = perfectly locked.
    Uses Hilbert transform to extract instantaneous phase.

    Args:
        env1: First envelope signal (T,)
        env2: Second envelope signal (T,)
        window_samples: Sliding window size in frames

    Returns:
        PLV values (T,) in [0, 1]
    """
    # Extract instantaneous phase via Hilbert transform
    phase1 = np.angle(hilbert(env1))
    phase2 = np.angle(hilbert(env2))

    # Phase difference
    phase_diff = phase1 - phase2

    # Sliding window mean of exp(i * phase_diff) magnitude
    # PLV = |<exp(i * Δφ)>| where <> is temporal average
    cos_diff = np.cos(phase_diff)
    sin_diff = np.sin(phase_diff)

    cos_mean = uniform_filter1d(cos_diff, window_samples, mode="reflect")
    sin_mean = uniform_filter1d(sin_diff, window_samples, mode="reflect")

    plv = np.sqrt(cos_mean**2 + sin_mean**2).astype(np.float32)
    return plv


def sliding_correlation(
    data1: np.ndarray,
    data2: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Compute sliding window Pearson correlation.

    Args:
        data1: First signal (T,)
        data2: Second signal (T,)
        window_samples: Window size in frames

    Returns:
        Correlation values (T,) in [-1, 1]
    """
    # Compute rolling means
    mean1 = uniform_filter1d(data1.astype(np.float64), window_samples, mode="reflect")
    mean2 = uniform_filter1d(data2.astype(np.float64), window_samples, mode="reflect")

    # Centered signals
    c1 = data1 - mean1
    c2 = data2 - mean2

    # Rolling covariance and variances
    cov = uniform_filter1d(c1 * c2, window_samples, mode="reflect")
    var1 = uniform_filter1d(c1 * c1, window_samples, mode="reflect")
    var2 = uniform_filter1d(c2 * c2, window_samples, mode="reflect")

    # Correlation with small epsilon for numerical stability
    # Use np.maximum to avoid sqrt of negative values (can happen with near-constant signals)
    eps = 1e-8
    corr = cov / (np.sqrt(np.maximum(var1 * var2, 0.0)) + eps)

    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def compute_onset_synchrony(
    onsets1: np.ndarray,
    onsets2: np.ndarray,
    tolerance_frames: int,
    total_frames: int,
) -> np.ndarray:
    """Compute onset synchrony score over time.

    Measures fraction of onsets in each signal that have a matching
    onset in the other signal within tolerance. Vectorized implementation.

    Args:
        onsets1: Onset frame indices (sparse, M elements)
        onsets2: Onset frame indices (sparse, N elements)
        tolerance_frames: Match tolerance in frames
        total_frames: Total number of frames (T)

    Returns:
        Synchrony score (T,) in [0, 1]
    """
    sync = np.zeros(total_frames, dtype=np.float32)

    if len(onsets1) == 0 or len(onsets2) == 0:
        return sync

    # Vectorized: for each onset1, find min distance to any onset2
    # Broadcasting: (M, 1) - (1, N) = (M, N)
    distances = np.abs(onsets1[:, np.newaxis] - onsets2[np.newaxis, :])
    min_distances = distances.min(axis=1)  # (M,)

    # Mark matched onsets (within tolerance)
    matched_mask = min_distances <= tolerance_frames
    matched_onsets = onsets1[matched_mask]

    # Mark frames around matched onsets
    for onset in matched_onsets:
        start = max(0, onset - tolerance_frames)
        end = min(total_frames, onset + tolerance_frames + 1)
        sync[start:end] = 1.0

    # Smooth the binary mask
    sync = gaussian_filter1d(sync, sigma=tolerance_frames / 2)
    return np.clip(sync, 0.0, 1.0).astype(np.float32)


def compute_lock_index(
    env1: np.ndarray,
    env2: np.ndarray,
    onsets1: np.ndarray,
    onsets2: np.ndarray,
    fps: float = 50.0,
    window_sec: float = 2.0,
) -> np.ndarray:
    """Compute combined lock index (groove tightness).

    Combines three metrics:
    - Envelope correlation (40%): amplitude shape similarity
    - Phase Locking Value (35%): phase synchrony
    - Onset synchrony (25%): timing alignment

    Result > 0.7 = "in the pocket", < 0.3 = independent

    Args:
        env1: First envelope signal (T,)
        env2: Second envelope signal (T,)
        onsets1: First onset frame indices
        onsets2: Second onset frame indices
        fps: Frames per second
        window_sec: Window size in seconds

    Returns:
        Lock index (T,) in [0, 1]
    """
    window_samples = int(window_sec * fps)
    tolerance_frames = int(0.05 * fps)  # 50ms tolerance
    total_frames = len(env1)

    # Component 1: Envelope correlation (40%)
    corr = sliding_correlation(env1, env2, window_samples)
    corr_norm = (corr + 1.0) / 2.0  # Map [-1,1] to [0,1]

    # Component 2: Phase Locking Value (35%)
    plv = compute_plv(env1, env2, window_samples)

    # Component 3: Onset synchrony (25%)
    sync = compute_onset_synchrony(onsets1, onsets2, tolerance_frames, total_frames)

    # Weighted combination
    lock_index = 0.40 * corr_norm + 0.35 * plv + 0.25 * sync

    return lock_index.astype(np.float32)


def _mean_power_spectrum(
    audio: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
) -> np.ndarray:
    """Compute mean power spectrum for reuse across pairs."""
    import librosa

    S = np.abs(librosa.stft(audio, n_fft=n_fft)) ** 2
    return S.mean(axis=1)


def _spectral_overlap_from_spectra(
    spec1: np.ndarray,
    spec2: np.ndarray,
) -> float:
    """Compute overlap from pre-normalized spectra."""
    overlap = 2.0 * np.minimum(spec1, spec2).sum()
    return float(np.clip(overlap, 0.0, 1.0))


def detect_call_response(
    env1: np.ndarray,
    env2: np.ndarray,
    threshold: float = 0.3,
    window_samples: int = 50,
) -> np.ndarray:
    """Detect call-and-response patterns between stems.

    Measures alternating activity: high when one stem is active
    while the other is quiet. Used for focus alternation.

    Args:
        env1: First envelope (T,)
        env2: Second envelope (T,)
        threshold: Activity threshold (0-1)
        window_samples: Smoothing window

    Returns:
        Call-response score (T,) in [0, 1]
    """
    # Binary activity masks
    active1 = (env1 > threshold).astype(np.float32)
    active2 = (env2 > threshold).astype(np.float32)

    # XOR: high when exactly one is active
    alternation = np.abs(active1 - active2)

    # Smooth to get continuous score
    score = gaussian_filter1d(alternation, sigma=window_samples / 4)

    return np.clip(score, 0.0, 1.0).astype(np.float32)


@dataclass
class CrossStemFeatures:
    """Container for cross-stem relationship metrics.

    All matrices are indexed by stem name pairs.
    Time-varying matrices have shape (T,) per pair.

    Attributes:
        plv: Phase Locking Value per stem pair (symmetric)
        lock_index: Combined lock metric per stem pair (symmetric)
        spectral_overlap: Static overlap matrix (n_stems, n_stems)
        call_response: Alternation score per stem pair (symmetric)
        stem_names: Ordered list of stem names
        fps: Frames per second
    """

    plv: Dict[Tuple[str, str], np.ndarray]
    lock_index: Dict[Tuple[str, str], np.ndarray]
    spectral_overlap: np.ndarray  # (n_stems, n_stems)
    call_response: Dict[Tuple[str, str], np.ndarray]
    stem_names: list
    fps: float

    def get_plv(self, stem1: str, stem2: str, frame: int) -> float:
        """Get PLV between two stems at a frame."""
        key = (stem1, stem2) if (stem1, stem2) in self.plv else (stem2, stem1)
        if key not in self.plv:
            return 0.0
        arr = self.plv[key]
        idx = min(max(0, frame), len(arr) - 1)
        return float(arr[idx])

    def get_lock_index(self, stem1: str, stem2: str, frame: int) -> float:
        """Get lock index between two stems at a frame."""
        key = (stem1, stem2) if (stem1, stem2) in self.lock_index else (stem2, stem1)
        if key not in self.lock_index:
            return 0.0
        arr = self.lock_index[key]
        idx = min(max(0, frame), len(arr) - 1)
        return float(arr[idx])

    def get_spectral_overlap(self, stem1: str, stem2: str) -> float:
        """Get spectral overlap between two stems (static)."""
        try:
            i = self.stem_names.index(stem1)
            j = self.stem_names.index(stem2)
            return float(self.spectral_overlap[i, j])
        except (ValueError, IndexError):
            return 0.0

    def get_call_response(self, stem1: str, stem2: str, frame: int) -> float:
        """Get call-response score between two stems at a frame."""
        key = (stem1, stem2) if (stem1, stem2) in self.call_response else (stem2, stem1)
        if key not in self.call_response:
            return 0.0
        arr = self.call_response[key]
        idx = min(max(0, frame), len(arr) - 1)
        return float(arr[idx])

    def to_npz(self, path: Path) -> None:
        """Save to compressed NPZ for disk caching.

        Dict fields use flat key naming to avoid serializing Python objects:
        plv[(bass, drums)] → plv__bass__drums
        """
        arrays = {
            "spectral_overlap": self.spectral_overlap,
            "fps": np.array(self.fps),
            # Store stem_names as fixed-length string array
            "stem_names": np.array(self.stem_names, dtype="U32"),
        }

        # Flatten dict keys: (stem_a, stem_b) → prefix__stem_a__stem_b
        for (stem_a, stem_b), arr in self.plv.items():
            arrays[f"plv__{stem_a}__{stem_b}"] = arr
        for (stem_a, stem_b), arr in self.lock_index.items():
            arrays[f"lock_index__{stem_a}__{stem_b}"] = arr
        for (stem_a, stem_b), arr in self.call_response.items():
            arrays[f"call_response__{stem_a}__{stem_b}"] = arr

        np.savez_compressed(path, **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> "CrossStemFeatures":
        """Load CrossStemFeatures from compressed NPZ file."""
        data = np.load(path, allow_pickle=False)

        plv: Dict[Tuple[str, str], np.ndarray] = {}
        lock_index: Dict[Tuple[str, str], np.ndarray] = {}
        call_response: Dict[Tuple[str, str], np.ndarray] = {}

        for key in data.files:
            if key.startswith("plv__"):
                _, stem_a, stem_b = key.split("__")
                plv[(stem_a, stem_b)] = data[key]
            elif key.startswith("lock_index__"):
                _, stem_a, stem_b = key.split("__")
                lock_index[(stem_a, stem_b)] = data[key]
            elif key.startswith("call_response__"):
                _, stem_a, stem_b = key.split("__")
                call_response[(stem_a, stem_b)] = data[key]

        return cls(
            plv=plv,
            lock_index=lock_index,
            spectral_overlap=data["spectral_overlap"],
            call_response=call_response,
            stem_names=list(data["stem_names"]),
            fps=float(data["fps"]),
        )


def extract_cross_stem_features(
    stem_audio: Dict[str, np.ndarray],
    stem_features: Dict[str, "StemFeatures"],
    sr: int = 22050,
    fps: float = 50.0,
) -> CrossStemFeatures:
    """Extract all cross-stem coupling features.

    Computes pairwise metrics for all stem combinations.
    Only upper triangle is computed (symmetric relationships).

    Args:
        stem_audio: Raw audio per stem (for spectral overlap)
        stem_features: Extracted StemFeatures per stem
        sr: Sample rate
        fps: Frames per second

    Returns:
        CrossStemFeatures with all pairwise metrics
    """
    stem_names = sorted(stem_features.keys())
    n_stems = len(stem_names)

    plv_dict: Dict[Tuple[str, str], np.ndarray] = {}
    lock_dict: Dict[Tuple[str, str], np.ndarray] = {}
    call_response_dict: Dict[Tuple[str, str], np.ndarray] = {}
    spectral_overlap = np.zeros((n_stems, n_stems), dtype=np.float32)

    # Diagonal is 1.0 for spectral overlap (100% overlap with self)
    np.fill_diagonal(spectral_overlap, 1.0)

    # Pre-compute normalized power spectra (avoid STFT per pair)
    spectra: Dict[str, np.ndarray] = {}
    for name in stem_names:
        audio = stem_audio.get(name)
        if audio is None:
            continue
        spec = _mean_power_spectrum(audio, sr=sr)
        spectra[name] = spec / (spec.sum() + 1e-8)

    for i, name1 in enumerate(stem_names):
        feat1 = stem_features[name1]

        # Convert onset times (seconds) to frame indices
        onsets1_frames = (feat1.onsets * fps).astype(np.int32)

        for j, name2 in enumerate(stem_names):
            if j <= i:  # Skip diagonal and lower triangle
                continue

            feat2 = stem_features[name2]

            # Convert onset times to frame indices
            onsets2_frames = (feat2.onsets * fps).astype(np.int32)

            # Time-varying metrics using energy_smooth envelopes
            plv_dict[(name1, name2)] = compute_plv(
                feat1.energy_smooth, feat2.energy_smooth
            )
            lock_dict[(name1, name2)] = compute_lock_index(
                feat1.energy_smooth,
                feat2.energy_smooth,
                onsets1_frames,
                onsets2_frames,
                fps,
            )
            call_response_dict[(name1, name2)] = detect_call_response(
                feat1.energy_smooth, feat2.energy_smooth
            )

            # Static spectral overlap (requires precomputed spectra)
            spec1 = spectra.get(name1)
            spec2 = spectra.get(name2)
            if spec1 is not None and spec2 is not None:
                overlap = _spectral_overlap_from_spectra(spec1, spec2)
                spectral_overlap[i, j] = overlap
                spectral_overlap[j, i] = overlap  # Symmetric

    return CrossStemFeatures(
        plv=plv_dict,
        lock_index=lock_dict,
        spectral_overlap=spectral_overlap,
        call_response=call_response_dict,
        stem_names=stem_names,
        fps=fps,
    )
