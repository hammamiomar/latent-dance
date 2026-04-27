"""Harmonic feature extraction for tension and chord analysis.

This module extracts harmonic features that drive visual dynamics:
- **Tension**: Psychoacoustic roughness (Sethares model) for SLERP blend
- **Tonal distance**: JSD from track-average chroma for prompt SLERP
- **Chroma centroid**: Circular mean of chroma for pitch-based position

Design:
- Pure functions, no state
- Vectorized NumPy — no Python loops in hot paths
- Graceful fallback for non-Western / atonal content (spectral entropy)
"""

from __future__ import annotations

import librosa
import numpy as np

try:  # Optional torch backend
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in tests
    torch = None


def compute_chroma(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    mag: np.ndarray | None = None,
    mode: str = "stft",
) -> np.ndarray:
    """Compute chroma features.

    Args:
        audio: Mono audio signal.
        sr: Sample rate.
        hop_length: Hop length in samples.
        mag: Optional magnitude spectrogram (for STFT chroma).
        mode: "stft" (fast, reuses mag) or "cqt" (slower, higher resolution).
    """
    if mode == "stft" and mag is not None:
        power = mag**2
        return librosa.feature.chroma_stft(S=power, sr=sr, hop_length=hop_length)
    return librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)


def compute_roughness_curve(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    n_peaks: int = 20,
    *,
    mag: np.ndarray | "torch.Tensor" | None = None,
    freqs: np.ndarray | "torch.Tensor" | None = None,
) -> np.ndarray:
    """Compute frame-wise roughness from STFT - FULLY VECTORIZED.

    Uses Sethares/Plomp-Levelt psychoacoustic roughness computed across
    all frames simultaneously. ~50x faster than per-frame loop.

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples (typically sr // fps)
        n_peaks: Number of spectral peaks to consider per frame

    Returns:
        Roughness per frame, shape (n_frames,), values in [0, 1]
    """
    if mag is None:
        # Compute STFT
        stft = librosa.stft(audio, hop_length=hop_length)
        magnitude = np.abs(stft)  # (n_bins, n_frames)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 2)
    else:
        magnitude = mag
        if freqs is None:
            raise ValueError("freqs must be provided when mag is provided")

    if torch is not None and isinstance(magnitude, torch.Tensor):
        return _compute_roughness_curve_torch(magnitude, freqs, n_peaks)

    n_bins, n_frames = magnitude.shape

    # Skip DC and very low frequencies (below 30 Hz)
    valid_mask = freqs > 30
    valid_freqs = freqs[valid_mask]
    valid_mag = magnitude[valid_mask, :]  # (valid_bins, n_frames)

    # Get top N peaks per frame using argpartition (O(n) vs O(n log n) for argsort)
    # Shape: (n_peaks, n_frames)
    peak_indices = np.argpartition(valid_mag, -n_peaks, axis=0)[-n_peaks:, :]

    # Gather peak frequencies: broadcast valid_freqs to (valid_bins, n_frames)
    # then use advanced indexing to select peaks
    peak_freqs = valid_freqs[peak_indices]  # (n_peaks, n_frames)

    # Gather peak amplitudes using take_along_axis
    peak_amps = np.take_along_axis(valid_mag, peak_indices, axis=0)  # (n_peaks, n_frames)

    # Normalize amplitudes per frame
    amp_sum = np.maximum(peak_amps.sum(axis=0, keepdims=True), 1e-10)
    peak_amps = peak_amps / amp_sum

    # Compute all pairwise roughness: (n_peaks, n_peaks, n_frames)
    # Expand dims for broadcasting
    f1 = peak_freqs[:, np.newaxis, :]  # (n_peaks, 1, n_frames)
    f2 = peak_freqs[np.newaxis, :, :]  # (1, n_peaks, n_frames)
    a1 = peak_amps[:, np.newaxis, :]
    a2 = peak_amps[np.newaxis, :, :]

    f_min = np.minimum(f1, f2)
    f_diff = np.abs(f2 - f1)

    # Critical bandwidth scaling (Plomp-Levelt model)
    s = 0.24 / (0.021 * f_min + 19)
    x = f_diff * s

    # Plomp-Levelt dissonance curve
    dissonance = a1 * a2 * (np.exp(-3.5 * x) - np.exp(-5.75 * x))
    dissonance = np.maximum(dissonance, 0)  # Ensure non-negative

    # Sum upper triangle only (avoid double counting and self-pairs)
    # Get indices once, apply to all frames
    i_upper, j_upper = np.triu_indices(n_peaks, k=1)
    roughness = dissonance[i_upper, j_upper, :].sum(axis=0)

    # Normalize to [0, 1] (empirical scaling factor)
    roughness = np.minimum(roughness * 4.0, 1.0)

    return roughness.astype(np.float32)


def _compute_roughness_curve_torch(
    magnitude: "torch.Tensor",
    freqs: "torch.Tensor",
    n_peaks: int,
) -> np.ndarray:
    """Torch backend for roughness curve (GPU-friendly)."""
    # Skip DC and very low frequencies (below 30 Hz)
    valid_mask = freqs > 30
    valid_freqs = freqs[valid_mask]
    valid_mag = magnitude[valid_mask, :]  # (valid_bins, n_frames)

    # Top-N peaks per frame
    peak_amps, peak_indices = torch.topk(valid_mag, k=n_peaks, dim=0)
    peak_freqs = valid_freqs[peak_indices]

    # Normalize amplitudes per frame
    amp_sum = torch.clamp(peak_amps.sum(dim=0, keepdim=True), min=1e-10)
    peak_amps = peak_amps / amp_sum

    # Pairwise roughness
    f1 = peak_freqs[:, None, :]
    f2 = peak_freqs[None, :, :]
    a1 = peak_amps[:, None, :]
    a2 = peak_amps[None, :, :]

    f_min = torch.minimum(f1, f2)
    f_diff = torch.abs(f2 - f1)
    s = 0.24 / (0.021 * f_min + 19.0)
    x = f_diff * s
    dissonance = a1 * a2 * (torch.exp(-3.5 * x) - torch.exp(-5.75 * x))
    dissonance = torch.clamp(dissonance, min=0.0)

    i_upper, j_upper = torch.triu_indices(n_peaks, n_peaks, offset=1, device=magnitude.device)
    roughness = dissonance[i_upper, j_upper, :].sum(dim=0)
    roughness = torch.clamp(roughness * 4.0, max=1.0)
    return roughness.detach().cpu().numpy().astype(np.float32)


def compute_spectral_entropy(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    mag: np.ndarray | "torch.Tensor" | None = None,
) -> np.ndarray:
    """Compute spectral entropy (fallback tension for atonal content).

    Spectral entropy measures the "flatness" of the spectrum:
    - Low entropy = energy concentrated in few bins (pure tones)
    - High entropy = energy spread across many bins (noise-like)

    This provides a tension proxy for non-Western or atonal music where
    chroma-based analysis may not apply.

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples

    Returns:
        Spectral entropy per frame, shape (n_frames,), normalized to [0, 1]
    """
    if mag is None:
        # Compute power spectrum
        stft = librosa.stft(audio, hop_length=hop_length)
        power = np.abs(stft) ** 2
    else:
        power = mag * mag

    if torch is not None and isinstance(power, torch.Tensor):
        return _compute_spectral_entropy_torch(power)

    # Normalize each frame to probability distribution
    frame_sum = np.sum(power, axis=0, keepdims=True)
    frame_sum = np.maximum(frame_sum, 1e-10)  # Avoid division by zero
    p = power / frame_sum

    # Compute entropy: -sum(p * log(p))
    # Use small epsilon to avoid log(0)
    p_safe = np.maximum(p, 1e-10)
    entropy = -np.sum(p * np.log(p_safe), axis=0)

    # Normalize to [0, 1] using max possible entropy
    # Max entropy = log(n_bins) when uniform distribution
    n_bins = power.shape[0]
    max_entropy = np.log(n_bins)
    normalized = entropy / max_entropy

    return normalized.astype(np.float32)


def _compute_spectral_entropy_torch(power: "torch.Tensor") -> np.ndarray:
    frame_sum = torch.clamp(power.sum(dim=0, keepdim=True), min=1e-10)
    p = power / frame_sum
    p_safe = torch.clamp(p, min=1e-10)
    entropy = -(p * torch.log(p_safe)).sum(dim=0)
    n_bins = power.shape[0]
    max_entropy = float(np.log(n_bins))
    normalized = entropy / max_entropy
    return normalized.detach().cpu().numpy().astype(np.float32)


def compute_chroma_centroid(chroma: np.ndarray) -> np.ndarray:
    """Compute circular mean of chroma bins per frame.

    Args:
        chroma: (12, T) chromagram

    Returns:
        (T,) values in [0, 1], where 0 and 1 represent the same pitch class.
        Frames with very low chroma energy return 0.5 (center).
    """
    if chroma is None or chroma.size == 0:
        return np.zeros((0,), dtype=np.float32)

    n_bins, _ = chroma.shape
    if n_bins != 12:
        # Fallback: normalize index by bin count
        bins = np.arange(n_bins, dtype=np.float32)
        weights = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)
        centroid = np.sum(weights * bins[:, None], axis=0) / max(1.0, n_bins - 1)
        return centroid.astype(np.float32)

    angles = 2.0 * np.pi * np.arange(12, dtype=np.float32) / 12.0
    cos_vals = np.cos(angles)[:, None]
    sin_vals = np.sin(angles)[:, None]

    x = np.sum(chroma * cos_vals, axis=0)
    y = np.sum(chroma * sin_vals, axis=0)
    mag = np.hypot(x, y)
    angle = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    centroid = angle / (2.0 * np.pi)
    centroid = np.where(mag < 1e-6, 0.5, centroid)
    return centroid.astype(np.float32)


def compute_tension(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    pitch_confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.3,
    n_peaks: int = 20,
    *,
    mag: np.ndarray | "torch.Tensor" | None = None,
    freqs: np.ndarray | "torch.Tensor" | None = None,
) -> np.ndarray:
    """Compute combined tension with intelligent fallback.

    Uses Sethares roughness when the audio has clear pitch content,
    falls back to spectral entropy for atonal or noise-like content.

    The pitch_confidence array (if provided) allows smooth blending:
    - High confidence → use roughness (tonal music)
    - Low confidence → use entropy (noise, speech, atonal)

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples
        pitch_confidence: Optional per-frame pitch confidence, shape (n_frames,)
        confidence_threshold: Below this, prefer entropy over roughness
        n_peaks: Number of spectral peaks for roughness calculation

    Returns:
        Tension per frame, shape (n_frames,), values in [0, 1]
    """
    # Compute both metrics
    roughness = compute_roughness_curve(
        audio, sr, hop_length, n_peaks, mag=mag, freqs=freqs
    )
    entropy = compute_spectral_entropy(audio, sr, hop_length, mag=mag)

    # Align lengths (librosa can produce ±1 frame difference)
    n_frames = min(len(roughness), len(entropy))
    roughness = roughness[:n_frames]
    entropy = entropy[:n_frames]

    if pitch_confidence is not None:
        # Blend based on pitch confidence
        confidence = pitch_confidence[:n_frames]
        confidence = np.clip(confidence, 0, 1)

        # Smooth transition: use roughness when confident, entropy otherwise
        # Create a sigmoid-like blend around the threshold
        blend = 1.0 / (1.0 + np.exp(-10 * (confidence - confidence_threshold)))
        tension = blend * roughness + (1 - blend) * entropy
    else:
        # No pitch info: estimate from spectral characteristics
        # High entropy usually means low pitch confidence
        # Use entropy as a self-selecting proxy
        # When entropy is high, trust entropy; when low, trust roughness
        entropy_blend = np.clip(entropy, 0, 1)
        tension = (1 - entropy_blend) * roughness + entropy_blend * entropy

    return tension.astype(np.float32)


def compute_tonal_distance(
    chroma: np.ndarray,
    smoothing_ms: float = 200.0,
    fps: float = 60.0,
) -> np.ndarray:
    """Compute per-frame tonal distance from track-average chroma profile.

    Uses Jensen-Shannon divergence between each frame's chroma distribution
    and the track-wide average. This captures harmonic SURPRISE — how much
    the current harmony departs from the song's tonal center.

    Key-agnostic: a modulation from C major to A minor reads as departure
    regardless of absolute key, because the chroma *distribution* changes.

    Chroma centroid failed for this purpose due to:
    - Key-locked stagnation (C major and A minor both center around C/E/G)
    - No normalization (absolute centroid varies by instrument timbre)
    - Circular mean instability near wrap-around

    JSD avoids all of these by comparing full 12-bin distributions.

    Args:
        chroma: (12, T) chromagram, energy per pitch class per frame
        smoothing_ms: Asymmetric smoothing time constant in ms
            (fast attack to catch departures, slow release for smooth return)
        fps: Frame rate for smoothing

    Returns:
        (T,) tonal distance in [0, 1], percentile-normalized.
        0 = home key, 1 = maximum harmonic departure for this track.
    """
    if chroma is None or chroma.shape[1] < 2:
        return np.zeros(max(1, chroma.shape[1] if chroma is not None else 1), dtype=np.float32)

    n_bins, n_frames = chroma.shape

    # Normalize each frame to a probability distribution
    frame_sums = np.maximum(chroma.sum(axis=0, keepdims=True), 1e-10)
    p_frames = chroma / frame_sums  # (12, T)

    # Track-average chroma profile (the "home key" distribution)
    q_avg = chroma.mean(axis=1, keepdims=True)  # (12, 1)
    q_avg = q_avg / np.maximum(q_avg.sum(), 1e-10)

    # Jensen-Shannon divergence: JSD(P || Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    # where M = 0.5 * (P + Q)
    # Vectorized across all frames simultaneously
    m = 0.5 * (p_frames + q_avg)  # (12, T)

    # KL divergence with numerical stability
    eps = 1e-10
    kl_pm = np.sum(p_frames * np.log(np.maximum(p_frames, eps) / np.maximum(m, eps)), axis=0)
    kl_qm = np.sum(q_avg * np.log(np.maximum(q_avg, eps) / np.maximum(m, eps)), axis=0)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm  # (T,)

    # JSD is bounded [0, ln(2)], normalize to [0, 1]
    jsd = np.clip(jsd / np.log(2), 0.0, 1.0)

    # Asymmetric smoothing: fast attack (catch departures), slow release (smooth return)
    attack_ms = smoothing_ms * 0.3  # Fast attack
    release_ms = smoothing_ms * 2.0  # Slow release

    attack_alpha = 1.0 - np.exp(-1.0 / max(attack_ms / 1000.0 * fps, 1.0))
    release_alpha = 1.0 - np.exp(-1.0 / max(release_ms / 1000.0 * fps, 1.0))

    smoothed = np.zeros(n_frames, dtype=np.float64)
    smoothed[0] = jsd[0]
    for i in range(1, n_frames):
        alpha = attack_alpha if jsd[i] > smoothed[i - 1] else release_alpha
        smoothed[i] = smoothed[i - 1] + alpha * (jsd[i] - smoothed[i - 1])

    # Percentile normalization (robust to outlier frames)
    p5, p95 = np.percentile(smoothed, [5, 95])
    if p95 - p5 > 1e-6:
        smoothed = np.clip((smoothed - p5) / (p95 - p5), 0.0, 1.0)
    else:
        smoothed = np.zeros_like(smoothed)

    return smoothed.astype(np.float32)
