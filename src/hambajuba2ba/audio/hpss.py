"""HPSS (Harmonic-Percussive Source Separation) utilities.

HPSS splits audio into harmonic (sustained, tonal) and percussive (transient)
components using median filtering in the spectrogram domain. This enables:
- Texture classification: percussive ratio tells us if a stem is drums vs pads
- Layer detection: normalized flux reveals when instruments enter/exit
- Physics mapping: different components get different visual dynamics
- v2: Time-series H/P energy curves for separate link targets (drums_harmonic, etc.)

Design:
- Small, focused functions (one job each)
- No state — pure functions for offline processing
- Use librosa's battle-tested implementations
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

try:  # Optional torch backend
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


def compute_normalized_flux(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    mag: np.ndarray | None = None,
) -> np.ndarray:
    """Compute amplitude-normalized spectral flux.

    Standard spectral flux conflates loudness with spectral change.
    Normalized flux divides by frame energy, isolating *textural* change:
    - A soft violin entrance registers as high normalized flux
    - A loud sustained chord registers as low normalized flux

    This is critical for layer detection: finding when instruments enter
    regardless of their volume.

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples (typically sr // fps)

    Returns:
        Normalized flux per frame, shape (n_frames,)
    """
    # Compute magnitude spectrogram if not provided
    if mag is None:
        stft = librosa.stft(audio, hop_length=hop_length)
        magnitude = np.abs(stft)
    else:
        magnitude = mag

    # Normalize each frame by its energy (avoid division by zero)
    frame_energy = np.sqrt(np.sum(magnitude**2, axis=0, keepdims=True))
    frame_energy = np.maximum(frame_energy, 1e-10)
    normalized = magnitude / frame_energy

    # Spectral flux: sum of positive differences between frames
    diff = np.diff(normalized, axis=1)
    diff = np.maximum(diff, 0)  # Half-wave rectification (only increases)
    flux = np.sum(diff, axis=0)

    # Prepend zero for frame alignment (flux at t requires t-1)
    flux = np.concatenate([[0], flux])

    # Normalize to [0, 1]
    max_flux = flux.max()
    if max_flux > 1e-10:
        flux = flux / max_flux

    return flux.astype(np.float32)


def compute_energy_db(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    ref_db: float = -40.0,
) -> np.ndarray:
    """Compute frame-wise energy in dB.

    Used for activity gating: frames below a threshold are "silent"
    and shouldn't trigger visual events.

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples
        ref_db: Reference floor in dB (values below this are clamped)

    Returns:
        Energy in dB per frame, shape (n_frames,), range [ref_db, 0]
    """
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    # Avoid log(0)
    rms = np.maximum(rms, 1e-10)

    # Normalize to peak, convert to dB
    rms_peak = rms.max()
    if rms_peak > 1e-10:
        rms = rms / rms_peak

    db = 20 * np.log10(rms)
    db = np.maximum(db, ref_db)  # Floor at ref_db

    return db.astype(np.float32)


@dataclass
class HPSSComponents:
    """Time-series HPSS component data for separate link targets.

    Enables drums_harmonic, drums_percussive, etc. as independent audio sources
    with their own energy curves for SAE steering and SLERP destinations.

    Attributes:
        harmonic_energy: (T,) Normalized energy [0,1] of harmonic component
        percussive_energy: (T,) Normalized energy [0,1] of percussive component
        hpss_ratio: Single value [0,1], 0=harmonic, 1=percussive (for classification)
    """

    harmonic_energy: np.ndarray
    percussive_energy: np.ndarray
    hpss_ratio: float


def compute_hpss_components(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    mag: np.ndarray | "torch.Tensor" | None = None,
    backend: str = "auto",
) -> HPSSComponents:
    """Compute time-series HPSS components for use as separate link targets.

    Unlike compute_hpss_ratio which returns a single value, this returns
    frame-aligned energy curves for both harmonic and percussive components.
    These can be used as independent audio sources for SAE steering.

    Example link targets enabled:
    - drums_harmonic: Cymbal sustain, hi-hat wash
    - drums_percussive: Kick/snare transients
    - other_harmonic: Pads, strings, sustained synths
    - other_percussive: Plucks, stabs, arp attacks

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples (typically sr // fps)

    Returns:
        HPSSComponents with time-series energy curves + ratio
    """
    if mag is None:
        stft = librosa.stft(audio, hop_length=hop_length)
        mag = np.abs(stft)

    use_torch = (
        torch is not None
        and isinstance(mag, torch.Tensor)
        and backend in ("auto", "torch")
    )

    if use_torch:
        harmonic_mag, percussive_mag = _compute_hpss_torch(mag)
        h_rms = torch.sqrt(torch.mean(harmonic_mag**2, dim=0))
        p_rms = torch.sqrt(torch.mean(percussive_mag**2, dim=0))
        h_max = torch.max(h_rms)
        p_max = torch.max(p_rms)
        if h_max > 1e-10:
            h_rms = h_rms / h_max
        if p_max > 1e-10:
            p_rms = p_rms / p_max
        total_h = torch.sqrt(torch.mean(harmonic_mag**2))
        total_p = torch.sqrt(torch.mean(percussive_mag**2))
        total = total_h + total_p
        ratio = float((total_p / total).detach().cpu().item()) if total > 1e-10 else 0.5

        return HPSSComponents(
            harmonic_energy=h_rms.detach().cpu().numpy().astype(np.float32),
            percussive_energy=p_rms.detach().cpu().numpy().astype(np.float32),
            hpss_ratio=ratio,
        )

    # CPU fallback: librosa
    if torch is not None and isinstance(mag, torch.Tensor):
        mag = mag.detach().cpu().numpy()
    harmonic_mag, percussive_mag = librosa.decompose.hpss(mag)

    # Compute RMS energy for each component directly from magnitude
    h_rms = librosa.feature.rms(S=harmonic_mag)[0]
    p_rms = librosa.feature.rms(S=percussive_mag)[0]

    # Normalize each to [0, 1] independently
    h_max = h_rms.max()
    p_max = p_rms.max()

    if h_max > 1e-10:
        h_rms = h_rms / h_max
    if p_max > 1e-10:
        p_rms = p_rms / p_max

    # Compute overall ratio (for classification compatibility)
    total_h = np.sqrt(np.mean(harmonic_mag**2))
    total_p = np.sqrt(np.mean(percussive_mag**2))
    total = total_h + total_p
    ratio = float(total_p / total) if total > 1e-10 else 0.5

    return HPSSComponents(
        harmonic_energy=h_rms.astype(np.float32),
        percussive_energy=p_rms.astype(np.float32),
        hpss_ratio=ratio,
    )


def _median_filter_torch(
    mag: "torch.Tensor",
    *,
    kernel_size: int,
    axis: int,
) -> "torch.Tensor":
    """Median filter along a single axis for 2D (freq x time) tensors."""
    if F is None:
        raise RuntimeError("torch.nn.functional is required for torch HPSS")
    if axis == 0:
        # Frequency axis
        x = mag.unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
        pad = (0, 0, kernel_size // 2, kernel_size // 2)
        x = F.pad(x, pad, mode="reflect")
        unfolded = x.unfold(2, kernel_size, 1)  # (1,1,F,T,k)
        return unfolded.median(dim=-1).values.squeeze(0).squeeze(0)
    if axis == 1:
        # Time axis
        x = mag.unsqueeze(0).unsqueeze(0)
        pad = (kernel_size // 2, kernel_size // 2, 0, 0)
        x = F.pad(x, pad, mode="reflect")
        unfolded = x.unfold(3, kernel_size, 1)  # (1,1,F,T,k)
        return unfolded.median(dim=-1).values.squeeze(0).squeeze(0)
    raise ValueError("axis must be 0 (freq) or 1 (time)")


def _softmask_torch(
    numerator: "torch.Tensor",
    denominator: "torch.Tensor",
    *,
    power: float = 2.0,
    eps: float = 1e-10,
) -> "torch.Tensor":
    denom = torch.clamp(denominator, min=eps)
    return (numerator**power) / (numerator**power + denom**power)


def _compute_hpss_torch(
    mag: "torch.Tensor",
    *,
    kernel_size: int | tuple[int, int] = (31, 31),
    power: float = 2.0,
    margin: float = 1.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch HPSS approximation using median filtering on magnitude spectrogram."""
    if isinstance(kernel_size, int):
        k_freq = k_time = kernel_size
    else:
        k_freq, k_time = kernel_size

    # Median filters on magnitude**power
    S = mag**power
    harmonic_med = _median_filter_torch(S, kernel_size=k_time, axis=1)
    percussive_med = _median_filter_torch(S, kernel_size=k_freq, axis=0)

    # Soft masks
    harm_mask = _softmask_torch(harmonic_med, percussive_med * margin, power=1.0)
    perc_mask = _softmask_torch(percussive_med, harmonic_med * margin, power=1.0)

    harmonic_mag = harm_mask * mag
    percussive_mag = perc_mask * mag

    return harmonic_mag, percussive_mag
