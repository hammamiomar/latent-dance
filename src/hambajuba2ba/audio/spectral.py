"""Shared spectrogram utilities for feature extraction.

Computes a single STFT magnitude per stem and reuses it across
HPSS, tension, flux normalization, centroid, and flatness to
avoid redundant FFT work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np

try:  # Optional torch backend
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in tests
    torch = None


SpectralBackend = Literal["librosa", "torch", "auto"]
SpectralDevice = Literal["auto", "cuda", "mps", "cpu"]


@dataclass
class Spectrogram:
    mag: np.ndarray | "torch.Tensor"
    freqs: np.ndarray | "torch.Tensor"
    backend: str
    device: str


def _resolve_backend(backend: SpectralBackend, device: SpectralDevice) -> tuple[str, str]:
    if backend == "auto":
        # Explicit GPU request wins; "auto" only self-selects CUDA —
        # MPS DSP is opt-in until validated on real songs.
        if torch is not None and device in ("cuda", "mps"):
            return "torch", device
        if torch is not None and device == "auto" and torch.cuda.is_available():
            return "torch", "cuda"
        return "librosa", "cpu"
    if backend == "torch":
        if torch is None:
            return "librosa", "cpu"
        if device == "auto":
            return "torch", "cuda" if torch.cuda.is_available() else "cpu"
        return "torch", device
    return "librosa", "cpu"


def compute_spectrogram(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    n_fft: int = 2048,
    backend: SpectralBackend = "auto",
    device: SpectralDevice = "auto",
) -> Spectrogram:
    """Compute a shared magnitude spectrogram for downstream features."""
    resolved_backend, resolved_device = _resolve_backend(backend, device)
    if resolved_backend == "torch":
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is unavailable.")
        audio_t = torch.as_tensor(audio, dtype=torch.float32, device=resolved_device)
        window = torch.hann_window(n_fft, device=resolved_device)
        stft = torch.stft(
            audio_t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = stft.abs()
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(mag.device)
        return Spectrogram(
            mag=mag,
            freqs=freqs,
            backend="torch",
            device=str(resolved_device),
        )

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return Spectrogram(
        mag=mag,
        freqs=freqs,
        backend="librosa",
        device="cpu",
    )
