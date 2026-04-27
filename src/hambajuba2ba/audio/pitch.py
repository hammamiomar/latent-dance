"""Pitch tracking for melodic feature extraction.

This module provides pitch tracking for vocals, bass, and other melodic stems.
Two backends are supported:
- **PESTO**: Sony CSL monophonic tracker (12x faster than CREPE, accurate for vocals/bass)
- **Basic Pitch**: Spotify polyphonic tracker (for piano, synths, chords)

Design:
- Lazy imports for heavy dependencies (pesto, basic_pitch)
- Vectorized NumPy — no Python loops in hot paths
- Frame alignment via resample_to_fps() for consistent timing
- Cents-based intervals: cents = 1200 * log2(hz / ref_hz)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

from .util import align_1d

logger = logging.getLogger(__name__)

_BASIC_PITCH_MODEL_PATH: str | None = None
_BASIC_PITCH_BACKEND: str | None = None
_BASIC_PITCH_LOGGED_LOAD = False
_BASIC_PITCH_LOGGED_INFER = False


def extract_pitch_pesto(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch using PESTO (monophonic).

    PESTO is a fast, accurate pitch tracker optimized for monophonic sources
    like vocals and bass. It's 12x faster than CREPE with similar accuracy.

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples (typically sr // fps)

    Returns:
        pitch_hz: (T,) array of pitch in Hz, 0 for unvoiced frames
        confidence: (T,) array of voicing confidence in [0, 1]
    """
    import torch
    import pesto

    # PESTO expects a PyTorch tensor
    audio_tensor = torch.from_numpy(audio.astype(np.float32))

    # PESTO step size is in ms
    step_size_ms = hop_length / sr * 1000

    # Predict pitch
    timesteps, pitch_hz, confidence, _ = pesto.predict(
        audio_tensor,
        sr,
        step_size=step_size_ms,
    )

    # Convert to numpy if torch tensors
    if hasattr(pitch_hz, 'numpy'):
        pitch_hz = pitch_hz.numpy()
        confidence = confidence.numpy()
        timesteps = timesteps.numpy()

    # Ensure 1D
    pitch_hz = np.atleast_1d(pitch_hz).flatten().astype(np.float32)
    confidence = np.atleast_1d(confidence).flatten().astype(np.float32)
    timesteps = np.atleast_1d(timesteps).flatten()

    # Calculate target frame count
    duration = len(audio) / sr
    target_fps = sr / hop_length
    n_frames = int(np.ceil(duration * target_fps))

    # Resample to target FPS grid
    pitch_hz = resample_to_fps(pitch_hz, timesteps, target_fps, duration)
    confidence = resample_to_fps(confidence, timesteps, target_fps, duration)

    # Align to exact frame count
    pitch_hz = _align(pitch_hz, n_frames)
    confidence = _align(confidence, n_frames)

    # Zero out pitch for low-confidence frames (unvoiced)
    pitch_hz = np.where(confidence > 0.3, pitch_hz, 0.0)

    return pitch_hz, confidence


def _get_basic_pitch_model_path():
    """Get the best available Basic Pitch model path (ONNX preferred)."""
    global _BASIC_PITCH_MODEL_PATH, _BASIC_PITCH_BACKEND, _BASIC_PITCH_LOGGED_LOAD

    if _BASIC_PITCH_MODEL_PATH is not None:
        return _BASIC_PITCH_MODEL_PATH

    from basic_pitch import ONNX_PRESENT, build_icassp_2022_model_path, FilenameSuffix

    if ONNX_PRESENT:
        # ONNX is fastest - use it
        _BASIC_PITCH_BACKEND = "onnx"
        _BASIC_PITCH_MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.onnx)
    else:
        # Fall back to default (CoreML on macOS, TF elsewhere)
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _BASIC_PITCH_BACKEND = "default"
        _BASIC_PITCH_MODEL_PATH = ICASSP_2022_MODEL_PATH
        logger.warning(
            "onnxruntime not available, Basic Pitch will use slower backend. "
            "Install with: pip install onnxruntime"
        )

    if not _BASIC_PITCH_LOGGED_LOAD:
        if _BASIC_PITCH_BACKEND == "onnx":
            providers = None
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
            except Exception:
                providers = None
            logger.info(
                "Basic Pitch ONNX model selected: %s%s",
                _BASIC_PITCH_MODEL_PATH,
                f" (providers={providers})" if providers else "",
            )
        else:
            logger.info("Basic Pitch model selected (non-ONNX): %s", _BASIC_PITCH_MODEL_PATH)
        _BASIC_PITCH_LOGGED_LOAD = True

    return _BASIC_PITCH_MODEL_PATH


def extract_pitch_polyphonic(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pitch using Basic Pitch (polyphonic).

    Basic Pitch is Spotify's polyphonic pitch tracker, suitable for
    piano, synths, and other instruments playing chords. Uses ONNX
    backend when available for ~3x faster inference.

    Args:
        audio: Mono audio signal, shape (n_samples,)
        sr: Sample rate in Hz
        hop_length: Hop length in samples

    Returns:
        pitch_hz: (T,) dominant pitch per frame in Hz, 0 for silence
        confidence: (T,) voicing confidence in [0, 1]
        polyphony_count: (T,) number of simultaneous notes per frame
    """
    import tempfile
    import soundfile as sf
    from basic_pitch.inference import predict as bp_predict

    # Calculate target frames
    duration = len(audio) / sr
    target_fps = sr / hop_length
    n_frames = int(np.ceil(duration * target_fps))

    # Get best available model (ONNX preferred)
    model_path = _get_basic_pitch_model_path()

    # Basic Pitch requires a file path, write to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
        sf.write(tmp.name, audio.astype(np.float32), sr)
        model_output, midi_data, note_events = bp_predict(
            tmp.name,
            model_or_model_path=model_path,
        )

    # Log once on first inference to confirm backend
    global _BASIC_PITCH_LOGGED_INFER
    if not _BASIC_PITCH_LOGGED_INFER:
        backend = _BASIC_PITCH_BACKEND or ("onnx" if str(model_path).endswith(".onnx") else "default")
        logger.info("Basic Pitch inference completed using %s backend (model=%s)", backend, model_path)
        _BASIC_PITCH_LOGGED_INFER = True

    # model_output contains:
    # - 'note': frame-wise note activations (T, 88) for 88 piano keys
    # - 'onset': onset probabilities (T, 88)
    # - 'contour': pitch contour (T, 264)

    frame_activations = model_output['note']  # (T, 88) for 88 piano keys

    # Get Basic Pitch's native frame rate
    bp_hop = 256  # Basic Pitch uses 256 samples at 22050 Hz
    bp_sr = 22050
    bp_fps = bp_sr / bp_hop
    bp_n_frames = frame_activations.shape[0]
    bp_timestamps = np.arange(bp_n_frames) / bp_fps

    # For each frame, find dominant pitch and count active notes
    # Piano key to Hz: A0 = 27.5 Hz, each key is a semitone apart
    midi_numbers = np.arange(21, 109)  # MIDI notes 21-108 (A0 to C8)
    key_freqs = 440.0 * (2.0 ** ((midi_numbers - 69) / 12.0))

    # Threshold for note detection
    threshold = 0.3

    # Vectorized: find max activation and count per frame
    max_activation = np.max(frame_activations, axis=1)
    max_idx = np.argmax(frame_activations, axis=1)
    active_count = np.sum(frame_activations > threshold, axis=1)

    # Dominant pitch (Hz) for frames with sufficient activation
    dominant_hz = np.where(max_activation > threshold, key_freqs[max_idx], 0.0)
    confidence = np.clip(max_activation, 0, 1)
    polyphony = active_count.astype(np.float32)

    # Resample to target FPS
    dominant_hz = resample_to_fps(
        dominant_hz.astype(np.float32), bp_timestamps, target_fps, duration
    )
    confidence = resample_to_fps(
        confidence.astype(np.float32), bp_timestamps, target_fps, duration
    )
    polyphony = resample_to_fps(polyphony, bp_timestamps, target_fps, duration)

    # Align to exact frame count
    dominant_hz = _align(dominant_hz, n_frames)
    confidence = _align(confidence, n_frames)
    polyphony = _align(polyphony, n_frames)

    return dominant_hz, confidence, polyphony


def resample_to_fps(
    data: np.ndarray,
    timestamps: np.ndarray,
    target_fps: float,
    duration: float,
) -> np.ndarray:
    """Resample irregular timestamps to fixed FPS grid.

    Uses linear interpolation to align data from variable-rate pitch
    trackers to a fixed frame rate for consistent indexing.

    Args:
        data: Values to resample, shape (T_original,)
        timestamps: Time points in seconds, shape (T_original,)
        target_fps: Target frame rate in Hz
        duration: Total duration in seconds

    Returns:
        Resampled data at fixed FPS, shape (T_target,)
    """
    if len(data) == 0 or len(timestamps) == 0:
        n_frames = int(np.ceil(duration * target_fps))
        return np.zeros(n_frames, dtype=np.float32)

    # Generate target time grid
    n_frames = int(np.ceil(duration * target_fps))
    target_times = np.arange(n_frames) / target_fps

    # Handle edge case: single frame
    if len(timestamps) == 1:
        return np.full(n_frames, data[0], dtype=np.float32)

    # Interpolate to target grid
    # Use 'linear' with bounds_error=False and fill_value for extrapolation
    interpolator = interp1d(
        timestamps,
        data,
        kind='linear',
        bounds_error=False,
        fill_value=(data[0], data[-1]),  # Extrapolate with edge values
    )

    resampled = interpolator(target_times).astype(np.float32)
    return resampled


def normalize_pitch(
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Normalize pitch to [0,1] using 5-95 percentile range.

    This per-stem normalization maps the vocal/bass range to a consistent
    [0, 1] scale suitable for spatial Y-axis mapping.

    Args:
        pitch_hz: Raw pitch in Hz, shape (T,), 0 for unvoiced
        confidence: Voicing confidence, shape (T,)
        confidence_threshold: Only consider frames above this confidence

    Returns:
        Normalized pitch in [0, 1], shape (T,), 0 for unvoiced frames
    """
    # Get voiced frames only
    voiced_mask = (pitch_hz > 0) & (confidence >= confidence_threshold)
    voiced_pitches = pitch_hz[voiced_mask]

    if len(voiced_pitches) < 10:
        # Not enough pitched content — return zeros
        return np.zeros_like(pitch_hz, dtype=np.float32)

    # Use 5-95 percentile for robust range estimation
    p5, p95 = np.percentile(voiced_pitches, [5, 95])

    if p95 - p5 < 10:  # Less than 10 Hz range
        # Very narrow range (sustained note) — use fixed scale
        p5 = voiced_pitches.min() - 20
        p95 = voiced_pitches.max() + 20

    # Normalize voiced frames to [0, 1]
    normalized = np.zeros_like(pitch_hz, dtype=np.float32)
    normalized[voiced_mask] = np.clip(
        (pitch_hz[voiced_mask] - p5) / (p95 - p5),
        0.0,
        1.0,
    )

    return normalized



# Backwards-compatible alias: delegate to shared util
_align = align_1d
