"""Beat and onset detection using hybrid approach.

- Beat This! (ISMIR 2024) for beat tracking - state-of-the-art accuracy
- madmom RNN for onset detection - best for detecting note onsets

Both with graceful librosa fallbacks if dependencies unavailable.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


def detect_beats(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """Detect beats using Beat This! with librosa fallback.

    Args:
        audio: Mono audio signal
        sr: Sample rate

    Returns:
        (beat_times, bpm) - times in seconds
    """
    try:
        beats, bpm = _detect_beats_beat_this(audio, sr)
        if bpm < 40.0 or bpm > 220.0:
            warnings.warn(
                f"Beat This! tempo out of range ({bpm:.1f} BPM), "
                "using librosa fallback"
            )
            return _detect_beats_librosa(audio, sr)
        return beats, bpm
    except ImportError:
        warnings.warn("beat_this unavailable, using librosa (less accurate)")
        return _detect_beats_librosa(audio, sr)
    except Exception as e:
        warnings.warn(f"beat_this failed ({e}), using librosa fallback")
        return _detect_beats_librosa(audio, sr)


def _detect_beats_beat_this(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """Beat This! - ISMIR 2024, state-of-the-art beat tracking."""
    from beat_this.inference import File2Beats
    import tempfile
    import soundfile as sf
    import os

    # Beat This! needs a file path, write temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        sf.write(temp_path, audio, sr)

        # Use GPU if available, CPU fallback
        try:
            file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
        except Exception:
            file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)

        beats, downbeats = file2beats(temp_path)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    # BPM from beat intervals
    beats = np.array(beats)
    if len(beats) > 1:
        intervals = np.diff(beats)
        bpm = 60.0 / np.median(intervals)
    else:
        bpm = 120.0

    return beats, bpm


def _detect_beats_librosa(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """Librosa fallback for beat detection."""
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr)

    # Handle both old (float) and new (array) librosa tempo format
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)

    return beats, tempo
