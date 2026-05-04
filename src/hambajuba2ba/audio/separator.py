"""Audio stem separation using python-audio-separator.

Wraps the audio-separator library for separating audio tracks into
component stems (bass, drums, vocals, other) using Demucs models.

Install: uv sync --extra audio (CPU) or --extra audio-gpu (CUDA)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class StemSeparator:
    """Wraps audio-separator for 4-stem audio separation.

    Separates audio into: bass, drums, vocals, other.
    Uses htdemucs_ft model by default for best quality.

    The audio-separator library handles:
    - Model download and caching
    - Device selection (CPU/GPU)
    - Memory-efficient processing

    Attributes:
        model_name: Demucs model variant (htdemucs_ft.yaml recommended)
        device: Torch device for separation ("cuda", "cpu", "mps")
    """

    STEM_NAMES = ["bass", "drums", "vocals", "other"]

    # Map audio-separator output names to our standard names
    STEM_MAP = {
        "Bass": "bass",
        "Drums": "drums",
        "Vocals": "vocals",
        "Other": "other",
    }

    def __init__(
        self,
        model_name: str = "htdemucs_ft.yaml",
        device: str = "cuda",
        output_dir: str | None = None,
    ):
        """Initialize separator.

        Args:
            model_name: Demucs model file (htdemucs_ft.yaml for best quality)
            device: Torch device ("cuda", "cpu", "mps")
            output_dir: Directory for stem output files (default: .cache/stems)
        """
        self.model_name = model_name
        # MPS isn't well-supported by audio-separator, fall back to CPU
        self.device = "cpu" if device == "mps" else device
        self.output_dir = output_dir or ".cache/stems"
        self._separator = None

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> bool:
        """Lazy-load the separator. Returns True if real separator available."""
        if self._separator is None:
            try:
                from audio_separator.separator import Separator

                logger.info(f"Loading audio-separator with model: {self.model_name}")
                logger.info(f"Stem output directory: {self.output_dir}")
                # AUDIO_SEPARATOR_MODEL_DIR controls where htdemucs_ft weights
                # are cached. Without this, defaults to /tmp/ which is ephemeral
                # on fresh GPU hosts and causes a 320MB re-download every cold boot.
                model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
                self._separator = Separator(
                    output_dir=self.output_dir,
                    output_format="wav",  # WAV for lossless
                    output_single_stem=None,  # Get all stems
                    **({"model_file_dir": model_dir} if model_dir else {}),
                )
                self._separator.load_model(model_filename=self.model_name)
                logger.info("Audio separator loaded")
                return True
            except ImportError:
                logger.warning(
                    "audio-separator not installed, using mock separation. "
                    "Install with: uv sync --extra audio"
                )
                self._separator = "mock"
                return False
        return self._separator != "mock"

    async def separate(
        self,
        audio_path: str,
        sample_rate: int = 44100,
    ) -> Dict[str, np.ndarray]:
        """Separate audio file into stems.

        Runs in executor to avoid blocking event loop.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate for output

        Returns:
            {"bass": array, "drums": array, "vocals": array, "other": array}
            Each array is mono (samples,) at the given sample rate.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.separate_sync,
            audio_path,
            sample_rate,
        )

    def separate_sync(
        self,
        audio_path: str,
        sample_rate: int = 44100,
    ) -> Dict[str, np.ndarray]:
        """Synchronous version of separate().

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate

        Returns:
            Dictionary of stem name to mono audio array
        """
        has_separator = self._ensure_loaded()

        if not has_separator:
            return self._mock_separate(audio_path, sample_rate)

        import librosa

        logger.info(f"Separating audio: {audio_path}")

        # Run separation - output goes to configured output_dir
        # Files are kept for caching (not using temp dir)
        output_files = self._separator.separate(audio_path)
        logger.info(f"Separation produced {len(output_files)} files in {self.output_dir}")
        if not output_files:
            raise RuntimeError(
                "Stem separation failed: audio-separator produced no output files"
            )

        # Load each stem file as numpy array
        stems = {}
        for filepath in output_files:
            # audio_separator may return just filenames, ensure we have full path
            filepath = Path(filepath)
            if not filepath.is_absolute():
                filepath = Path(self.output_dir) / filepath

            filename = filepath.stem

            # Match stem name from filename (e.g., "song_(Drums).wav")
            stem_name = None
            for sep_name, our_name in self.STEM_MAP.items():
                if sep_name in filename:
                    stem_name = our_name
                    break

            if stem_name is None:
                logger.debug(f"Skipping unknown stem: {filename}")
                continue

            # Load and convert to mono
            audio, sr = librosa.load(str(filepath), sr=sample_rate, mono=True)
            stems[stem_name] = audio.astype(np.float32)
            logger.debug(f"Loaded {stem_name}: {len(audio)} samples")

        missing = sorted(set(self.STEM_NAMES) - set(stems))
        if missing:
            raise RuntimeError(
                "Stem separation failed: missing required stems "
                + ", ".join(missing)
            )

        logger.info(f"Separation complete: {list(stems.keys())}")
        return stems

    def _mock_separate(
        self,
        audio_path: str,
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        """Mock separation for testing without audio-separator.

        Uses simple frequency filtering as a fallback.
        Good enough for development/testing UI flow.
        """
        import librosa

        logger.warning("Using mock separation (audio-separator not available)")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

        # Create "fake" stems by frequency filtering
        stems = {}
        stems["bass"] = self._bandpass_filter(audio, sr, 20, 200)
        stems["drums"] = self._bandpass_filter(audio, sr, 100, 4000)
        stems["vocals"] = self._bandpass_filter(audio, sr, 300, 3000)
        stems["other"] = self._bandpass_filter(audio, sr, 2000, 16000)

        return stems

    def _bandpass_filter(
        self,
        audio: np.ndarray,
        sr: int,
        low: float,
        high: float,
    ) -> np.ndarray:
        """Simple bandpass filter for mock separation.

        Uses second-order sections (sos) for numerical stability.
        Falls back to scaled original audio if filter produces NaN/Inf.
        """
        from scipy import signal

        nyq = sr / 2

        # Normalize to [0, 1] range (fraction of Nyquist)
        low_norm = low / nyq
        high_norm = high / nyq

        # Clamp to safe range (0.001 to 0.999)
        low_norm = max(low_norm, 0.001)
        high_norm = min(high_norm, 0.999)

        # Ensure low < high (edge case when low freq gets clamped above high)
        if low_norm >= high_norm:
            # Just return scaled audio for invalid range
            return (audio * 0.25).astype(np.float32)

        try:
            # Use sos format for better numerical stability
            sos = signal.butter(4, [low_norm, high_norm], btype="band", output="sos")
            filtered = signal.sosfiltfilt(sos, audio)

            # Validate output - check for NaN/Inf
            if not np.isfinite(filtered).all():
                logger.warning(f"Filter produced NaN/Inf for {low}-{high}Hz, using fallback")
                return (audio * 0.25).astype(np.float32)

            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"Filter failed for {low}-{high}Hz: {e}")
            return (audio * 0.25).astype(np.float32)


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    import librosa

    duration = librosa.get_duration(path=audio_path)
    return duration
