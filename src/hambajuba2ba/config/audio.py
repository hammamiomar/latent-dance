"""Audio processing configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _default_song_library_dir() -> str:
    """Use workspace persistent storage when available, local cache otherwise."""
    if Path("/workspace").exists():
        return "/workspace/.cache/hambajuba2ba/songs"
    return ".cache/songs"


@dataclass
class AudioConfig:
    """Audio processing settings."""

    sample_rate: int = 44100
    max_upload_mb: int = 500
    allowed_extensions: Tuple[str, ...] = (
        "mp3",
        "wav",
        "flac",
        "ogg",
        "m4a",
        "aac",
        "wma",
    )
    cache_ttl_seconds: int = 3600
    default_bpm: float = 120.0

    # Feature extraction level ("core" = faster, "full" = all features)
    feature_level: str = "full"

    # Feature backend selection ("auto" prefers torch+CUDA when available)
    feature_backend: str = "auto"

    # Feature device selection ("auto" prefers CUDA when available)
    feature_device: str = "auto"

    # Chroma mode ("stft" reuses shared spectrogram, "cqt" is slower but higher-res)
    chroma_mode: str = "stft"

    # HPSS backend ("auto" prefers torch when available)
    hpss_backend: str = "auto"

    # Cross-stem coupling scope ("physical" or "all")
    coupling_stems: str = "physical"

    # Enable on-disk feature cache
    enable_feature_cache: bool = True

    # Cache directories (relative to workspace or absolute)
    stems_cache_dir: str = ".cache/stems"
    uploads_dir: str = ".cache/uploads"
    song_library_dir: str = field(default_factory=_default_song_library_dir)

    def ensure_cache_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        Path(self.stems_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.uploads_dir).mkdir(parents=True, exist_ok=True)
        Path(self.song_library_dir).mkdir(parents=True, exist_ok=True)
