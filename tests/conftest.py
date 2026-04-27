"""Pytest configuration and shared fixtures.

This file sets up the Python path for imports and provides
common fixtures for testing the audio, SAE, and config modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root and src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


# === Audio Fixtures ===


@pytest.fixture
def sample_audio_mono():
    """1 second of deterministic test audio at 44.1kHz.

    Creates a 440Hz sine wave (A4 note) with a 880Hz harmonic.
    This provides predictable spectral content for testing.
    """
    sr = 44100
    t = np.linspace(0, 1, sr, dtype=np.float32)
    # 440Hz sine + 880Hz harmonic (2nd harmonic)
    return np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)


@pytest.fixture
def sample_audio_short():
    """Short 0.1 second audio for quick tests."""
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sample_stems(sample_audio_mono):
    """Mock stem dictionary with 4 stems.

    Simulates output from StemSeparator with different amplitudes
    for each stem (bass loudest, other quietest).
    """
    return {
        "bass": (sample_audio_mono * 0.8).astype(np.float32),
        "drums": (sample_audio_mono * 0.6).astype(np.float32),
        "vocals": (sample_audio_mono * 0.4).astype(np.float32),
        "other": (sample_audio_mono * 0.2).astype(np.float32),
    }


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio_mono):
    """Write sample audio to a temporary WAV file.

    Uses soundfile to create a valid WAV file for testing
    the StemSeparator's file loading.
    """
    import soundfile as sf

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), sample_audio_mono, 44100)
    return str(audio_path)


# === SAE Fixtures ===


@pytest.fixture
def mock_sae():
    """Small SAE for testing (no GPU needed).

    Creates a minimal SAE with:
    - 64 sparse features
    - 32-dimensional input/output
    - Top-4 sparsity
    """
    from hambajuba2ba.generation.sae import SparseAutoencoder

    return SparseAutoencoder(n_dirs_local=64, d_model=32, k=4)


@pytest.fixture
def mock_sae_weights(tmp_path, mock_sae):
    """Temporary SAE weights on disk for SteeringManager tests.

    Creates directory structure matching what SteeringManager expects:
    {tmp_path}/down.2.1/final/config.json
    {tmp_path}/down.2.1/final/state_dict.pth
    """
    block_dir = tmp_path / "down.2.1" / "final"
    block_dir.mkdir(parents=True)
    mock_sae.save_to_disk(str(block_dir))
    return tmp_path


# === Config Fixtures ===


@pytest.fixture
def cuda_config():
    """Config for CUDA device."""
    from hambajuba2ba.config import PipelineConfig

    return PipelineConfig(device="cuda", dtype="float16")


@pytest.fixture
def mps_config():
    """Config for MPS device (Apple Silicon).

    Should automatically coerce float16 to float32 for MPS stability.
    """
    from hambajuba2ba.config import PipelineConfig

    return PipelineConfig(device="mps", dtype="float16")


@pytest.fixture
def cpu_config():
    """Config for CPU device."""
    from hambajuba2ba.config import PipelineConfig

    return PipelineConfig(device="cpu", dtype="float32")
