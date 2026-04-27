"""Tests for StemSeparator and bandpass filter.

These tests validate the audio separation pipeline and specifically
catch the NaN bug in bandpass filtering that was causing librosa errors.
"""

import numpy as np

from hambajuba2ba.audio import StemSeparator


class TestStemSeparator:
    """Tests for StemSeparator class."""

    def test_init_default_device(self):
        """Separator should initialize with default device."""
        sep = StemSeparator()
        assert sep.device in ["cuda", "cpu"]

    def test_init_cpu_device(self):
        """Separator should accept cpu device."""
        sep = StemSeparator(device="cpu")
        assert sep.device == "cpu"

    def test_init_mps_falls_back_to_cpu(self):
        """MPS should fall back to CPU for audio-separator.

        audio-separator doesn't support MPS, so we fall back to CPU
        for reliable stem separation on Apple Silicon.
        """
        sep = StemSeparator(device="mps")
        assert sep.device == "cpu"  # MPS not supported by audio-separator

    def test_mock_separator_returns_all_stems(self, temp_audio_file):
        """Mock separator should return all 4 stems."""
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(temp_audio_file)

        assert set(stems.keys()) == {"bass", "drums", "vocals", "other"}

    def test_stems_are_finite(self, temp_audio_file):
        """All stem arrays must be finite (no NaN/Inf).

        This test catches the bug where bandpass filtering produced NaN
        values that later crashed librosa with:
        "Audio buffer is not finite everywhere"
        """
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(temp_audio_file)

        for name, data in stems.items():
            assert np.isfinite(data).all(), f"{name} contains NaN/Inf"

    def test_stems_are_float32(self, temp_audio_file):
        """Stems must be float32 for librosa compatibility."""
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(temp_audio_file)

        for name, data in stems.items():
            assert data.dtype == np.float32, f"{name} is {data.dtype}"

    def test_stems_have_same_length(self, temp_audio_file):
        """All stems should have the same length as each other."""
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(temp_audio_file)

        lengths = [len(data) for data in stems.values()]
        assert len(set(lengths)) == 1, "Stems have different lengths"

    def test_stems_not_empty(self, temp_audio_file):
        """Stems should not be empty arrays."""
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(temp_audio_file)

        for name, data in stems.items():
            assert len(data) > 0, f"{name} is empty"

    def test_custom_sample_rate(self, temp_audio_file):
        """Separator should resample to target rate."""
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(temp_audio_file, sample_rate=22050)

        # All stems should have data (resampled)
        for data in stems.values():
            assert len(data) > 0


class TestBandpassFilter:
    """Tests for the bandpass filter used in mock separation.

    These tests specifically target the edge cases that caused NaN
    values before the scipy sosfiltfilt fix.
    """

    def test_low_freq_edge_case(self):
        """Filter should handle 20Hz (near Nyquist edge).

        20Hz with 44100Hz sample rate creates low_norm = 20/22050 ≈ 0.0009
        which is very close to the stability boundary.
        """
        sep = StemSeparator(device="cpu")
        audio = np.random.randn(44100).astype(np.float32)

        # This was causing NaN before the fix
        result = sep._bandpass_filter(audio, 44100, 20, 200)

        assert np.isfinite(result).all()
        assert result.dtype == np.float32

    def test_high_freq_edge_case(self):
        """Filter should handle high frequencies near Nyquist."""
        sep = StemSeparator(device="cpu")
        audio = np.random.randn(44100).astype(np.float32)

        result = sep._bandpass_filter(audio, 44100, 10000, 20000)

        assert np.isfinite(result).all()
        assert result.dtype == np.float32

    def test_invalid_frequency_range(self):
        """Filter should handle low_norm >= high_norm gracefully.

        When low > high (invalid range), the filter should return
        a scaled fallback rather than crashing or producing NaN.
        """
        sep = StemSeparator(device="cpu")
        audio = np.random.randn(44100).astype(np.float32)

        # Invalid range: low > high
        result = sep._bandpass_filter(audio, 44100, 500, 100)

        assert np.isfinite(result).all()  # Should return scaled fallback

    def test_filter_preserves_length(self):
        """Filter should preserve audio length."""
        sep = StemSeparator(device="cpu")
        audio = np.random.randn(44100).astype(np.float32)

        result = sep._bandpass_filter(audio, 44100, 100, 4000)

        assert len(result) == len(audio)

    def test_filter_output_is_different(self):
        """Filter should actually modify the audio."""
        sep = StemSeparator(device="cpu")
        audio = np.random.randn(44100).astype(np.float32)

        result = sep._bandpass_filter(audio, 44100, 100, 4000)

        # Should not be identical (unless edge case returns fallback)
        assert not np.allclose(result, audio) or np.allclose(result, audio * 0.25)

    def test_filter_reduces_out_of_band(self):
        """Filter should reduce energy outside passband."""
        sep = StemSeparator(device="cpu")

        # Pure 440Hz tone
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        # Filter to high frequencies only (should reduce 440Hz)
        result = sep._bandpass_filter(audio, 44100, 2000, 10000)

        # Energy should be reduced
        assert np.abs(result).mean() < np.abs(audio).mean()
