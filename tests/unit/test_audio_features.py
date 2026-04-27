"""Tests for StemAnalyzer and StemFeatures.

These tests validate audio feature extraction, ensuring envelope,
brightness, and onset detection produce valid normalized values.
"""

import numpy as np
import pytest

from hambajuba2ba.audio import StemAnalyzer, StemFeatures, extract_all_features


class TestStemAnalyzer:
    """Tests for StemAnalyzer class."""

    def test_init(self, sample_audio_mono):
        """Analyzer should initialize with audio data."""
        analyzer = StemAnalyzer(sample_audio_mono, sr=44100, fps=30)
        assert analyzer.sr == 44100
        assert analyzer.fps == 30

    def test_envelope_is_normalized(self, sample_audio_mono):
        """Envelope should be in [0, 1] range."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        assert features.envelope.min() >= 0.0
        assert features.envelope.max() <= 1.0

    def test_envelope_not_empty(self, sample_audio_mono):
        """Envelope should not be empty."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        assert len(features.envelope) > 0

    def test_brightness_is_normalized(self, sample_audio_mono):
        """Brightness (spectral centroid) should be in [0, 1] range."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        assert features.brightness.min() >= 0.0
        assert features.brightness.max() <= 1.0

    def test_brightness_not_empty(self, sample_audio_mono):
        """Brightness should not be empty."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        assert len(features.brightness) > 0

    def test_timestamps_match_envelope_length(self, sample_audio_mono):
        """Timestamps should align with envelope frames."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        assert len(features.timestamps) == len(features.envelope)

    def test_timestamps_match_brightness_length(self, sample_audio_mono):
        """Timestamps should align with brightness frames."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        assert len(features.timestamps) == len(features.brightness)

    def test_duration_matches_audio(self, sample_audio_mono):
        """Duration should match input audio length."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        expected_duration = len(sample_audio_mono) / 44100
        assert abs(features.duration - expected_duration) < 0.01

    def test_onsets_are_within_duration(self, sample_audio_mono):
        """Onset times should be within audio duration."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        for onset_time in features.onsets:
            assert 0 <= onset_time <= features.duration


class TestStemFeatures:
    """Tests for StemFeatures dataclass."""

    def test_sample_at_time_interpolates(self, sample_audio_mono):
        """Sampling should interpolate between frames."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        val = features.sample_at_time(0.5, "envelope")
        assert 0.0 <= val <= 1.0

    def test_sample_at_time_clamps_high(self, sample_audio_mono):
        """Sampling should clamp times beyond duration."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        val = features.sample_at_time(1000.0, "envelope")
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0

    def test_sample_at_time_clamps_low(self, sample_audio_mono):
        """Sampling should clamp negative times."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        val = features.sample_at_time(-1.0, "envelope")
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0

    def test_sample_at_time_brightness(self, sample_audio_mono):
        """Sampling should work for brightness channel."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        val = features.sample_at_time(0.5, "brightness")
        assert 0.0 <= val <= 1.0

    def test_sample_at_time_invalid_feature(self, sample_audio_mono):
        """Sampling invalid feature should raise ValueError."""
        features = StemAnalyzer(sample_audio_mono, sr=44100, fps=30).extract()

        with pytest.raises(ValueError):
            features.sample_at_time(0.5, "invalid_feature")


class TestExtractAllFeatures:
    """Tests for extract_all_features function."""

    def test_returns_all_stems(self, sample_stems):
        """Should return features for all input stems."""
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        assert set(features.keys()) == set(sample_stems.keys())

    def test_all_features_are_stem_features(self, sample_stems):
        """All returned values should be StemFeatures."""
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        for name, feat in features.items():
            assert isinstance(feat, StemFeatures)

    def test_all_envelopes_are_finite(self, sample_stems):
        """All envelopes must be finite."""
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        for name, feat in features.items():
            assert np.isfinite(feat.envelope).all(), f"{name} envelope has NaN/Inf"

    def test_all_brightness_are_finite(self, sample_stems):
        """All brightness values must be finite."""
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        for name, feat in features.items():
            assert np.isfinite(feat.brightness).all(), f"{name} brightness has NaN/Inf"

    def test_all_onsets_are_finite(self, sample_stems):
        """All onset arrays must be finite."""
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        for name, feat in features.items():
            assert np.isfinite(feat.onsets).all(), f"{name} onsets has NaN/Inf"
