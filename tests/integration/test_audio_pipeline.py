"""Integration tests for audio processing pipeline.

These tests verify the full flow from audio file through
stem separation to feature extraction.
"""

import numpy as np
import soundfile as sf

from hambajuba2ba.audio import StemSeparator, extract_all_features


class TestAudioPipeline:
    """End-to-end audio processing tests."""

    def test_separator_to_features(self, tmp_path):
        """Full pipeline: audio file → stems → features."""
        # Create test audio
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        audio_path = tmp_path / "test.wav"
        sf.write(str(audio_path), audio, sr)

        # Separate
        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(str(audio_path))

        # Verify stems
        assert len(stems) == 4
        for name, data in stems.items():
            assert np.isfinite(data).all()

        # Extract features
        features, _ = extract_all_features(stems, sr=sr, fps=30)

        # Verify features
        assert len(features) == 4
        for name, feat in features.items():
            assert feat.duration > 0
            assert len(feat.envelope) > 0
            assert np.isfinite(feat.envelope).all()

    def test_pipeline_handles_short_audio(self, tmp_path):
        """Pipeline should handle very short audio clips."""
        sr = 44100
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        audio_path = tmp_path / "short.wav"
        sf.write(str(audio_path), audio, sr)

        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(str(audio_path))
        features, _ = extract_all_features(stems, sr=sr, fps=30)

        assert len(features) == 4

    def test_pipeline_handles_near_silence(self, tmp_path):
        """Near-silent audio should produce finite features, not NaN.

        Feature extraction normalizes against total energy, so an input at
        the noise floor (-60 dBFS) exercises every divide-by-tiny path.
        Digitally pure silence is not the interesting case: the separator
        rejects all-zero audio as invalid before DSP ever runs.
        """
        sr = 44100
        duration = 1.0
        rng = np.random.default_rng(seed=0)
        audio = rng.uniform(-1e-3, 1e-3, int(sr * duration)).astype(np.float32)

        audio_path = tmp_path / "silence.wav"
        sf.write(str(audio_path), audio, sr)

        sep = StemSeparator(device="cpu")
        stems = sep.separate_sync(str(audio_path))
        features, _ = extract_all_features(stems, sr=sr, fps=30)

        # Should still produce valid features (all zeros or near-zero)
        assert len(features) == 4
        for feat in features.values():
            assert np.isfinite(feat.envelope).all()

    def test_features_are_cacheable(self, sample_stems):
        """Features should be serializable for caching.

        The API caches extracted features with a TTL, so all
        attributes must be accessible and serializable.
        """
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        # Try to access all attributes (should not raise)
        for name, feat in features.items():
            _ = feat.envelope.tolist()
            _ = feat.brightness.tolist()  # Normalized spectral centroid
            _ = feat.onsets.tolist()
            _ = feat.timestamps.tolist()
            _ = feat.duration

    def test_sample_at_time_across_features(self, sample_stems):
        """Sampling should work consistently across all stems."""
        features, _ = extract_all_features(sample_stems, sr=44100, fps=30)

        t = 0.5
        for name, feat in features.items():
            env = feat.sample_at_time(t, "envelope")
            bright = feat.sample_at_time(t, "brightness")

            assert 0 <= env <= 1
            assert 0 <= bright <= 1
