"""Tests for song intelligence profiles and curve transfer payloads."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from hambajuba2ba.audio.coupling import CrossStemFeatures
from hambajuba2ba.audio.features import StemFeatures
from hambajuba2ba.audio.profile import (
    MAJOR_PROFILE,
    build_song_analysis,
    build_song_curves,
    build_song_profile,
    build_stem_profile,
    detect_sections,
    estimate_key,
    pack_song_curves_binary,
    summarize_tension_arc,
)


def make_features(
    *,
    duration: float = 24.0,
    fps: int = 4,
    hpss_ratio: float = 0.2,
    energy: float = 0.5,
    brightness: float = 0.1,
    chroma: np.ndarray | None = None,
    pitch_confidence: np.ndarray | None = None,
    tension: np.ndarray | None = None,
    novelty_long: np.ndarray | None = None,
) -> StemFeatures:
    timestamps = np.linspace(0.0, duration, int(duration * fps) + 1, dtype=np.float32)
    n = len(timestamps)
    energy_curve = np.full(n, energy, dtype=np.float32)
    flux = np.zeros(n, dtype=np.float32)
    flux[:: max(1, fps * 2)] = 1.0

    return StemFeatures(
        envelope=energy_curve.copy(),
        energy_smooth=energy_curve.copy(),
        transient=flux.copy(),
        flux=flux.copy(),
        brightness=np.full(n, brightness, dtype=np.float32),
        flatness=np.full(n, 0.1, dtype=np.float32),
        flash=flux.copy(),
        sustain=energy_curve.copy(),
        onsets=np.arange(0.0, duration, 2.0, dtype=np.float32),
        timestamps=timestamps,
        duration=duration,
        fps=float(fps),
        hpss_ratio=hpss_ratio,
        tension=tension if tension is not None else np.linspace(0.0, 1.0, n, dtype=np.float32),
        tonal_distance=np.linspace(1.0, 0.0, n, dtype=np.float32),
        chroma=chroma,
        pitch_confidence=pitch_confidence,
        novelty_long=novelty_long,
    )


def test_stem_profile_is_descriptive_for_physical_stem():
    features = make_features(
        hpss_ratio=0.85,
        energy=0.75,
        brightness=0.02,
        pitch_confidence=np.zeros(97, dtype=np.float32),
    )

    profile = build_stem_profile("drums", features)

    assert profile.name == "drums"
    assert profile.role == "percussive"
    assert profile.texture == "percussive"
    assert profile.frequency_range == "mid"
    assert profile.mean_energy == 0.75
    assert profile.has_pitch is False
    assert profile.has_tension is True
    assert profile.onset_density > 0


def test_estimate_key_from_harmonic_chroma():
    chroma = np.tile(MAJOR_PROFILE[:, None], (1, 16)).astype(np.float32)
    features = {"other": make_features(chroma=chroma, hpss_ratio=0.1)}

    key, confidence = estimate_key(features)

    assert key == "C major"
    assert confidence > 0.99


def test_detect_sections_from_novelty_peaks():
    timestamps = np.arange(0, 31, dtype=np.float32)
    novelty = np.zeros_like(timestamps)
    novelty[10] = 1.0
    novelty[22] = 0.9

    assert detect_sections(novelty, timestamps) == [0.0, 10.0, 22.0]


def test_summarize_tension_arc_reports_trends():
    timestamps = np.arange(0, 10, dtype=np.float32)
    tension = np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)

    arc = summarize_tension_arc(tension, timestamps, n_points=4)

    assert len(arc) == 4
    assert arc[0]["trend"] == "stable"
    assert {point["trend"] for point in arc[1:]} == {"rising"}


def test_song_profile_filters_to_physical_stems_and_coupling():
    features = {
        "bass": make_features(hpss_ratio=0.2),
        "drums": make_features(hpss_ratio=0.9),
        "drums_high": make_features(hpss_ratio=0.9),
    }
    cross = CrossStemFeatures(
        plv={},
        lock_index={
            ("bass", "drums"): np.full(97, 0.7, dtype=np.float32),
            ("drums", "drums_high"): np.full(97, 0.4, dtype=np.float32),
        },
        spectral_overlap=np.eye(3, dtype=np.float32),
        call_response={},
        stem_names=["bass", "drums", "drums_high"],
        fps=4.0,
    )

    profile = build_song_profile(features, cross, bpm=123.0, duration=24.0)

    assert set(profile.stems) == {"bass", "drums"}
    assert profile.coupling == {"bass-drums": pytest.approx(0.7)}
    assert profile.bpm == 123.0
    assert profile.sections[0] == 0.0


def test_song_analysis_describes_all_available_link_targets_with_metadata_when_supplied():
    features = {
        "bass": make_features(hpss_ratio=0.25, energy=0.8),
        "drums": make_features(hpss_ratio=0.9, energy=0.65, brightness=0.35),
        "vocals": make_features(
            hpss_ratio=0.2,
            energy=0.45,
            pitch_confidence=np.full(97, 0.8, dtype=np.float32),
        ),
        "other": make_features(hpss_ratio=0.3, energy=0.35, brightness=0.6),
        "drums_high": make_features(hpss_ratio=0.85, energy=0.55, brightness=0.9),
        "other_high": make_features(hpss_ratio=0.35, energy=0.4, brightness=0.85),
    }

    analysis = build_song_analysis(
        features,
        None,
        bpm=123.0,
        duration=24.0,
        metadata={
            "filename": "Robert_Miles_-_Children_Dream_Version.wav",
        },
    )

    assert analysis["version"] == "hamba-song-analysis/v1"
    assert analysis["anonymous"] is False
    assert "filename included" in analysis["metadata_policy"]
    assert analysis["metadata"]["filename"] == "Robert_Miles_-_Children_Dream_Version.wav"
    assert "artist" not in analysis["metadata"]
    assert "title" not in analysis["metadata"]
    assert "genre" not in analysis["metadata"]

    targets = analysis["link_targets"]
    assert "bass" in targets
    assert "drums_percussive" in targets
    assert "drums_high" in targets
    assert "tension" in targets
    assert "global" in targets
    assert targets["drums_percussive"]["preferred_intensity_source"] == "transient"
    assert "prompt_position" in targets["vocals"]["good_for"]
    assert analysis["ranked_drivers"]["primary_driver"]
    assert analysis["ranked_drivers"]["rhythmic_hits"]
    assert analysis["curve_catalog"]["targets"]["drums_high"]
    assert "transient" in analysis["curve_catalog"]["targets"]["drums_high"]


def test_song_analysis_remains_anonymous_without_metadata():
    features = {
        "bass": make_features(hpss_ratio=0.25, energy=0.8),
        "drums": make_features(hpss_ratio=0.9, energy=0.65),
    }

    analysis = build_song_analysis(features, None, bpm=123.0, duration=24.0)

    assert analysis["anonymous"] is True
    assert analysis["metadata"] == {}
    assert "metadata unavailable" in analysis["metadata_policy"]
    section_summary = analysis["structure"]["section_target_summary"]
    assert section_summary
    assert section_summary[0]["ranked_targets"]["rhythmic_hits"]


def test_build_song_curves_names_lock_index_pairs():
    features = {
        "bass": make_features(),
        "drums": make_features(),
    }
    cross = CrossStemFeatures(
        plv={},
        lock_index={("bass", "drums"): np.linspace(0.0, 1.0, 97, dtype=np.float32)},
        spectral_overlap=np.eye(2, dtype=np.float32),
        call_response={},
        stem_names=["bass", "drums"],
        fps=4.0,
    )

    curves = build_song_curves(features, cross)

    assert "timestamps" in curves
    assert "tension" in curves
    assert "tonal_distance" in curves
    assert "novelty_long" in curves
    assert "target:bass:energy_smooth" in curves
    assert "target:drums:transient" in curves
    assert "target:drums_percussive:energy_smooth" in curves
    assert "lock_index:bass-drums" in curves
    assert curves["target:bass:energy_smooth"].dtype == np.float32
    assert curves["lock_index:bass-drums"].dtype == np.float32


def test_pack_song_curves_binary_uses_named_float32_format():
    curves = {
        "timestamps": np.array([0.0, 1.0], dtype=np.float32),
        "tension": np.array([0.25, 0.75], dtype=np.float32),
    }

    payload = pack_song_curves_binary(curves)

    offset = 0
    count = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    assert count == 2

    names = []
    for _ in range(count):
        name_length = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        name = payload[offset : offset + name_length].decode("utf-8")
        offset += name_length
        float_count = struct.unpack_from("<I", payload, offset)[0]
        offset += 4 + float_count * 4
        names.append(name)

    assert names == ["timestamps", "tension"]
