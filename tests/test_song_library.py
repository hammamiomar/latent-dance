from __future__ import annotations

import asyncio

import numpy as np

from app.caching import CacheManager
from app.routers.audio import (
    _save_feature_cache,
    activate_library_song,
    get_audio_status,
    list_song_library,
)
from hambajuba2ba.audio.features import FEATURE_CACHE_VERSION, StemFeatures
from hambajuba2ba.audio.library import SongLibrary
from hambajuba2ba.config import PipelineConfig, load_from_env


def test_song_library_upserts_and_lists_metadata(tmp_path):
    library = SongLibrary(tmp_path)

    first = library.upsert_song(
        content_hash="abc123",
        filename="first.wav",
        source_type="upload",
        source_uri=None,
        duration=12.5,
        bpm=128.0,
        stems=["bass", "drums"],
        feature_version=FEATURE_CACHE_VERSION,
        feature_level="full",
        coupling_stems="physical",
        sample_rate=44100,
        mix_path=tmp_path / "abc123" / "mix.wav",
    )
    second = library.upsert_song(
        content_hash="abc123",
        filename="renamed.wav",
        source_type="upload",
        source_uri=None,
        duration=12.5,
        bpm=128.0,
        stems=["bass", "drums", "vocals", "other"],
        feature_version=FEATURE_CACHE_VERSION,
        feature_level="full",
        coupling_stems="physical",
        sample_rate=44100,
        mix_path=tmp_path / "abc123" / "mix.wav",
    )

    assert first.song_id == "abc123"
    assert second.filename == "renamed.wav"
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert library.get_song("abc123") == second
    assert library.list_songs() == [second]


def test_song_library_dir_can_be_configured_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HAMBAJUBA_AUDIO_SONG_LIBRARY_DIR", str(tmp_path))

    config = load_from_env(PipelineConfig())

    assert config.audio.song_library_dir == str(tmp_path)


def test_activate_library_song_creates_runtime_audio_cache(tmp_path):
    config = PipelineConfig()
    config.audio.song_library_dir = str(tmp_path)
    config.audio.feature_level = "full"
    config.audio.coupling_stems = "physical"
    config.audio.sample_rate = 44100

    content_hash = "songhash"
    cache_dir = tmp_path / content_hash
    cache_dir.mkdir(parents=True)
    (cache_dir / "mix.wav").write_bytes(b"mix")
    for stem in ("bass", "drums", "vocals", "other"):
        (cache_dir / f"{stem}.wav").write_bytes(b"stem")

    features = {stem: _tiny_features() for stem in ("bass", "drums", "vocals", "other")}
    _save_feature_cache(
        content_hash=content_hash,
        features=features,
        cross_stem=None,
        bpm=96.0,
        cache_root=tmp_path,
        feature_level=config.audio.feature_level,
        coupling_stems=config.audio.coupling_stems,
        sample_rate=config.audio.sample_rate,
    )

    library = SongLibrary(tmp_path)
    library.upsert_song(
        content_hash=content_hash,
        filename="cached.wav",
        source_type="upload",
        source_uri=None,
        duration=2.0,
        bpm=96.0,
        stems=features.keys(),
        feature_version=FEATURE_CACHE_VERSION,
        feature_level=config.audio.feature_level,
        coupling_stems=config.audio.coupling_stems,
        sample_rate=config.audio.sample_rate,
        mix_path=cache_dir / "mix.wav",
    )
    audio_cache = CacheManager(default_ttl=60)

    listed = asyncio.run(list_song_library(song_library=library, config=config))
    response = asyncio.run(
        activate_library_song(
            content_hash,
            audio_cache=audio_cache,
            song_library=library,
            config=config,
        )
    )

    payload = audio_cache.get(response.audio_id)
    assert listed.songs[0].song_id == content_hash
    assert listed.songs[0].status == "ready"
    assert response.song_id == content_hash
    assert response.filename == "cached.wav"
    assert response.stems == ["bass", "drums", "vocals", "other"]
    assert response.song_profile["bpm"] == 96.0
    assert response.song_analysis["anonymous"] is False
    assert response.song_analysis["metadata"]["filename"] == "cached.wav"
    assert response.song_sections[0] == 0.0
    assert payload["filename"] == "cached.wav"
    assert payload["stems"] == response.stems
    assert payload["stem_files"]["bass"].endswith("bass.wav")
    assert payload["song_profile"]["bpm"] == 96.0
    assert payload["song_analysis"]["anonymous"] is False
    assert payload["song_curves_binary"]

    status = asyncio.run(get_audio_status(response.audio_id, audio_cache=audio_cache))
    assert status.status == "complete"
    assert status.song_profile["bpm"] == 96.0
    assert status.song_analysis["anonymous"] is False
    assert status.song_analysis["metadata"]["filename"] == "cached.wav"
    assert status.song_sections[0] == 0.0


def _tiny_features() -> StemFeatures:
    timestamps = np.linspace(0.0, 2.0, 4, dtype=np.float32)
    zeros = np.zeros_like(timestamps)
    return StemFeatures(
        envelope=zeros,
        energy_smooth=zeros,
        transient=zeros,
        flux=zeros,
        brightness=zeros,
        flatness=zeros,
        flash=zeros,
        sustain=zeros,
        onsets=np.zeros(0, dtype=np.float32),
        timestamps=timestamps,
        duration=2.0,
        fps=2.0,
        hpss_ratio=0.0,
        tempo=96.0,
    )
