"""The audio processing pipeline must not block the event loop.

Heavy DSP stages (beat detection, virtual stems, feature extraction, cache IO)
run in worker threads via asyncio.to_thread so status polls and live WebSocket
sessions stay responsive while a song is processed. Regression guard for the
event-loop stall fixed in the July 2026 preflight.
"""

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.caching import CacheManager
from app.routers import audio
from hambajuba2ba.config import PipelineConfig

# A blocked loop gaps for the whole heavy stage; a free loop gaps only by
# scheduler noise. Keep these far enough apart that the two can't overlap.
HEAVY_STAGE_S = 1.2
MAX_LOOP_GAP_S = 0.5


class _FakeSongLibrary:
    def __init__(self, root: Path):
        self.root = root

    def upsert_song(self, **kwargs) -> None:
        return None


def _write_stub_song(cache_root: Path, content_hash: str, sr: int) -> None:
    """Materialize a cached song (4 stems + mix) so the cache-hit path runs."""
    song_dir = cache_root / content_hash
    song_dir.mkdir(parents=True)
    samples = np.zeros(int(sr * 0.2), dtype=np.float32)
    for stem in ("bass", "drums", "vocals", "other"):
        sf.write(str(song_dir / f"{stem}.wav"), samples, sr)
    sf.write(str(song_dir / "mix.wav"), samples, sr)


def test_pipeline_stages_run_off_the_event_loop(tmp_path, monkeypatch):
    config = PipelineConfig()
    stage_thread_ids: list[int] = []

    def fake_detect_beats(audio_data, sr):
        stage_thread_ids.append(threading.get_ident())
        return np.array([0.0]), 120.0

    def fake_extract_virtual_stems(stems, sr):
        stage_thread_ids.append(threading.get_ident())
        return stems

    def fake_extract_all_features(all_stems, **kwargs):
        stage_thread_ids.append(threading.get_ident())
        time.sleep(HEAVY_STAGE_S)
        return {}, None

    def fake_build_payload(**kwargs):
        stage_thread_ids.append(threading.get_ident())
        return {
            "stems": kwargs.get("stems", []),
            "duration": kwargs.get("duration", 0.0),
            "bpm": kwargs.get("bpm", 120.0),
        }

    monkeypatch.setattr(audio, "detect_beats", fake_detect_beats)
    monkeypatch.setattr(audio, "extract_virtual_stems", fake_extract_virtual_stems)
    monkeypatch.setattr(audio, "extract_all_features", fake_extract_all_features)
    monkeypatch.setattr(audio, "_build_audio_cache_payload", fake_build_payload)
    monkeypatch.setattr(audio, "_load_feature_cache", lambda *a, **kw: None)
    monkeypatch.setattr(audio, "_save_feature_cache", lambda **kw: None)

    _write_stub_song(tmp_path, "stubhash", config.audio.sample_rate)

    async def scenario() -> None:
        loop_gaps: list[float] = []
        stop = asyncio.Event()

        async def heartbeat() -> None:
            prev = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                loop_gaps.append(now - prev)
                prev = now

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            result = await audio._process_audio_pipeline(
                audio_id="offload-test",
                tmp_path=None,
                content_hash="stubhash",
                config=config,
                audio_cache=CacheManager(),
                song_library=_FakeSongLibrary(tmp_path),
                cache_root=tmp_path,
                filename="stub.wav",
            )
        finally:
            stop.set()
            await heartbeat_task

        assert result.bpm == 120.0
        assert sorted(result.stems) == ["bass", "drums", "other", "vocals"]

        # Every heavy stage must have run in a worker thread
        main_thread = threading.get_ident()
        assert stage_thread_ids, "no pipeline stages were exercised"
        assert all(tid != main_thread for tid in stage_thread_ids)

        # The loop must have stayed responsive throughout the heavy stage
        assert max(loop_gaps) < MAX_LOOP_GAP_S, (
            f"event loop stalled for {max(loop_gaps) * 1000:.0f}ms "
            f"(heavy stage is {HEAVY_STAGE_S * 1000:.0f}ms)"
        )

    asyncio.run(scenario())
