"""Base strategy for generation modes.

Owns the frame loop, shared state, and manager lifecycle.
Subclasses override hooks for backend-specific logic
(prompt encoding, GPU forward, destination blending, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel

from app.caching import CacheManager
from app.generation import FrameItem
from app.schemas import ExtendedStemActivity, SongIntelligenceMessage, TrackInfo
from hambajuba2ba.config.slots import BlockLinkConfig, get_base_stem
from hambajuba2ba.audio.prominence import compute_all_prominences
from hambajuba2ba.audio.sampler import AudioSampler
from hambajuba2ba.bridge import SteeringComputation, PhysicsManager, SpatialManager
from hambajuba2ba.bridge.clock import AudioClock
from hambajuba2ba.bridge.composition import CompositionEngine
from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.device import empty_cache
from app.strategies.managers import FrameManager

logger = logging.getLogger("uvicorn")

BINARY_KIND_JPEG_FRAME = b"\x01"
BINARY_KIND_SONG_CURVES = b"\x02"


class GenerationStrategy(ABC):
    """Abstract base class for generation strategies.

    Owns all shared state (audio clock, slot configs, managers,
    frame timing, lookahead, composition). Subclasses implement
    hooks for backend-specific behavior.
    """

    def __init__(
        self,
        pipeline: Any,
        config: PipelineConfig,
        websocket: WebSocket,
        audio_cache: CacheManager,
        *,
        gpu_lock: asyncio.Lock,
        cpu_executor: ThreadPoolExecutor,
    ):
        # Injected dependencies
        self.pipeline = pipeline
        self.config = config
        self.websocket = websocket
        self.cache = audio_cache
        self._gpu_lock = gpu_lock
        self._cpu_executor = cpu_executor

        # Audio clock (PLL-style sync with frontend)
        self.clock = AudioClock(
            rate=config.strategy.audio_clock_rate,
            alpha=config.strategy.audio_clock_alpha,
        )

        # Slot configs: per-slot steering configuration.
        # Values are mutable dataclasses; handlers and hooks mutate them in-place.
        # _get_slot_configs() returns the dict directly when _auto_mode=True,
        # or copies with auto_config=False when _auto_mode=False.
        self.slot_configs: Dict[str, BlockLinkConfig] = {}
        self._init_default_slot_configs()

        # Audio features (loaded from cache during setup)
        self.stem_features: Dict[str, Any] = {}
        self.stem_classifications: Dict[str, Any] = {}
        self._bpm: float = config.audio.default_bpm
        self._bpm_raw: float = config.audio.default_bpm

        # Managers (initialized in setup, not __init__)
        self._physics: Optional[PhysicsManager] = None
        self._spatial: Optional[SpatialManager] = None
        self._steering: Optional[SteeringComputation] = None
        self._frames: Optional[FrameManager] = None
        self.audio_sampler: Optional[AudioSampler] = None
        self._composition: Optional[CompositionEngine] = None

        # Frame timing
        self._fps: float = config.streaming.fps
        self._next_due_ts: Optional[float] = None
        self._prev_frame_t_start: float = 0.0
        self._frame_count: int = 0
        self._last_frame_time: float = 0.0
        self._last_encode_skip_log: float = 0.0

        # Predictive lookahead
        self._lookahead_s: float = 0.0
        self._lookahead_update_interval: float = 5.0
        self._last_lookahead_update: float = 0.0

        # Steering mode. AUTO allows auto_config derivation; MANUAL preserves
        # explicit block configs. Prominence/rank scaling still applies later.
        self._auto_mode: bool = True
        self._last_activity_time: float = 0.0
        self._last_block_activity: Dict[str, Dict[str, float]] = {}

        # Noise seeds for composition
        self._noise_seed_a: int | None = config.seed
        self._noise_seed_b: int | None = config.seed + 1

        # Track
        self._track_duration: float = 0.0

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_default_slot_configs(self) -> None:
        """Populate self.slot_configs with backend-specific defaults."""
        ...

    @abstractmethod
    async def handle_message(self, message: BaseModel) -> Optional[dict]:
        """Handle control messages."""
        ...

    # ------------------------------------------------------------------
    # Overridable hooks — subclasses MAY override (defaults are no-ops)
    # ------------------------------------------------------------------

    def _auto_derive_slot_configs(self) -> None:
        """Derive slot config fields from stem classification. Override per backend."""
        pass

    async def _setup_backend(self, params: BaseModel, cached: dict) -> None:
        """Backend-specific setup (e.g. encode prompt, init destinations)."""
        pass

    def _cleanup_backend(self) -> None:
        """Backend-specific teardown (e.g. clear embeddings)."""
        pass

    # ------------------------------------------------------------------
    # Lifecycle — setup / cleanup (shared across all backends)
    # ------------------------------------------------------------------

    async def setup(self, params: BaseModel) -> None:
        """Initialize strategy state for generation.

        Shared skeleton: load cache, classify stems, init managers,
        call backend hook, init composition, start clock.
        """
        from hambajuba2ba.audio import classify_component

        # Load audio features from cache
        cached = self.cache.get(params.audio_id)
        if cached is None:
            raise ValueError(f"Audio {params.audio_id} not found in cache")

        self.stem_features = cached.get("features", {})
        self._bpm_raw = cached.get("bpm", 120.0)
        if self._bpm_raw <= 0 or not math.isfinite(self._bpm_raw):
            self._bpm_raw = 120.0
        self._bpm = min(max(self._bpm_raw, 60.0), 180.0)
        self._track_duration = cached.get("duration", 0.0)

        # Classify stems (drives physics model selection)
        self.stem_classifications = {}
        for stem, features in self.stem_features.items():
            if features.hpss_ratio is not None:
                self.stem_classifications[stem] = classify_component(features)
            else:
                self.stem_classifications[stem] = None

        logger.info(
            f"Loaded features: stems={list(self.stem_features.keys())}, "
            f"BPM={self._bpm_raw:.1f} (physics={self._bpm:.1f})"
        )
        classified_count = sum(1 for c in self.stem_classifications.values() if c is not None)
        logger.info(f"Classified {classified_count}/{len(self.stem_features)} stems for physics")

        self._auto_derive_slot_configs()
        self._init_managers(cached)

        # Send track info
        track_info = TrackInfo(
            audio_id=params.audio_id,
            duration=cached.get("duration", 0.0),
            bpm=self._bpm_raw,
            stems=list(self.stem_features.keys()),
        )
        await self.websocket.send_json(track_info.model_dump())

        # Send song intelligence once per track setup. The packed curves are
        # prefixed so the frontend can never confuse them with JPEG frames.
        song_profile = cached.get("song_profile")
        if song_profile is not None:
            sections = cached.get("song_sections") or song_profile.get("sections", [])
            await self.websocket.send_json(
                SongIntelligenceMessage(
                    audio_id=params.audio_id,
                    profile=song_profile,
                    sections=sections,
                    analysis=cached.get("song_analysis"),
                ).model_dump()
            )
            curves_binary = cached.get("song_curves_binary")
            if curves_binary:
                await self.websocket.send_bytes(BINARY_KIND_SONG_CURVES + curves_binary)

        # Backend-specific setup (prompt encoding, destinations, etc.)
        await self._setup_backend(params, cached)

        # Composition (shared — uses pipeline.engine._latent_shape)
        self._init_composition(cached)

        # Boundary-time cache relief on any device (scene start)
        empty_cache(self.pipeline.device)
        logger.info("Cleared device cache before generation")

        # Start clock
        now = time.perf_counter()
        self.clock.seek(0.0, now)
        self.clock.play(0.0, now)
        self._last_frame_time = now
        self._next_due_ts = None

    def _init_managers(self, cached: dict) -> None:
        """Initialize all managers."""
        self._frames = FrameManager(self._cpu_executor)

        self._physics = PhysicsManager(self._bpm)
        self._physics.initialize(self.slot_configs, self.stem_features, self.stem_classifications)
        logger.info(f"Initialized physics for {len(self._physics.stems)} stems")

        self._spatial = SpatialManager(
            device=self.pipeline.device,
            # Follow the config dtype (float16 on CUDA, float32 on mps/cpu)
            # so masks match model dtype on every device.
            dtype=self.config.get_torch_dtype(),
            latent_h=self.config.latent_height,
            latent_w=self.config.latent_width,
        )
        self._spatial.initialize()

        self.audio_sampler = AudioSampler(
            stem_features=self.stem_features,
            fps=self._fps,
            cross_stem_features=cached.get("cross_stem_features"),
            classifications=self.stem_classifications,
        )

        self._steering = SteeringComputation()
        self._steering.initialize(list(self.stem_features.keys()))

    def _init_composition(self, cached: dict) -> None:
        """Initialize composition system (noise circular walk).

        Pre-computes two noise buffers and builds motion signals from
        drums beat grid + tonal distance.
        """
        engine = self.pipeline.engine
        if engine is None:
            logger.warning("Engine not available, skipping composition init")
            return

        self._composition = CompositionEngine(
            shape=engine._latent_shape,
            device=engine.device,
            dtype=engine.dtype,
        )
        if self._noise_seed_a is not None:
            noise_a = engine.make_noise(self._noise_seed_a)
            self._composition.load_noise("a", noise_a, self._noise_seed_a)
        if self._noise_seed_b is not None:
            noise_b = engine.make_noise(self._noise_seed_b)
            self._composition.load_noise("b", noise_b, self._noise_seed_b)
        logger.info(
            "CompositionEngine: seed_a=%s, seed_b=%s", self._noise_seed_a, self._noise_seed_b
        )

        drums = self.stem_features.get("drums")
        if drums is None:
            for stem_features in self.stem_features.values():
                if stem_features.beat_frames is not None:
                    drums = stem_features
                    break

        if drums is not None and drums.beat_frames is not None and len(drums.beat_frames) > 1:
            import librosa
            sr = cached.get("sr", 44100)
            hop_length = sr // int(drums.fps)
            beat_times = librosa.frames_to_time(
                drums.beat_frames, sr=sr, hop_length=hop_length
            )
            energy_at_beats = drums.energy_at_beats if drums.energy_at_beats is not None else np.ones(len(beat_times), dtype=np.float32)

            tonal_distance = None
            for stem_features in self.stem_features.values():
                if stem_features.tonal_distance is not None:
                    tonal_distance = stem_features.tonal_distance
                    break

            if tonal_distance is None:
                tonal_distance = np.zeros(len(drums.timestamps), dtype=np.float32)

            self._composition.load_motion(
                beat_times=beat_times,
                energy_at_beats=energy_at_beats,
                tonal_distance=tonal_distance,
                transient=drums.transient,
                energy_smooth=drums.energy_smooth,
                timestamps=drums.timestamps,
            )
            logger.info(
                "Composition motion: %d beats, tempo=%.1f BPM, tonal_distance=%s",
                len(beat_times),
                drums.tempo or self._bpm,
                "available" if tonal_distance.any() else "unavailable",
            )
        else:
            logger.warning("No drums beat data, composition will use static noise_a")

    async def cleanup(self) -> None:
        """Clean up resources when generation stops."""
        self.clock.pause()
        self._frame_count = 0
        self._next_due_ts = None
        self._lookahead_s = 0.0
        self._last_lookahead_update = 0.0
        self._prev_frame_t_start = 0.0
        if self._frames:
            await self._frames.cleanup()
            self._frames.clear_history()

        self.stem_features = {}
        self.stem_classifications = {}
        self._composition = None

        if self._physics:
            self._physics.clear()
        if self._spatial:
            self._spatial.clear()
        if self._steering:
            self._steering.reset()

        self._cleanup_backend()

    @abstractmethod
    def _is_ready(self) -> bool:
        """Can we generate? (e.g. prompt encoded, models loaded)"""
        ...

    @abstractmethod
    def _pre_generate(self, audio_time: float, dt: float) -> Any:
        """Pre-GPU work (e.g. destination blending). Returns context for _gpu_forward."""
        ...

    @abstractmethod
    def _gpu_forward(self, steering: dict, pre_ctx: Any, audio_time: float) -> Optional[tuple]:
        """Run the GPU pipeline. Returns (cpu_tensor, prep_ms, infer_ms, d2h_ms) or None."""
        ...

    # ------------------------------------------------------------------
    # Frame loop — template method (shared across all backends)
    # ------------------------------------------------------------------

    async def next_frame_batch(self) -> List[FrameItem]:
        """Generate the next batch of frames.

        Template method: shared frame loop skeleton with 4 hooks
        (_is_ready, _pre_generate, _gpu_forward, _build_telemetry)
        that subclasses override for backend-specific behavior.
        """
        if not self.clock.playing:
            await asyncio.sleep(0.1)
            return []

        if not self._is_ready():
            await asyncio.sleep(0.1)
            return []

        t_start = time.perf_counter()

        # Use hardware-measured FPS for scheduling
        self._fps = self._frames.measured_fps

        # Predictive lookahead
        self._update_lookahead(t_start)
        raw_audio_time = self.clock.now(t_start)
        audio_time = (
            min(raw_audio_time + self._lookahead_s, self._track_duration)
            if self._track_duration > 0
            else raw_audio_time
        )
        self._frame_count += 1

        # Compute dt
        dt = t_start - self._last_frame_time if self._last_frame_time > 0 else 1 / self._fps
        dt = min(dt, 0.1)
        self._last_frame_time = t_start

        # Periodic cache clear — deliberately CUDA-only: this is a
        # CUDA-allocator workaround (slated for measure-then-delete);
        # an MPS cache clear mid-loop would be a new, unmeasured stall.
        if self.pipeline.device == "cuda" and self._frame_count % 300 == 0:
            empty_cache(self.pipeline.device)

        # Collect previous encode
        prev_jpeg = await self._frames.collect_previous()

        # Compute steering
        t_steer = time.perf_counter()
        steering, block_activity = self._steering.compute(
            block_configs=self._get_slot_configs(),
            audio_sampler=self.audio_sampler,
            physics=self._physics,
            audio_time=audio_time,
            dt=dt,
            classifications=self.stem_classifications,
        )
        self._last_block_activity = block_activity

        # Pre-generate hook (SAE: destination blending, MF-RAE: concept prep)
        pre_ctx = self._pre_generate(audio_time, dt)
        steer_ms = (time.perf_counter() - t_steer) * 1000

        # If encoder is falling behind, skip GPU work to reduce latency
        frame_interval_ms = 1000.0 / max(1.0, self._fps)
        pending_age_ms = self._frames.pending_age_ms() if self._frames else 0.0
        if self._frames and self._frames.is_encode_busy() and pending_age_ms > frame_interval_ms:
            now = time.perf_counter()
            if now - self._last_encode_skip_log > 1.0:
                logger.info(
                    f"[ENCODE] Busy ({pending_age_ms:.1f}ms), "
                    "skipping frame to keep latency low"
                )
                self._last_encode_skip_log = now

            items = []
            if prev_jpeg is not None:
                items.append(
                    FrameItem(
                        kind="frame",
                        payload=self._binary_frame_payload(prev_jpeg),
                        due_ts=self._compute_due_ts(t_start),
                        produced_at=self._prev_frame_t_start or t_start,
                    )
                )
            self._prev_frame_t_start = t_start
            items.extend(self._build_telemetry(t_start, audio_time, pre_ctx))
            return items

        # GPU forward + D2H transfer
        loop = asyncio.get_running_loop()

        async with self._gpu_lock:
            result = await loop.run_in_executor(
                None, self._gpu_forward, steering, pre_ctx, audio_time,
            )

        if result is None:
            return []

        cpu_tensor, prep_ms, infer_ms, d2h_ms = result

        # Check encode again (might have finished during GPU+D2H)
        if prev_jpeg is None:
            prev_jpeg = self._frames.collect_if_ready()

        # Start new encode in background
        self._frames.start_encode(cpu_tensor, self.config.jpeg_quality)

        # Record timing
        t_end = time.perf_counter()
        self._frames.record_timing(
            total_ms=(t_end - t_start) * 1000,
            steer_ms=steer_ms + prep_ms,
            infer_ms=infer_ms,
            d2h_ms=d2h_ms,
            audio_time=audio_time,
        )
        self._frames.log_timing_if_due()

        # Build result
        items = []
        if prev_jpeg is not None:
            items.append(
                FrameItem(
                    kind="frame",
                    payload=self._binary_frame_payload(prev_jpeg),
                    due_ts=self._compute_due_ts(t_start),
                    produced_at=self._prev_frame_t_start or t_start,
                )
            )
        self._prev_frame_t_start = t_start

        items.extend(self._build_telemetry(t_start, audio_time, pre_ctx))
        return items

    def _build_telemetry(
        self, t_start: float, audio_time: float, pre_ctx: Any
    ) -> List[FrameItem]:
        """Build telemetry items (throttled). Override to add backend-specific telemetry."""
        items = []

        # Activity telemetry (~10Hz)
        if t_start - self._last_activity_time > 0.1:
            extended = self.audio_sampler.get_extended_activity(audio_time)
            prominence = self._compute_prominence_map(audio_time)
            items.append(FrameItem(
                kind="json",
                payload=ExtendedStemActivity(
                    audio_time=audio_time,
                    stems=extended,
                    prominence=prominence,
                    blocks=self._last_block_activity or None,
                ).model_dump(),
                due_ts=None,
            ))
            self._last_activity_time = t_start

        return items

    # ------------------------------------------------------------------
    # Shared concrete methods — used by all backends
    # ------------------------------------------------------------------

    def _binary_frame_payload(self, jpeg: bytes) -> bytes:
        """Prefix JPEG frames with an explicit binary kind byte."""
        return BINARY_KIND_JPEG_FRAME + jpeg

    def _get_slot_configs(self) -> Dict[str, BlockLinkConfig]:
        """Return active slot configs with global auto_config override applied."""
        if self._auto_mode:
            return self.slot_configs
        return {
            slot: replace(config, auto_config=False)
            for slot, config in self.slot_configs.items()
        }

    def _compute_due_ts(self, now: float) -> float:
        """Return the next due_ts for frame pacing."""
        interval = 1.0 / max(1.0, self._fps)
        if self._next_due_ts is None or self._next_due_ts < now - 2 * interval:
            self._next_due_ts = now + interval
        else:
            self._next_due_ts += interval
        return self._next_due_ts

    def get_perf_snapshot(self) -> dict:
        """Encoder/timing stats + scheduler state for perf telemetry.

        The one sanctioned way for transport code to read strategy perf
        state — WebSocketManager must not reach into private attributes.
        """
        snapshot = self._frames.get_perf_snapshot() if self._frames else {}
        snapshot["measured_fps"] = self._frames.measured_fps if self._frames else 0.0
        snapshot["lookahead_ms"] = self._lookahead_s * 1000
        return snapshot

    @property
    def measured_interval_s(self) -> float:
        """Seconds per frame from measured FPS (config rate before calibration)."""
        if self._frames is not None:
            return self._frames.measured_interval_s
        return 1.0 / max(1.0, self._fps)

    def _update_lookahead(self, now: float) -> None:
        """Recompute predictive lookahead from FrameManager measurements.

        Lookahead = frame_production + double_buffer_delay + client_render.
        Updated via EMA every ~5s for smooth transitions.
        """
        if now - self._last_lookahead_update < self._lookahead_update_interval:
            return
        self._last_lookahead_update = now

        summary = self._frames.get_timing_summary()
        if not summary:
            return

        avg_total_ms = summary["avg_total_ms"]
        frame_interval_ms = 1000.0 / max(1.0, self._frames.measured_fps)
        client_render_ms = self.config.strategy.client_render_ms

        estimate_ms = avg_total_ms + frame_interval_ms + client_render_ms
        estimate_s = estimate_ms / 1000.0

        if self._lookahead_s == 0.0:
            self._lookahead_s = estimate_s
        else:
            self._lookahead_s = 0.8 * self._lookahead_s + 0.2 * estimate_s

    def _compute_prominence_map(self, audio_time: float) -> Optional[dict]:
        """Compute prominence map for telemetry (per base stem)."""
        if self._steering is None or self.audio_sampler is None:
            return None

        frame_idx = self.audio_sampler.time_to_frame(audio_time)
        slot_configs = self._get_slot_configs()

        stem_ranks: Dict[str, Optional[int]] = {}
        for cfg in slot_configs.values():
            if not cfg.enabled:
                continue
            base = get_base_stem(cfg.link_target)
            if base:
                stem_ranks[base] = cfg.sae_rank

        if not stem_ranks:
            return None

        prominence = compute_all_prominences(
            engine=self._steering.get_prominence_engine(),
            stem_ranks=stem_ranks,
            audio_sampler=self.audio_sampler,
            audio_time=audio_time,
            frame_idx=frame_idx,
            dt=1.0 / max(1.0, self._fps),
        )

        result = {}
        for stem, value in prominence.items():
            result[stem] = {
                "prominence": value,
                "surprise_active": self._steering.get_prominence_engine().get_surprise_active(stem),
                "rank": stem_ranks.get(stem),
            }
        return result

    def _get_active_base_stems(self) -> list[str]:
        """Return list of active base stems for destination reactive weighting."""
        active = []
        for cfg in self._get_slot_configs().values():
            if not cfg.enabled:
                continue
            base = get_base_stem(cfg.link_target) if self._steering else None
            if base:
                active.append(base)
        return active
