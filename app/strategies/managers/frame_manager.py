"""Frame encoding and double-buffering manager.

Handles the async pipeline: encodes previous frame in background
while GPU computes the current frame. This overlaps CPU encode time
with GPU compute time for ~30% throughput improvement.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, Optional

import torch

from hambajuba2ba.generation.encoding import encode_cpu_tensor

logger = logging.getLogger("uvicorn")


@dataclass
class FrameTiming:
    """Timing breakdown for a single frame."""

    total_ms: float
    steer_ms: float   # Steering + physics + spatial + destinations
    infer_ms: float   # GPU inference (launch + cudaSync wait)
    d2h_ms: float     # D2H memcpy (after sync)
    encode_ms: float  # Actual JPEG encode time (measured inside thread)
    audio_time: float


class FrameManager:
    """Manages frame encoding pipeline with double-buffering.

    Encodes the previous frame while GPU computes the current one.
    This overlaps CPU and GPU work for better throughput.

    Also serves as the source of truth for measured FPS via EMA
    smoothing of actual frame production times.

    Usage:
        manager = FrameManager(cpu_executor)

        # In frame loop:
        prev_jpeg = await manager.collect_previous()  # Non-blocking
        # ... GPU work ...
        cpu_tensor = gpu_to_cpu_tensor(gpu_img)
        manager.start_encode(cpu_tensor, jpeg_quality)
    """

    # Frames needed before switching from bootstrap to measured FPS
    CALIBRATION_THRESHOLD = 30

    # Bootstrap FPS: conservative default so scheduler never outpaces GPU
    BOOTSTRAP_FPS = 45.0

    def __init__(
        self,
        cpu_executor: ThreadPoolExecutor,
        history_size: int = 100,
    ):
        """Initialize frame manager.

        Args:
            cpu_executor: Thread pool for CPU-bound encode work
            history_size: Number of frame timings to keep for profiling
        """
        self._executor = cpu_executor
        self._pending_task: Optional[asyncio.Future] = None
        self._pending_start: float = 0.0
        self._last_busy_warning: float = 0.0

        # Instance-level timing history (FIX: was global, leaked across sessions)
        self._timing_history: Deque[FrameTiming] = deque(maxlen=history_size)

        # Encode timing (actual TurboJPEG time, measured inside executor thread)
        self._last_encode_ms: float = 0.0

        # Periodic profiling log (perf_counter-based, not audio_time)
        self._last_profile_log: float = 0.0

        # FPS measurement via EMA (0.85/0.15 converges in ~10 frames)
        self._measured_fps: float = self.BOOTSTRAP_FPS
        self._calibration_count: int = 0

    async def collect_previous(self) -> Optional[bytes]:
        """Collect encoded JPEG from previous frame if ready.

        Non-blocking: if encode not done yet, returns None and leaves
        task running (we'll catch it next frame or drop it).

        Returns:
            JPEG bytes if ready, None otherwise
        """
        if self._pending_task is None:
            return None

        if not self._pending_task.done():
            # Not ready yet - don't block GPU, leave running
            return None

        # Task complete - collect result
        try:
            jpeg, enc_ms = self._pending_task.result()
            self._last_encode_ms = enc_ms
            return jpeg
        except Exception as e:
            logger.warning(f"Encode task failed: {e}")
            return None
        finally:
            self._pending_task = None

    def collect_if_ready(self) -> Optional[bytes]:
        """Synchronous version of collect_previous for use in GPU callback."""
        if self._pending_task is None:
            return None

        if not self._pending_task.done():
            return None

        try:
            jpeg, enc_ms = self._pending_task.result()
            self._last_encode_ms = enc_ms
            return jpeg
        except Exception:
            return None
        finally:
            self._pending_task = None

    def start_encode(
        self,
        cpu_tensor: torch.Tensor,
        jpeg_quality: int = 75,
    ) -> None:
        """Start background JPEG encoding for current frame.

        If a previous task is still running, logs warning and skips.
        This prevents queue buildup when encode is slower than generation.

        Args:
            cpu_tensor: CPU tensor to encode (H, W, C) or (C, H, W)
            jpeg_quality: JPEG compression quality (0-100)
        """
        if self._pending_task is not None and not self._pending_task.done():
            now = time.perf_counter()
            if now - self._last_busy_warning > 1.0:
                logger.warning("Previous encode still running, skipping")
                self._last_busy_warning = now
            return

        loop = asyncio.get_running_loop()
        self._pending_start = time.perf_counter()
        self._pending_task = loop.run_in_executor(
            self._executor,
            self._timed_encode,
            cpu_tensor,
            jpeg_quality,
        )

    def _timed_encode(self, cpu_tensor: torch.Tensor, quality: int) -> tuple:
        """Encode to JPEG and measure actual encode time."""
        t0 = time.perf_counter()
        jpeg = encode_cpu_tensor(cpu_tensor, quality)
        enc_ms = (time.perf_counter() - t0) * 1000
        return jpeg, enc_ms

    def is_encode_busy(self) -> bool:
        """Return True if an encode task is still running."""
        return self._pending_task is not None and not self._pending_task.done()

    def pending_age_ms(self) -> float:
        """Return age (ms) of the pending encode task, or 0 if none."""
        if not self.is_encode_busy():
            return 0.0
        return (time.perf_counter() - self._pending_start) * 1000.0

    def record_timing(
        self,
        total_ms: float,
        steer_ms: float,
        infer_ms: float,
        d2h_ms: float,
        audio_time: float,
    ) -> None:
        """Record frame timing for profiling.

        Args:
            total_ms: Total frame time
            steer_ms: Steering + physics + spatial + destinations
            infer_ms: GPU inference time (launch + cudaSync)
            d2h_ms: D2H memcpy time (after sync)
            audio_time: Audio playback time for this frame
        """
        timing = FrameTiming(
            total_ms=total_ms,
            steer_ms=steer_ms,
            infer_ms=infer_ms,
            d2h_ms=d2h_ms,
            encode_ms=self._last_encode_ms,
            audio_time=audio_time,
        )
        self._timing_history.append(timing)
        self._calibration_count += 1

        # Update measured FPS via EMA after calibration phase
        if self._calibration_count == self.CALIBRATION_THRESHOLD:
            # Initial calibration: use rolling average
            n = len(self._timing_history)
            avg_total = sum(t.total_ms for t in self._timing_history) / n
            if avg_total > 0:
                self._measured_fps = 1000.0 / avg_total
        elif self._calibration_count > self.CALIBRATION_THRESHOLD and total_ms > 0:
            # Ongoing: EMA update (0.85/0.15 smooths spikes, converges in ~10 frames)
            instantaneous_fps = 1000.0 / total_ms
            self._measured_fps = 0.85 * self._measured_fps + 0.15 * instantaneous_fps

    def get_timing_summary(self) -> dict:
        """Get average timing breakdown across recent frames.

        Returns:
            Dict with avg_total_ms, avg_steer_ms, avg_infer_ms, avg_d2h_ms, avg_encode_ms, fps
        """
        if len(self._timing_history) < 5:
            return {}

        n = len(self._timing_history)
        avg_total = sum(t.total_ms for t in self._timing_history) / n
        avg_steer = sum(t.steer_ms for t in self._timing_history) / n
        avg_infer = sum(t.infer_ms for t in self._timing_history) / n
        avg_d2h = sum(t.d2h_ms for t in self._timing_history) / n
        avg_encode = sum(t.encode_ms for t in self._timing_history) / n
        fps = 1000.0 / avg_total if avg_total > 0 else 0

        return {
            "avg_total_ms": avg_total,
            "avg_steer_ms": avg_steer,
            "avg_infer_ms": avg_infer,
            "avg_d2h_ms": avg_d2h,
            "avg_encode_ms": avg_encode,
            "fps": fps,
        }

    def get_perf_snapshot(self) -> dict:
        """Return a snapshot of encoder and timing stats for telemetry."""
        summary = self.get_timing_summary()
        return {
            "encode_ms": summary.get("avg_encode_ms", 0.0),
            "avg_total_ms": summary.get("avg_total_ms", 0.0),
            "avg_steer_ms": summary.get("avg_steer_ms", 0.0),
            "avg_infer_ms": summary.get("avg_infer_ms", 0.0),
            "avg_d2h_ms": summary.get("avg_d2h_ms", 0.0),
            "pending_age_ms": self.pending_age_ms(),
            "encode_busy": self.is_encode_busy(),
        }

    def log_timing_if_due(self) -> None:
        """Log timing summary periodically (~every 3 seconds).

        Uses perf_counter for throttling (not audio_time) to avoid
        burst logging when multiple frames share the same time bucket.
        """
        now = time.perf_counter()
        if now - self._last_profile_log < 3.0:
            return
        self._last_profile_log = now

        summary = self.get_timing_summary()
        if not summary:
            return

        logger.info(
            f"[FRAME] {summary['fps']:.1f} FPS | "
            f"steer={summary['avg_steer_ms']:.1f} "
            f"infer={summary['avg_infer_ms']:.1f} "
            f"d2h={summary['avg_d2h_ms']:.1f} "
            f"enc={summary['avg_encode_ms']:.1f} "
            f"total={summary['avg_total_ms']:.1f}ms"
        )

    @property
    def measured_fps(self) -> float:
        """Current EMA-smoothed FPS (bootstrap 45 until calibrated)."""
        return self._measured_fps

    @property
    def measured_interval_s(self) -> float:
        """Inverse of measured_fps (seconds per frame)."""
        return 1.0 / self._measured_fps

    @property
    def last_encode_ms(self) -> float:
        """Get encode time from previous frame."""
        return self._last_encode_ms

    async def cleanup(self) -> None:
        """Wait for pending encode to complete before shutdown."""
        if self._pending_task is None:
            return

        try:
            await asyncio.wait_for(self._pending_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        finally:
            self._pending_task = None

    def clear_history(self) -> None:
        """Clear timing history and reset FPS measurement (on session end)."""
        self._timing_history.clear()
        self._measured_fps = self.BOOTSTRAP_FPS
        self._calibration_count = 0
