"""Unified WebSocket manager with pluggable generation strategies.

This module eliminates code duplication across different generation modes
by providing a single producer-consumer implementation that works with
any GenerationStrategy.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.generation import FrameItem
from app.schemas import ClientMessage, StartGeneration, Stop, StopGeneration, ErrorMessage
from app.strategies.base import GenerationStrategy
from hambajuba2ba.config import PipelineConfig

logger = logging.getLogger("uvicorn")

# Create TypeAdapter for validating discriminated union messages
ClientMessageAdapter = TypeAdapter(ClientMessage)


class RuthlessConsumer:
    """Low-latency delivery with strict frame dropping.

    This consumer enforces tight synchronization with frame due times,
    ruthlessly dropping frames that are too late. This maintains
    audio-visual sync even when generation can't keep up.

    Key features:
    - Drops frames beyond late_tolerance
    - Maintains tight sync with due_ts
    - Allows graceful FPS degradation under load

    Use for: Audio mode
    """

    def __init__(
        self,
        frames_q: asyncio.Queue,
        ctrl_q: asyncio.Queue,
        websocket: WebSocket,
        stop_event: asyncio.Event,
        max_queue_frames: int,
        drop_to_frames: int,
        queue_timeout: float = 0.2,
        late_tolerance: float = 0.05,
    ):
        self.frames_q = frames_q
        self.ctrl_q = ctrl_q
        self.websocket = websocket
        self.stop_event = stop_event
        self.max_queue_frames = max_queue_frames
        self.drop_to_frames = drop_to_frames
        self.queue_timeout = queue_timeout
        self.late_tolerance = late_tolerance

        # SLO delivery metrics
        self._latencies: deque = deque(maxlen=100)
        self._delivery_intervals: deque = deque(maxlen=100)
        self._prev_sent_at: float = 0.0
        self._frames_sent: int = 0
        self._frames_dropped_late: int = 0
        self._frames_dropped_overflow: int = 0

    async def run(self):
        """Run the ruthless consumer loop."""
        while not self.stop_event.is_set():
            # Drain telemetry first (non-blocking)
            try:
                while True:
                    ctrl = self.ctrl_q.get_nowait()
                    if ctrl.kind == "json":
                        await self.websocket.send_json(ctrl.payload)
            except asyncio.QueueEmpty:
                pass

            # Get next frame
            try:
                item = await asyncio.wait_for(self.frames_q.get(), timeout=self.queue_timeout)
            except asyncio.TimeoutError:
                continue

            if item.kind != "frame":
                continue

            # Drop old frames if queue is too backed up
            if self.frames_q.qsize() > self.max_queue_frames:
                drops = max(0, self.frames_q.qsize() - self.drop_to_frames)
                for _ in range(drops):
                    try:
                        _ = self.frames_q.get_nowait()
                        self._frames_dropped_overflow += 1
                    except asyncio.QueueEmpty:
                        break

            # Enforce strict sync: drop frames that are too late
            if item.due_ts is not None:
                now = time.perf_counter()
                lateness = now - item.due_ts

                if lateness > self.late_tolerance:
                    self._frames_dropped_late += 1
                    continue

                # Wait if ahead of schedule
                if item.due_ts > now:
                    await asyncio.sleep(item.due_ts - now)

            # Record delivery metrics before sending
            sent_at = time.perf_counter()
            if item.produced_at is not None:
                self._latencies.append(sent_at - item.produced_at)
            if self._prev_sent_at > 0:
                self._delivery_intervals.append(sent_at - self._prev_sent_at)
            self._prev_sent_at = sent_at
            self._frames_sent += 1

            # Send frame
            await self.websocket.send_bytes(item.payload)


    def get_delivery_stats(self) -> dict:
        """Compute delivery SLO metrics from rolling windows.

        Returns percentiles via sorted-array (~100 floats, <1us).
        """
        stats: dict = {}

        # Latency percentiles (production → send)
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            n = len(sorted_lat)
            stats["delivery_p50_ms"] = sorted_lat[n // 2] * 1000
            stats["delivery_p95_ms"] = sorted_lat[min(n - 1, int(n * 0.95))] * 1000

        # Jitter (variation in delivery intervals)
        if len(self._delivery_intervals) >= 2:
            intervals = list(self._delivery_intervals)
            mean_interval = sum(intervals) / len(intervals)
            jitters = [abs(iv - mean_interval) for iv in intervals]
            stats["jitter_mean_ms"] = (sum(jitters) / len(jitters)) * 1000
            sorted_jitter = sorted(jitters)
            n = len(sorted_jitter)
            stats["jitter_p95_ms"] = sorted_jitter[min(n - 1, int(n * 0.95))] * 1000

        # Drop rate
        total = self._frames_sent + self._frames_dropped_late + self._frames_dropped_overflow
        if total > 0:
            stats["drop_rate"] = (self._frames_dropped_late + self._frames_dropped_overflow) / total
        else:
            stats["drop_rate"] = 0.0

        return stats


class WebSocketManager:
    """Unified WebSocket manager with pluggable generation strategy.

    This class provides a complete producer-consumer implementation that
    works with any GenerationStrategy. It handles:
    - Message validation and routing
    - Producer task (calls strategy.next_frame_batch())
    - Consumer task (RuthlessConsumer for strict audio sync)
    - Lifecycle management (setup, cleanup, stop)

    Usage:
        strategy = create_strategy(mode, pipeline, config, websocket, cache, ...)
        manager = WebSocketManager(strategy, websocket, config)
        await manager.run()
    """

    def __init__(
        self,
        strategy: GenerationStrategy,
        websocket: WebSocket,
        config: PipelineConfig,
    ):
        """Initialize WebSocket manager.

        Args:
            strategy: Generation strategy from create_strategy()
            websocket: WebSocket connection
            config: Pipeline configuration
        """
        self.strategy = strategy
        self.websocket = websocket
        self.config = config

        # Compute pacing parameters from streaming config
        self.fps = config.streaming.fps
        self.frame_interval = 1.0 / max(1.0, self.fps)
        self.max_queue_frames = int(config.streaming.max_queue_frames_multiplier * self.fps)
        self.drop_to_frames = int(config.streaming.drop_to_frames_multiplier * self.fps)
        self.queue_timeout = config.streaming.queue_timeout
        self.late_tolerance = config.streaming.late_tolerance

        # Shared resources
        self.stop_event = asyncio.Event()
        self.frames_q: asyncio.Queue[FrameItem] = asyncio.Queue(
            maxsize=self.max_queue_frames * 2
        )
        self.ctrl_q: asyncio.Queue[FrameItem] = asyncio.Queue(
            maxsize=config.streaming.control_queue_size
        )

        # Task handles
        self.producer_task: Optional[asyncio.Task] = None
        self.consumer_task: Optional[asyncio.Task] = None
        self.consumer: Optional[RuthlessConsumer] = None

        # Batch timing telemetry
        self.batch_times = deque(maxlen=30)  # Rolling window for FPS calculation
        self._fps_log_counter = 0  # Counter for throttled logging
        self._last_perf_emit: float = 0.0

    async def producer(self):
        """Producer task: generates frames via strategy.

        This task repeatedly calls strategy.next_frame_batch() and
        enqueues the resulting frames and telemetry.
        """
        while not self.stop_event.is_set():
            try:
                t0 = time.perf_counter()

                # Generate batch via strategy
                items = await self.strategy.next_frame_batch()

                # Enqueue frames
                frame_count = 0
                for item in items:
                    if item.kind == "frame":
                        # Drop oldest if queue is full
                        while self.frames_q.qsize() >= self.max_queue_frames:
                            try:
                                _ = self.frames_q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        await self.frames_q.put(item)
                        frame_count += 1
                    elif item.kind == "json":
                        # Telemetry (non-blocking)
                        try:
                            self.ctrl_q.put_nowait(item)
                        except asyncio.QueueFull:
                            pass

                # Throughput logging (throttled to every 150 frames ~3s at 50 FPS)
                dt = time.perf_counter() - t0
                self.batch_times.append(dt)
                self._fps_log_counter += 1

                if self._fps_log_counter >= 150 and len(self.batch_times) >= 10:
                    avg = sum(self.batch_times) / len(self.batch_times)
                    gen_fps = 1.0 / max(1e-9, avg)
                    logger.info(
                        f"Generation: {gen_fps:.1f} FPS, queued={self.frames_q.qsize()}"
                    )
                    self._fps_log_counter = 0  # Reset counter

                # Perf telemetry (~1Hz)
                now = time.perf_counter()
                if now - self._last_perf_emit >= 1.0 and len(self.batch_times) >= 5:
                    avg = sum(self.batch_times) / len(self.batch_times)
                    gen_fps = 1.0 / max(1e-9, avg)
                    # Strategy perf state (includes measured_fps + lookahead_ms)
                    try:
                        snapshot = self.strategy.get_perf_snapshot()
                    except Exception:
                        snapshot = {}
                    # Merge delivery SLO metrics from consumer
                    delivery_stats = {}
                    if self.consumer is not None:
                        try:
                            delivery_stats = self.consumer.get_delivery_stats()
                        except Exception:
                            pass

                    perf_payload = {
                        "type": "perf_stats",
                        "gen_fps": gen_fps,
                        "queue_depth": self.frames_q.qsize(),
                        **snapshot,
                        **delivery_stats,
                    }
                    try:
                        self.ctrl_q.put_nowait(FrameItem(kind="json", payload=perf_payload))
                    except asyncio.QueueFull:
                        pass
                    self._last_perf_emit = now

                # Pacing: use measured FPS so sleep is meaningful.
                # config.streaming.fps (60) is the ceiling; measured FPS (~50) is reality.
                target_dt = self.strategy.measured_interval_s * max(1, frame_count)
                sleep_for = target_dt - dt
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0)

                # Backpressure removed: RuthlessConsumer handles frame dropping gracefully.
                # The previous threshold (50% queue) triggered on every frame at high FPS,
                # halving effective throughput. Let the consumer drop stale frames instead.

            except Exception as e:
                logger.error(f"Producer error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def run(self):
        """Main WebSocket control loop.

        Handles incoming messages and manages producer/consumer lifecycle.
        """
        try:
            while True:
                # Receive and validate message
                try:
                    msg_data = await self.websocket.receive_json()
                    message = ClientMessageAdapter.validate_python(msg_data)
                except ValidationError as e:
                    logger.warning(f"Invalid message: {e}")
                    await self.websocket.send_json(
                        ErrorMessage(message=f"Invalid message: {e}").model_dump()
                    )
                    continue
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    break

                # Handle message
                if isinstance(message, StartGeneration):
                    logger.info("Starting generation")

                    # Setup strategy
                    try:
                        await self.strategy.setup(message)
                    except Exception as e:
                        logger.error(f"Setup failed: {e}", exc_info=True)
                        await self.websocket.send_json(
                            ErrorMessage(message=str(e)).model_dump()
                        )
                        continue

                    # Clear stop event and start tasks
                    self.stop_event.clear()

                    if not self.producer_task or self.producer_task.done():
                        self.producer_task = asyncio.create_task(self.producer())

                    if not self.consumer_task or self.consumer_task.done():
                        self.consumer = RuthlessConsumer(
                            self.frames_q,
                            self.ctrl_q,
                            self.websocket,
                            self.stop_event,
                            self.max_queue_frames,
                            self.drop_to_frames,
                            self.queue_timeout,
                            self.late_tolerance,
                        )
                        self.consumer_task = asyncio.create_task(self.consumer.run())

                elif isinstance(message, Stop):
                    logger.info("Stopping generation and disconnecting")
                    self.stop_event.set()
                    break

                elif isinstance(message, StopGeneration):
                    # Stop generation but keep session alive for new songs
                    logger.info("Stopping generation (session stays alive)")
                    self.stop_event.set()

                    # Cancel producer/consumer tasks
                    if self.producer_task and not self.producer_task.done():
                        self.producer_task.cancel()
                        await asyncio.gather(self.producer_task, return_exceptions=True)
                        self.producer_task = None
                    if self.consumer_task and not self.consumer_task.done():
                        self.consumer_task.cancel()
                        await asyncio.gather(self.consumer_task, return_exceptions=True)
                        self.consumer_task = None

                    # Cleanup strategy resources
                    await self.strategy.cleanup()
                    # DON'T BREAK - session stays open for new generation

                else:
                    # Mode-specific message, forward to strategy
                    try:
                        response = await self.strategy.handle_message(message)
                        if response:
                            await self.websocket.send_json(response)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}", exc_info=True)
                        await self.websocket.send_json(
                            ErrorMessage(message=str(e)).model_dump()
                        )

        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)

        finally:
            # Cleanup
            logger.info("Cleaning up WebSocket manager")
            self.stop_event.set()

            if self.producer_task:
                self.producer_task.cancel()
                await asyncio.gather(self.producer_task, return_exceptions=True)

            if self.consumer_task:
                self.consumer_task.cancel()
                await asyncio.gather(self.consumer_task, return_exceptions=True)

            await self.strategy.cleanup()
