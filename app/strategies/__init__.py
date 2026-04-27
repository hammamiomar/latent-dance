"""Strategy factory.

Creates the appropriate generation strategy for the requested mode.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import WebSocket

from app.caching import CacheManager
from app.strategies.base import GenerationStrategy
from app.strategies.sae_steering_strategy import SAESteeringStrategy
from hambajuba2ba.config import PipelineConfig

__all__ = [
    "GenerationStrategy",
    "SAESteeringStrategy",
    "create_strategy",
]


def create_strategy(
    mode: str,
    pipeline: Any,
    config: PipelineConfig,
    websocket: WebSocket,
    audio_cache: CacheManager,
    *,
    gpu_lock: asyncio.Lock,
    cpu_executor: ThreadPoolExecutor,
) -> GenerationStrategy:
    """Create strategy for the requested mode.

    Args:
        mode: Mode name (only "sae_steering" supported currently)
        pipeline: Generation pipeline
        config: Pipeline configuration
        websocket: WebSocket connection
        audio_cache: Audio feature cache manager
        gpu_lock: asyncio.Lock for serializing GPU operations
        cpu_executor: ThreadPoolExecutor for parallel CPU work
    """
    if mode != "sae_steering":
        raise ValueError(
            f"Unknown mode: {mode}. Only 'sae_steering' is supported."
        )

    return SAESteeringStrategy(
        pipeline, config, websocket, audio_cache,
        gpu_lock=gpu_lock, cpu_executor=cpu_executor,
    )
