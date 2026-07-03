"""Strategy factory.

Creates the generation strategy for the requested mode via the backend
registry (app/backends.py). New backends register there — this factory
has no per-mode branching.
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
    """Create the strategy for a registered backend mode.

    Args:
        mode: Backend mode name (must be registered in app.backends)
        pipeline: Generation pipeline (the active backend's)
        config: Pipeline configuration
        websocket: WebSocket connection
        audio_cache: Audio feature cache manager
        gpu_lock: asyncio.Lock for serializing GPU operations
        cpu_executor: ThreadPoolExecutor for parallel CPU work
    """
    # Local import: app.backends imports strategy classes at registration
    # time, so a module-level import here would be a cycle.
    from app.backends import get_backend

    spec = get_backend(mode)
    return spec.strategy_class(
        pipeline, config, websocket, audio_cache,
        gpu_lock=gpu_lock, cpu_executor=cpu_executor,
    )
