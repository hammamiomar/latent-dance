"""Dependency injection providers for FastAPI routes.

Centralizes dependency providers for HTTP and WebSocket endpoints.
All shared resources are provided through FastAPI's Depends() system.

Note: Separate functions for Request vs WebSocket are needed because
FastAPI's dependency resolution doesn't support Union types.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request, WebSocket

from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.generation.pipeline import SAESteerablePipeline

from .caching import CacheManager


# HTTP endpoint dependencies (use Request)
def get_config(request: Request) -> PipelineConfig:
    """Dependency provider for pipeline configuration (HTTP endpoints)."""
    return request.app.state.config


def get_audio_cache(request: Request) -> CacheManager:
    """Dependency provider for audio feature cache (HTTP endpoints)."""
    return request.app.state.audio_cache


# WebSocket endpoint dependencies (use WebSocket)
def get_pipeline_ws(websocket: WebSocket) -> SAESteerablePipeline:
    """Dependency provider for the diffusion pipeline (WebSocket endpoints)."""
    return websocket.app.state.pipeline


def get_config_ws(websocket: WebSocket) -> PipelineConfig:
    """Dependency provider for pipeline configuration (WebSocket endpoints)."""
    return websocket.app.state.config


def get_audio_cache_ws(websocket: WebSocket) -> CacheManager:
    """Dependency provider for audio feature cache (WebSocket endpoints)."""
    return websocket.app.state.audio_cache


def get_gpu_lock_ws(websocket: WebSocket) -> asyncio.Lock:
    """Dependency provider for GPU lock (WebSocket endpoints)."""
    return websocket.app.state.gpu_lock


def get_cpu_executor_ws(websocket: WebSocket) -> ThreadPoolExecutor:
    """Dependency provider for CPU executor (WebSocket endpoints)."""
    return websocket.app.state.cpu_executor
