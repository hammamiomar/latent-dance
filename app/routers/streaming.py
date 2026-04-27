"""WebSocket streaming endpoint for generation.

Handles real-time image generation driven by audio stem features.
Backend-agnostic — the mode parameter selects the strategy.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.caching import CacheManager
from app.dependencies import (
    get_audio_cache_ws,
    get_config_ws,
    get_pipeline_ws,
    get_gpu_lock_ws,
    get_cpu_executor_ws,
)
from app.strategies import create_strategy
from app.websocket_manager import WebSocketManager
from hambajuba2ba.config import PipelineConfig

logger = logging.getLogger("uvicorn")
router = APIRouter()


@router.websocket("/ws/stream/{mode}")
async def unified_stream(
    websocket: WebSocket,
    mode: str,
    pipeline=Depends(get_pipeline_ws),
    config: PipelineConfig = Depends(get_config_ws),
    audio_cache: CacheManager = Depends(get_audio_cache_ws),
    gpu_lock: asyncio.Lock = Depends(get_gpu_lock_ws),
    cpu_executor: ThreadPoolExecutor = Depends(get_cpu_executor_ws),
):
    """WebSocket endpoint for real-time SAE-steered generation.

    Connect to /ws/stream/sae_steering and send:
    {
        "action": "start_sae_steering",
        "audio_id": "uuid-from-upload",
        "prompt": "portrait of a cyberpunk woman",
        "seed": 42
    }

    You'll receive:
    - Binary frames (JPEG images)
    - JSON telemetry (stem activity levels)

    Args:
        websocket: WebSocket connection
        mode: Generation mode (only "sae_steering" supported)
        pipeline: SAE-steerable diffusion pipeline
        config: Pipeline configuration
        audio_cache: Audio feature cache
    """
    await websocket.accept()
    logger.info(f"Client connected to {mode} mode")

    try:
        # Create strategy for this mode (pass resolved dependencies explicitly)
        try:
            strategy = create_strategy(
                mode, pipeline, config, websocket, audio_cache,
                gpu_lock=gpu_lock, cpu_executor=cpu_executor,
            )
        except ValueError as e:
            logger.error(f"Invalid mode: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
            return

        # Create and run WebSocket manager
        manager = WebSocketManager(strategy, websocket, config)
        await manager.run()

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {mode} mode")
    except Exception as e:
        logger.error(f"Error in {mode} mode: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaned up {mode} mode connection")
