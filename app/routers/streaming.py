"""WebSocket streaming endpoint for generation.

Handles real-time image generation driven by audio stem features.
Backend-agnostic — the mode parameter selects the strategy.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.backends import BackendCapabilities
from app.caching import CacheManager
from app.dependencies import (
    get_audio_cache_ws,
    get_capabilities_ws,
    get_config_ws,
    get_pipeline_ws,
    get_gpu_lock_ws,
    get_cpu_executor_ws,
    get_server_mode_ws,
)
from app.schemas import CapabilitiesMessage, ErrorMessage
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
    capabilities: BackendCapabilities = Depends(get_capabilities_ws),
    server_mode: str = Depends(get_server_mode_ws),
):
    """WebSocket endpoint for real-time audio-reactive generation.

    Connect to /ws/stream/{mode} where mode matches the backend this
    process serves (HAMBA_MODE). The first message you receive is the
    backend's capability manifest; then send a start message, e.g.:
    {
        "action": "start_sae_steering",
        "audio_id": "uuid-from-upload",
        "prompt": "portrait of a cyberpunk woman",
        "seed": 42
    }

    You'll receive:
    - {"type": "capabilities", ...} once on connect
    - Binary frames (JPEG images)
    - JSON telemetry (stem activity levels)

    Args:
        websocket: WebSocket connection
        mode: Requested generation mode (must match the server's)
        pipeline: The active backend's pipeline
        config: Pipeline configuration
        audio_cache: Audio feature cache
        capabilities: Active backend capability manifest
        server_mode: Backend mode this process serves
    """
    await websocket.accept()
    logger.info(f"Client connected to {mode} mode")

    # One model per process: reject mode mismatches loudly instead of
    # letting a wrong-pipeline strategy fail mid-generation.
    if mode != server_mode:
        message = f"Server is running '{server_mode}', not '{mode}'"
        logger.error(message)
        await websocket.send_json(ErrorMessage(message=message).model_dump())
        await websocket.close()
        return

    # Capabilities hello — the frontend learns the backend's control-input
    # manifest before any generation starts.
    await websocket.send_json(
        CapabilitiesMessage(capabilities=capabilities.to_dict()).model_dump()
    )

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
