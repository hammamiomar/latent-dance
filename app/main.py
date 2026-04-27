"""FastAPI application - SDXL-Turbo with SAE steering.

This is the main entry point for the latent-dance backend.
Pipeline initialization happens in the lifespan handler.
"""

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.caching import CacheManager
from app.routers import audio, streaming
from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.generation.pipeline import SAESteerablePipeline

# Configure logging to show all app logs.
# Force-configure the root logger — logging.basicConfig() is a no-op when
# Uvicorn CLI has already attached handlers. We need to replace them.
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
))
_root.handlers = [_handler]

# Reduce uvicorn access log noise (e.g., polling endpoints)
class _UvicornAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Check the formatted message (robust across uvicorn versions)
            msg = record.getMessage()
            if "/api/audio/status/" in msg:
                return False
        except Exception:
            return True
        return True


logging.getLogger("uvicorn.access").addFilter(_UvicornAccessFilter())

# Set our app loggers to INFO
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("hambajuba2ba").setLevel(logging.INFO)

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load SDXL-Turbo + SAE on startup, cleanup on shutdown.

    This runs once when the server starts. All expensive model loading
    happens here so requests don't pay the cost.
    """
    # Config auto-detects device and dtype (see config.py __post_init__)
    config = PipelineConfig()

    logger.info("=" * 60)
    logger.info("STARTUP: Loading SDXL-Turbo with SAE steering")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Dtype: {config.dtype}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info("=" * 60)
    logger.info(f"Config: {config.width}x{config.height}")

    logger.info("Loading SDXL-Turbo pipeline...")
    pipeline = SAESteerablePipeline(config)
    pipeline.load()  # Downloads models, compiles CUDA graph, warmup
    logger.info("Pipeline ready!")

    # Execution resources (injected into strategies via DI, not globals)
    gpu_lock = asyncio.Lock()
    cpu_executor = ThreadPoolExecutor(max_workers=config.cpu_workers)

    # Audio cache for stem features
    audio_cache = CacheManager(default_ttl=config.audio.cache_ttl_seconds)

    # Store for dependency injection
    app.state.pipeline = pipeline
    app.state.config = config
    app.state.gpu_lock = gpu_lock
    app.state.cpu_executor = cpu_executor
    app.state.audio_cache = audio_cache

    logger.info("Ready! SAE steering enabled.")
    yield

    # Cleanup
    logger.info("Shutting down...")
    pipeline.cleanup()
    cpu_executor.shutdown(wait=True)
    logger.info("Cleanup complete.")


app = FastAPI(
    title="latent-dance",
    description="Audio-reactive visuals via SAE steering",
    lifespan=lifespan,
)

# Only the routers we need
app.include_router(audio.router)  # /api/audio/upload
app.include_router(streaming.router)  # /ws/stream/{mode}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "mode": "sae_steering"}


# ─── Static frontend serving (production) ────────────────────────────────
# In production, the built SPA lives at frontend/dist/ (copied into image).
# In development, this dir doesn't exist — Vite dev server handles it instead.
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """SPA catch-all: serve static files or fall back to index.html."""
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn

    # Load config for server settings
    config = PipelineConfig()
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_config=None,  # Preserve our logging.basicConfig — uvicorn.run() overwrites it otherwise
    )
