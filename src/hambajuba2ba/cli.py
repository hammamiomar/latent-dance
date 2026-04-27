"""CLI entry point — `uv run hambajuba` starts the server."""

import uvicorn

from hambajuba2ba.config import PipelineConfig


def serve() -> None:
    config = PipelineConfig()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )
