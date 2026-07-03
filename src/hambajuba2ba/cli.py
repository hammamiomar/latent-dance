"""CLI entry point — `uv run hambajuba` starts the server."""

import os
import sys

import uvicorn

from hambajuba2ba.config import load_from_env


def serve() -> None:
    # `app/` is a repo directory, not an installed package — run from the
    # repo root. The uvicorn CLI puts cwd on sys.path; console scripts
    # don't, so do it here. (Real packaging is the desktop-distributable
    # phase's problem.)
    sys.path.insert(0, os.getcwd())
    config = load_from_env()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )
