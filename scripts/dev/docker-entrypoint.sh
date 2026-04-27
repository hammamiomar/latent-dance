#!/bin/bash
# Public container entrypoint for latent-dance.
set -euo pipefail

echo "=== latent-dance public runtime starting ==="

mkdir -p /workspace/.cache/huggingface/hub
mkdir -p /workspace/.cache/huggingface/datasets
mkdir -p /workspace/.cache/torch/hub/checkpoints
mkdir -p /workspace/.cache/torchinductor
mkdir -p /workspace/.cache/audio-separator-models
mkdir -p /workspace/.cache/hambajuba2ba/artifacts

case "${MODE:-api}" in
    api)
        echo "Starting API server on :8000"
        cd /app
        exec uv run --extra audio-gpu uvicorn app.main:app --host 0.0.0.0 --port 8000
        ;;
    shell)
        echo "Shell mode requested"
        cd /app
        exec /bin/bash
        ;;
    *)
        echo "Unknown MODE='${MODE}'. Supported: api, shell" >&2
        exit 2
        ;;
esac
