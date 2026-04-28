# Stage 1: frontend build
FROM oven/bun:1 AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/bun.lock ./
RUN bun install --frozen-lockfile
COPY frontend/ ./
RUN bun run build

# Stage 2: public runtime image. Model and SAE artifacts download to /workspace cache.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TORCH_HOME=/workspace/.cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
    AUDIO_SEPARATOR_MODEL_DIR=/workspace/.cache/audio-separator-models \
    HAMBA_ARTIFACT_DIR=/workspace/.cache/hambajuba2ba/artifacts \
    HAMBA_ARTIFACT_REPO=surokpro2/sdxl-saes \
    TRANSFORMERS_NO_TF=1 \
    USE_TF=0 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MODE=api \
    PATH="/root/.cargo/bin:/root/.local/bin:/app/scripts:${PATH}" \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates ffmpeg libturbojpeg0-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra audio-gpu --no-dev --no-install-project

COPY README.md LICENSE NOTICE ./
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
RUN uv sync --frozen --extra audio-gpu --no-dev

COPY --from=frontend-builder /frontend/dist /app/frontend/dist

EXPOSE 8000

COPY scripts/dev/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
