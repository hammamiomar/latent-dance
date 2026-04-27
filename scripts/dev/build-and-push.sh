#!/bin/bash
# build-and-push.sh — Build and push the public Docker image.
#
# Usage:
#   ./scripts/dev/build-and-push.sh [docker build flags]
#
# One-time setup on the build host:
#   git clone https://github.com/hammamiomar/latent-dance.git && cd latent-dance
#   docker login ghcr.io -u <github-username> --password-stdin

set -euo pipefail

REGISTRY="ghcr.io"
IMAGE="hammamiomar/latent-dance"
SHA=$(git rev-parse --short HEAD)

if ! command -v docker &>/dev/null; then
    echo "Error: docker not found" >&2
    exit 1
fi

if [ ! -f pyproject.toml ]; then
    echo "Error: run from repo root (where pyproject.toml lives)" >&2
    exit 1
fi

# Public builds do not require Git LFS. SAE weights download at runtime from Hugging Face.

echo "Building ${REGISTRY}/${IMAGE}:${SHA}..."
docker build \
    --platform linux/amd64 \
    -t "${REGISTRY}/${IMAGE}:latest" \
    -t "${REGISTRY}/${IMAGE}:${SHA}" \
    "$@" \
    .

echo "Pushing to GHCR..."
docker push "${REGISTRY}/${IMAGE}:latest"
docker push "${REGISTRY}/${IMAGE}:${SHA}"

echo "Done! Pushed:"
echo "  ${REGISTRY}/${IMAGE}:latest"
echo "  ${REGISTRY}/${IMAGE}:${SHA}"
