# Deployment

## Public Docker Image

The public GPU runtime is published to GitHub Container Registry:

```bash
ghcr.io/hammamiomar/latent-dance:latest
```

It is built by `.github/workflows/docker-build-push.yml` on manual dispatch or
tagged releases. If the package has not been published yet, build the same image
locally with the command below.

Build locally:

```bash
docker build --platform linux/amd64 -t ghcr.io/hammamiomar/latent-dance:latest .
```

Run:

```bash
docker run --gpus all -p 8000:8000 -v latent-dance-cache:/workspace ghcr.io/hammamiomar/latent-dance:latest
```

The host needs an NVIDIA GPU, recent NVIDIA drivers, Docker, and NVIDIA
Container Toolkit. The Linux dependency lock selects PyTorch CUDA 12.8 wheels,
which are the Blackwell-tested path used for the public demo target. The image
is not intentionally Blackwell-only, but real-time performance is only claimed
for the tested high-end CUDA setup.

The image defaults to `MODE=api` and exposes only HTTP port `8000`. It does not
start a remote shell service. Models and SAE artifacts download into
`/workspace/.cache`.

## First Boot Downloads

- SDXL-Turbo and TinyVAE via Hugging Face cache.
- Demucs/audio-separator models via their configured cache.
- SAE weights from upstream `surokpro2/sdxl-saes` checkpoint directories.

## Modes

| `MODE` | Behavior |
|---|---|
| `api` | Starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`. |
| `shell` | Opens an interactive shell inside `/app`. |

Private GPU sync workflows are intentionally not part of the public Docker image.
