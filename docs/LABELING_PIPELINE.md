# SAE Labeling Pipeline

This repository includes the public method code used to produce and inspect the
SAE feature labels hosted on Hugging Face:

```text
hammamiomar/sdxl-turbo-sae-labels
```

The labeling code is for research reproducibility. It is not required to run the
latent-dance realtime app.

## Attribution Boundary

The sparse autoencoders were not trained by latent-dance. They come from
Surkov et al. / EPFL's `sdxl-unbox` work. latent-dance contributes the label
generation, validation, runtime steering integration, and public dataset built
on top of those upstream SAE checkpoints.

## Stages

1. `clustering.py`: groups decoder weights with UMAP/HDBSCAN to provide prior
   structure for labeling.
2. `stage1_generate.py`: optionally uses Modal to generate SDXL-Turbo images
   and capture SAE activations across the four supported UNet attention blocks.
3. `stage2_select.py`: ranks features and selects high-activation images/crops.
4. `stage3_annotate.py`: asks VLMs to annotate selected feature image sets.
5. `stage4_supplement.py`: adds zero-cost prompt/statistical signals.
6. `stage5_fuse.py`: fuses labels and embeddings into final feature metadata.
7. `stage6_validate.py`: runs ground-truth gates and detection checks.
8. `stage7_factors.py`: groups labeled features into user-facing factors.

## Optional Modal Reproduction

Modal is intentionally not part of the default install. To reproduce Stage 1:

```bash
uv sync --extra labeling-modal
modal setup
uv run python -m hambajuba2ba.labeling.stage1_generate --n-images 50000
```

By default, the Modal worker downloads public SAE weights from the Hugging Face
dataset into a persistent Modal cache volume. To upload local weights instead,
set:

```bash
export HAMBAJUBA_MODAL_SAE_WEIGHTS_ROOT=/path/to/sae_weights
```

The directory must contain:

```text
down.2.1/final/state_dict.pth
mid.0/final/state_dict.pth
up.0.0/final/state_dict.pth
up.0.1/final/state_dict.pth
```

Useful Modal environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HAMBAJUBA_MODAL_APP_NAME` | `hambajuba-sae-labeling` | Modal app name. |
| `HAMBAJUBA_MODAL_OUTPUT_VOLUME` | `hambajuba-sae-labeling-output` | Output volume for generated images and activations. |
| `HAMBAJUBA_MODAL_ARTIFACT_CACHE_VOLUME` | `hambajuba-sae-artifact-cache` | Cache volume for downloaded SAE weights. |
| `HAMBA_ARTIFACT_REPO` | `surokpro2/sdxl-saes` | Hugging Face repo containing upstream SAE weights. |
| `HAMBA_ARTIFACT_REPO_TYPE` | `model` | Hugging Face repo type for upstream SAE weights. |
| `HAMBA_ARTIFACT_WEIGHTS_SUBDIR` | empty | Optional subdirectory if using a mirrored checkpoint layout. |

Download generated results:

```bash
modal volume get hambajuba-sae-labeling-output /generated ./data/labeling/sae_images/generated/
modal volume get hambajuba-sae-labeling-output /activations.jsonl ./data/labeling/sae_images/
```

## Runtime Users

Runtime users should not run the labeling pipeline. Use the published Hugging
Face labels and the artifact downloader documented in `docs/ARTIFACTS.md`.
