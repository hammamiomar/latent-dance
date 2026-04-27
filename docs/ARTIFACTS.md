# Public Artifacts

The public GitHub repository intentionally does not track heavyweight SAE checkpoint files.

## Hugging Face Artifacts

Upstream runtime SAE weights:

```text
surokpro2/sdxl-saes
```

latent-dance uses the `k10_hidden5120` checkpoints for these four upstream
directories:

```text
unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
```

Public label/activation dataset:

```text
hammamiomar/sdxl-turbo-sae-labels
```

The runtime also supports short-name local layouts:

```text
down.2.1/final/config.json
down.2.1/final/state_dict.pth
down.2.1/final/mean.pt
down.2.1/final/std.pt
mid.0/final/...
up.0.0/final/...
up.0.1/final/...
```

or:

```text
down.2.1/config.json
down.2.1/state_dict.pth
mid.0/...
up.0.0/...
up.0.1/...
```

The app first checks `./data/sdxl/sae_weights`. If files are absent and automatic download is enabled, it downloads the needed upstream checkpoint directories from Hugging Face into `HAMBA_ARTIFACT_DIR`.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HAMBA_ARTIFACT_REPO` | `surokpro2/sdxl-saes` | Hugging Face repo id. |
| `HAMBA_ARTIFACT_REPO_TYPE` | `model` | Hugging Face repo type. |
| `HAMBA_ARTIFACT_WEIGHTS_SUBDIR` | empty | Optional subdirectory containing SAE checkpoints. |
| `HAMBA_ARTIFACT_DIR` | `~/.cache/hambajuba2ba/artifacts` | Local artifact cache. |
| `HAMBAJUBA_SAE_WEIGHTS_DIR` | `./data/sdxl/sae_weights` | Local preferred weights directory. |
| `HAMBAJUBA_SAE_AUTO_DOWNLOAD` | `true` | Disable to require local files. |

## Manual Check

```bash
uv run python scripts/artifacts/download_sae_weights.py --check
```

To materialize the downloaded snapshot into a local directory for tooling that needs real files:

```bash
uv run python scripts/artifacts/download_sae_weights.py --materialize data/sdxl/sae_weights
```

The optional labeling reproduction pipeline can also download these artifacts
inside Modal. See `docs/LABELING_PIPELINE.md`.
