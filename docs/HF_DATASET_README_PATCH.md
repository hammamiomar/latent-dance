# Hugging Face Dataset README Corrections

These wording changes were applied to `hammamiomar/sdxl-turbo-sae-labels` on
2026-04-27:

https://huggingface.co/datasets/hammamiomar/sdxl-turbo-sae-labels/commit/0355ef0982a4c8598ce9dabab96caa64c7df1ce9

The upstream SAE weight repository link was added in:

https://huggingface.co/datasets/hammamiomar/sdxl-turbo-sae-labels/commit/a4e2c8f6e05ec0bd74e2aca84d8388558c03d2cf

The public repository link and project name were updated to latent-dance in:

https://huggingface.co/datasets/hammamiomar/sdxl-turbo-sae-labels/commit/8fd9cac23ee7c4b69a11f3c595d65473ce4c0b97

## Required Attribution

Added near the top:

> Important attribution: the SDXL-Turbo sparse autoencoders/checkpoints used here were trained and released by Surkov et al. / EPFL through `sdxl-unbox`. This dataset does not claim authorship of the SAE training. It contributes labels, activation logs, VLM consensus annotations, factor groupings, and analysis for those upstream SAE features, built for latent-dance.

## Rename Section

Changed `## SAE Architecture` to:

```markdown
## Upstream SAE Architecture
```

Added immediately below it:

```markdown
The SAE architecture and checkpoints come from Surkov et al. / EPFL's `sdxl-unbox` work. latent-dance uses those upstream SAEs and provides the feature labeling and activation-analysis dataset below.
```

## Fix Acknowledgments

Replaced:

```markdown
- SAE training approach inspired by [sdxl-unbox](https://arxiv.org/abs/2410.22366) (Surkov et al., NeurIPS 2025)
```

with:

```markdown
- SDXL-Turbo SAE architecture and checkpoints by [sdxl-unbox](https://github.com/surkovv/sdxl-unbox) / [sdxl-unbox.epfl.ch](https://sdxl-unbox.epfl.ch/) (Surkov et al., EPFL). These SAEs were not trained by latent-dance.
```

## Cleanup

Removed `.DS_Store` from the dataset repository.

## Upstream Weight Link

Added the upstream Hugging Face checkpoint repository link:

```markdown
https://huggingface.co/surokpro2/sdxl-saes/tree/main
```

## Public Project Link

Updated the dataset card to point at the public repository:

```markdown
https://github.com/hammamiomar/latent-dance
```
