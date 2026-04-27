"""Optional Modal deployment for Stage 1 labeling reproduction.

Runs SDXL-Turbo (1-step, uncompiled) on an A100 GPU. Hooks all 4 UNet
attention blocks to capture sparse SAE activations in a single forward
pass per image. Saves images as JPEG + activations as JSONL.

This is research/reproducibility infrastructure for recreating the public
Hugging Face label dataset. It is not required for normal hambajuba2ba runtime.

Setup (one-time):
    uv sync --extra labeling-modal
    modal setup

Run via local orchestrator:
    uv run python -m hambajuba2ba.labeling.stage1_generate [--n-images 50000]

Download results:
    modal volume get hambajuba-sae-labeling-output /generated ./data/labeling/sae_images/generated/
    modal volume get hambajuba-sae-labeling-output /activations.jsonl ./data/labeling/sae_images/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import torch

# ─── Modal infrastructure ────────────────────────────────────────

APP_NAME = os.getenv("HAMBAJUBA_MODAL_APP_NAME", "hambajuba-sae-labeling")
OUTPUT_VOLUME_NAME = os.getenv(
    "HAMBAJUBA_MODAL_OUTPUT_VOLUME",
    "hambajuba-sae-labeling-output",
)
ARTIFACT_CACHE_VOLUME_NAME = os.getenv(
    "HAMBAJUBA_MODAL_ARTIFACT_CACHE_VOLUME",
    "hambajuba-sae-artifact-cache",
)
ARTIFACT_REPO_ID = os.getenv(
    "HAMBA_ARTIFACT_REPO",
    "surokpro2/sdxl-saes",
)
ARTIFACT_REPO_TYPE = os.getenv("HAMBA_ARTIFACT_REPO_TYPE", "model")
ARTIFACT_WEIGHTS_SUBDIR = os.getenv("HAMBA_ARTIFACT_WEIGHTS_SUBDIR", "")
LOCAL_WEIGHTS_ROOT = os.getenv("HAMBAJUBA_MODAL_SAE_WEIGHTS_ROOT")

# include_source=False prevents Modal from auto-mounting the entire
# hambajuba2ba package (which would pull in generation/, audio/, etc.
# and their heavy dependencies like turbojpeg). We explicitly mount
# only the labeling subpackage via add_local_dir below.
app = modal.App(APP_NAME, include_source=False)

# UNet attention block paths for hook registration
_UNET_BLOCK_PATHS: dict[str, str] = {
    "down.2.1": "down_blocks.2.attentions.1",
    "mid.0": "mid_block.attentions.0",
    "up.0.0": "up_blocks.0.attentions.0",
    "up.0.1": "up_blocks.0.attentions.1",
}

_UPSTREAM_SDXL_SAE_DIRS: dict[str, str] = {
    "down.2.1": "unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001",
    "mid.0": "unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001",
    "up.0.0": "unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001",
    "up.0.1": "unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001",
}

# Container image: PyTorch + diffusers + SAE weights
labeling_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.0",
        "diffusers==0.35.2",
        "transformers==4.57.1",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "datasets",
        "numpy",
        "pillow",
    )
    .run_commands("mkdir -p /root/hambajuba2ba && touch /root/hambajuba2ba/__init__.py")
)

# Mount only the labeling subpackage (avoids turbojpeg / generation imports).
_LABELING_SRC = Path(__file__).resolve().parent
labeling_image = labeling_image.add_local_dir(
    str(_LABELING_SRC),
    remote_path="/root/hambajuba2ba/labeling",
)

# Optional fast path: upload local weights into the Modal image. By default the
# worker downloads public artifacts from Hugging Face instead.
if LOCAL_WEIGHTS_ROOT:
    _local_weights_root = Path(LOCAL_WEIGHTS_ROOT).expanduser().resolve()
    for _block in _UNET_BLOCK_PATHS:
        _final_dir = _local_weights_root / _block / "final"
        if not _final_dir.exists():
            raise FileNotFoundError(
                "HAMBAJUBA_MODAL_SAE_WEIGHTS_ROOT must contain "
                f"{_block}/final, missing: {_final_dir}"
            )
        labeling_image = labeling_image.add_local_dir(
            str(_final_dir),
            remote_path=f"/sae_weights/{_block}/final",
        )

# Persistent volume for outputs (images + activations)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
artifact_cache_vol = modal.Volume.from_name(
    ARTIFACT_CACHE_VOLUME_NAME,
    create_if_missing=True,
)


# ─── Minimal SAE encode (avoids importing generation package) ────


def encode_topk(
    x: "torch.Tensor",
    encoder_weight: "torch.Tensor",
    pre_bias: "torch.Tensor",
    latent_bias: "torch.Tensor",
    k: int = 10,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Minimal top-k SAE encoding without importing SparseAutoencoder.

    Reproduces the exact logic from SparseAutoencoder.encode_topk():
        latents_pre_act = (x - pre_bias) @ encoder.T + latent_bias
        topk + relu

    Args:
        x: Input activations, shape (n_positions, d_model).
        encoder_weight: SAE encoder weights, shape (n_features, d_model).
        pre_bias: Pre-encoder centering bias, shape (d_model,).
        latent_bias: Post-projection bias, shape (n_features,).
        k: Number of top activations to keep.

    Returns:
        (indices, values) each shape (n_positions, k). Values are ReLU'd.
    """
    import torch

    x = x - pre_bias
    latents_pre_act = x @ encoder_weight.T + latent_bias
    vals, inds = torch.topk(latents_pre_act, k=k, dim=-1)
    return inds, torch.relu(vals)


def navigate_unet(unet: "torch.nn.Module", path: str) -> "torch.nn.Module":
    """Navigate UNet module tree by dot-separated path."""
    module = unet
    for attr in path.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
    return module


def resolve_sae_weights_root() -> Path:
    """Resolve SAE weights in the Modal container.

    Prefer locally mounted weights, otherwise download the public artifact
    bundle from Hugging Face into a persistent Modal volume cache.
    """
    mounted_root = Path("/sae_weights")
    if all(find_sae_block_dir(mounted_root, block) for block in _UNET_BLOCK_PATHS):
        return mounted_root

    from huggingface_hub import snapshot_download

    print(f"Downloading SAE weights from Hugging Face: {ARTIFACT_REPO_ID}")
    snapshot_root = Path(
        snapshot_download(
            repo_id=ARTIFACT_REPO_ID,
            repo_type=ARTIFACT_REPO_TYPE,
            allow_patterns=artifact_allow_patterns(),
            cache_dir="/cache/huggingface",
        )
    )
    weights_root = downloaded_weights_root(snapshot_root)
    missing = [
        block
        for block in _UNET_BLOCK_PATHS
        if find_sae_block_dir(weights_root, block) is None
    ]
    if missing:
        raise FileNotFoundError(
            "Missing SAE weights after Hugging Face download: "
            f"{', '.join(missing)}. Expected files under "
            f"{ARTIFACT_REPO_ID}/{ARTIFACT_WEIGHTS_SUBDIR or '<repo root>'}."
        )
    artifact_cache_vol.commit()
    return weights_root


def artifact_allow_patterns() -> list[str]:
    """Return narrow snapshot patterns for local and upstream checkpoint layouts."""
    prefix = ARTIFACT_WEIGHTS_SUBDIR.strip("/")
    patterns: list[str] = []
    for block in _UNET_BLOCK_PATHS:
        candidates = [f"{block}/final", block, _UPSTREAM_SDXL_SAE_DIRS[block]]
        for candidate in candidates:
            path = f"{prefix}/{candidate}" if prefix else candidate
            patterns.append(f"{path}/*")
    return patterns


def downloaded_weights_root(snapshot_root: Path) -> Path:
    prefix = ARTIFACT_WEIGHTS_SUBDIR.strip("/")
    if not prefix:
        return snapshot_root
    return snapshot_root / prefix


def find_sae_block_dir(weights_root: Path, block: str) -> Path | None:
    candidates = [
        weights_root / block / "final",
        weights_root / block,
        weights_root / _UPSTREAM_SDXL_SAE_DIRS[block],
        weights_root / _UPSTREAM_SDXL_SAE_DIRS[block] / "final",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists() and (
            candidate / "state_dict.pth"
        ).exists():
            return candidate
    return None


# ─── GPU generation class ───────────────────────────────────────


@app.cls(
    image=labeling_image,
    gpu="A100",
    volumes={"/output": output_vol, "/cache": artifact_cache_vol},
    timeout=14400,  # 4 hours
)
class ImageGenerator:
    """Persistent GPU worker that generates images and captures SAE activations.

    SDXL-Turbo and 4 SAE encoder weights are loaded once in setup().
    generate_batch() hooks all 4 attention blocks to extract sparse
    top-k activations, then aggregates per-feature statistics.
    """

    @modal.enter()
    def setup(self) -> None:
        """Load SDXL-Turbo pipeline and SAE encoder weights for all 4 blocks."""
        import torch
        from diffusers import StableDiffusionXLPipeline

        print("Loading SDXL-Turbo...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to("cuda")

        # Load SAE encoder weights (need full encoder for encode_topk)
        print("Loading SAE encoder weights...")
        self.sae_weights: dict[str, dict[str, torch.Tensor]] = {}
        weights_root = resolve_sae_weights_root()
        for block in _UNET_BLOCK_PATHS:
            weight_path = find_sae_block_dir(weights_root, block)
            if weight_path is None:
                raise FileNotFoundError(f"Missing SAE weights for block: {block}")
            state_dict = torch.load(
                weight_path / "state_dict.pth",
                map_location="cuda",
                weights_only=True,
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            self.sae_weights[block] = {
                "encoder_weight": state_dict["encoder.weight"].to(torch.float32),
                "pre_bias": state_dict["pre_bias"].to(torch.float32),
                "latent_bias": state_dict["latent_bias"].to(torch.float32),
            }
            n_f = self.sae_weights[block]["encoder_weight"].shape[0]
            d = self.sae_weights[block]["encoder_weight"].shape[1]
            print(f"  {block}: encoder ({n_f}, {d})")

        # Register forward hooks on all 4 attention blocks
        self.captured_activations: dict[str, "torch.Tensor"] = {}
        self.hooks: list = []

        for block, unet_path in _UNET_BLOCK_PATHS.items():
            attn_module = navigate_unet(self.pipe.unet, unet_path)
            handle = attn_module.register_forward_hook(
                self._make_capture_hook(block)
            )
            self.hooks.append(handle)

        Path("/output/generated").mkdir(parents=True, exist_ok=True)
        print("Setup complete.")

    def _make_capture_hook(self, block: str):
        """Create a forward hook that captures attention block output."""

        def hook(module, input, output):
            if isinstance(output, tuple):
                self.captured_activations[block] = output[0]
            else:
                self.captured_activations[block] = output.sample

        return hook

    def _aggregate_features(
        self,
        indices: "torch.Tensor",
        values: "torch.Tensor",
        spatial_w: int,
    ) -> list[dict]:
        """Aggregate per-feature stats across spatial positions (vectorized).

        For each unique feature that fired, computes mean, max, sum
        and the spatial (row, col) of the maximum activation.

        Returns list of dicts: {f, mean, max, sum, row, col}.
        Typically 500-2000 unique features per block per image.
        """
        import torch

        n_positions = indices.shape[0]

        # Flatten for scatter operations
        pos_indices = torch.arange(n_positions, device=indices.device)
        pos_indices = pos_indices.unsqueeze(1).expand_as(indices).reshape(-1)
        feat_ids = indices.reshape(-1)
        feat_vals = values.reshape(-1)

        unique_feats, inverse = torch.unique(feat_ids, return_inverse=True)
        n_unique = unique_feats.shape[0]

        # Sum + count for mean
        feat_sum = torch.zeros(n_unique, device=indices.device, dtype=torch.float32)
        feat_count = torch.zeros(n_unique, device=indices.device, dtype=torch.float32)
        feat_sum.scatter_add_(0, inverse, feat_vals.float())
        feat_count.scatter_add_(0, inverse, torch.ones_like(feat_vals, dtype=torch.float32))

        # Max value per feature
        feat_max = torch.full((n_unique,), float("-inf"), device=indices.device, dtype=torch.float32)
        feat_max.scatter_reduce_(0, inverse, feat_vals.float(), reduce="amax")

        # Argmax position: find position of max value per feature
        is_max = (feat_vals.float() == feat_max[inverse])
        max_positions = pos_indices.clone()
        max_positions[~is_max] = n_positions  # sentinel for non-max
        feat_max_pos = torch.full((n_unique,), n_positions, device=indices.device, dtype=torch.long)
        feat_max_pos.scatter_reduce_(0, inverse, max_positions, reduce="amin")

        feat_mean = feat_sum / feat_count

        # Move to CPU once
        unique_cpu = unique_feats.cpu().tolist()
        mean_cpu = feat_mean.cpu().tolist()
        max_cpu = feat_max.cpu().tolist()
        sum_cpu = feat_sum.cpu().tolist()
        pos_cpu = feat_max_pos.cpu().tolist()

        return [
            {
                "f": int(unique_cpu[i]),
                "mean": round(mean_cpu[i], 2),
                "max": round(max_cpu[i], 2),
                "sum": round(sum_cpu[i], 2),
                "row": pos_cpu[i] // spatial_w,
                "col": pos_cpu[i] % spatial_w,
            }
            for i in range(n_unique)
        ]

    @modal.method()
    def generate_batch(self, prompts: list[str], start_idx: int) -> int:
        """Generate images from prompts and capture SAE activations.

        Each forward pass hooks all 4 blocks, reshapes (B,C,H,W) to
        (H*W, d_model), runs encode_topk, and aggregates per-feature stats.
        Saves JPEG + appends to JSONL.

        Returns number of images generated.
        """
        import json

        import torch

        activations_path = Path("/output/activations.jsonl")
        generated_dir = Path("/output/generated")
        generated_dir.mkdir(parents=True, exist_ok=True)

        n_generated = 0

        for batch_offset, prompt in enumerate(prompts):
            idx = start_idx + batch_offset

            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

            self.captured_activations.clear()

            generator = torch.Generator(device="cuda").manual_seed(idx)
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=generator,
                output_type="pil",
            )
            image = result.images[0]
            image.save(str(generated_dir / f"{idx:05d}.jpg"), quality=90)

            # Process captured activations from all 4 blocks
            activation_record: dict = {"id": idx, "prompt": prompt}

            for block in _UNET_BLOCK_PATHS:
                if block not in self.captured_activations:
                    continue

                act = self.captured_activations[block]
                B, C, H, W = act.shape

                # (B,C,H,W) -> (H*W, d_model) = (256, 1280) for 512x512
                act_flat = act.permute(0, 2, 3, 1).reshape(-1, C).float()

                weights = self.sae_weights[block]
                indices, values = encode_topk(
                    act_flat,
                    weights["encoder_weight"],
                    weights["pre_bias"],
                    weights["latent_bias"],
                    k=10,
                )

                activation_record[block] = self._aggregate_features(
                    indices, values, spatial_w=W,
                )

            with open(str(activations_path), "a") as f:
                f.write(json.dumps(activation_record) + "\n")

            n_generated += 1
            if n_generated % 100 == 0:
                print(f"  Generated {n_generated}/{len(prompts)} (image {idx})")

        output_vol.commit()
        print(f"  Batch complete: {n_generated} images "
              f"(indices {start_idx}-{start_idx + n_generated - 1})")
        return n_generated

    @modal.method()
    def count_existing(self) -> int:
        """Count existing activation records on the volume (for resume)."""
        activations_path = Path("/output/activations.jsonl")
        if not activations_path.exists():
            return 0
        count = 0
        with open(str(activations_path)) as f:
            for _ in f:
                count += 1
        return count


# ─── Entry point ─────────────────────────────────────────────────


@app.local_entrypoint()
def main() -> None:
    """Direct Modal entrypoint. Use stage1_generate.py for full pipeline."""
    print("Use the stage1_generate.py orchestrator for full pipeline:")
    print("  uv run python -m hambajuba2ba.labeling.stage1_generate --n-images 50000")
