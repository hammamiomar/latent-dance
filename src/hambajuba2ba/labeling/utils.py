"""Shared utilities for the labeling pipeline.

Extracted from clip_direction.py. UNet navigation, SAE weight
loading, and image generation helpers used across multiple stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# UNet attention block paths (same as InlineSAEManager.DEFAULT_BLOCK_PATHS)
BLOCK_PATHS: dict[str, str] = {
    "down.2.1": "down_blocks.2.attentions.1",
    "mid.0": "mid_block.attentions.0",
    "up.0.0": "up_blocks.0.attentions.0",
    "up.0.1": "up_blocks.0.attentions.1",
}


# ─── Weight loading ──────────────────────────────────────────────


def load_sae_block(
    weight_path: Path,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load decoder weights + mean activations from checkpoint files.

    Self-contained — reads tensors directly, no SparseAutoencoder import needed.

    Returns:
        (decoder_weight, mean_activations)
        decoder_weight: (d_model, n_features) e.g. (1280, 5120)
        mean_activations: (n_features,) e.g. (5120,)
    """
    state_dict = torch.load(
        weight_path / "state_dict.pth",
        map_location=device,
        weights_only=True,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    decoder_weight = state_dict["decoder.weight"]  # (d_model, n_features)
    mean = torch.load(weight_path / "mean.pt", map_location=device, weights_only=True)

    return decoder_weight, mean


def load_activation_stats(weight_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mean and std activation statistics for a block."""
    mean = torch.load(weight_path / "mean.pt", map_location="cpu", weights_only=True)
    std = torch.load(weight_path / "std.pt", map_location="cpu", weights_only=True)
    return mean.numpy(), std.numpy()


# ─── UNet navigation + steering hooks ────────────────────────────


def navigate_unet(unet: nn.Module, path: str) -> nn.Module:
    """Navigate UNet module tree by dot-separated path.

    Example: navigate_unet(unet, "down_blocks.2.attentions.1")
    """
    module = unet
    for attr in path.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
    return module


def make_steering_hook(
    direction: torch.Tensor,
    strength: float,
) -> callable:
    """Forward hook that adds a scaled decoder direction to the attention output.

    Handles both tuple returns (return_dict=False) and dataclass
    returns (Transformer2DModelOutput) from diffusers attention blocks.
    """

    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
            C = out.shape[1]
            return (out + strength * direction.view(1, C, 1, 1),) + output[1:]
        else:
            out = output.sample
            C = out.shape[1]
            return type(output)(sample=out + strength * direction.view(1, C, 1, 1))

    return hook


# ─── Image generation ────────────────────────────────────────────


def generate_image(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    seed: int,
    device: str = "cuda",
) -> Image.Image:
    """Generate one image with SDXL-Turbo (1 step, no CFG)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=generator,
        output_type="pil",
    )
    return result.images[0]
