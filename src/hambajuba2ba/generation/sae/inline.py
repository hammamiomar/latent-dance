"""SAE feature steering via compiled module wrappers.

Wraps UNet attention blocks with SteeredModule — a thin nn.Module that adds
a steering vector after each attention forward pass. All mutable state lives
in registered buffers (updated via copy_()/fill_()), so the entire forward
path is traceable by torch.compile(fullgraph=True).

The steering math per block:
    out = attention(hidden_states)
    out = out + (strength * activation_map) * direction

Where:
    direction  — SAE decoder column for the active feature  (hidden_dim,)
    strength   — scalar amplitude, updated per-frame         (1,)
    activation_map — spatial weighting, pre-interpolated     (1, H, W)

Usage:
    manager = InlineSAEManager(unet, sae_weights_dir, device, dtype)
    manager.set_steering({"down.2.1": (2301, 15.0)})    # feature + strength
    manager.update_strengths({"down.2.1": (2301, 8.0)})  # per-frame update
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hambajuba2ba.artifacts import find_sae_block_dir

from .model import SparseAutoencoder

logger = logging.getLogger(__name__)


class SteeredModule(nn.Module):
    """Wraps a UNet attention block to inject SAE steering as tensor ops.

    All dynamic values are registered buffers so torch.compile/Dynamo can
    trace through the forward pass without graph breaks.
    """

    def __init__(
        self,
        original_module: nn.Module,
        decoder_weight: torch.Tensor,
    ):
        """Wrap an attention module with steering capability.

        Args:
            original_module: The attention block to wrap.
            decoder_weight: SAE decoder matrix (hidden_dim, n_features).
                           Columns are feature directions.
        """
        super().__init__()
        self.module = original_module

        # Full decoder — columns are feature directions, indexed by feature_id
        # Shape: (hidden_dim, n_features) e.g., (1280, 5120)
        self.register_buffer("decoder_weight", decoder_weight)

        hidden_dim = decoder_weight.shape[0]
        dev, dt = decoder_weight.device, decoder_weight.dtype

        # Active feature direction — set via set_feature(), updated with copy_()
        self.register_buffer("direction", torch.zeros(hidden_dim, dtype=dt, device=dev))

        # Scalar steering amplitude — set via set_strength(), updated with fill_()
        self.register_buffer("strength", torch.zeros(1, dtype=dt, device=dev))

        # Spatial weighting mask — (1, 1, 1) broadcasts to any size during warmup,
        # then fixed to per-block resolution by init_activation_map()
        self.register_buffer(
            "activation_map", torch.ones(1, 1, 1, dtype=dt, device=dev)
        )
        self._activation_map_initialized = False
        self._target_h = 1
        self._target_w = 1

    # ---- Per-feature setup (called when feature_id changes) ---------------

    def set_feature(self, feature_id: int) -> None:
        """Load a feature direction from the decoder matrix.

        Copies column `feature_id` from the decoder into the direction buffer.
        """
        n_features = self.decoder_weight.shape[1]
        if feature_id < 0 or feature_id >= n_features:
            raise ValueError(f"feature_id {feature_id} out of range [0, {n_features})")
        self.direction.copy_(self.decoder_weight[:, feature_id])

    def set_strength(self, value: float) -> None:
        """Set steering amplitude in-place (Dynamo-safe).

        Clamped to ±1000 as a practical safety bound — steering beyond this
        range produces garbage regardless of dtype.
        """
        self.strength.fill_(max(-1000.0, min(1000.0, value)))

    def clear(self) -> None:
        """Zero out steering for this block."""
        self.strength.fill_(0.0)

    # ---- Spatial activation maps ------------------------------------------

    def init_activation_map(self, height: int, width: int) -> None:
        """Pre-allocate a fixed-size activation map buffer.

        Must be called once during setup (before compile warmup) so the buffer
        shape is baked into the CUDA graph. Use update_activation_map() for
        per-frame in-place updates.
        """
        shape = (1, height, width)
        if self._activation_map_initialized:
            if tuple(self.activation_map.shape) == shape:
                return
            raise RuntimeError(
                "activation_map already initialized with shape "
                f"{tuple(self.activation_map.shape)}; refusing to resize to {shape}"
            )

        self._target_h = height
        self._target_w = width
        self.activation_map = torch.ones(
            shape,
            dtype=self.decoder_weight.dtype,
            device=self.decoder_weight.device,
        )
        self._activation_map_initialized = True

    def update_activation_map(self, values: torch.Tensor) -> None:
        """Update activation map in-place (compile-safe).

        Pre-interpolates to the block's target resolution OUTSIDE the compiled
        graph, so forward() can use the map without F.interpolate.
        """
        if values.dim() == 2:
            values = values.unsqueeze(0)

        target_h = getattr(self, "_target_h", self.activation_map.shape[1])
        target_w = getattr(self, "_target_w", self.activation_map.shape[2])

        if values.shape[1] != target_h or values.shape[2] != target_w:
            values = (
                F.interpolate(
                    values.unsqueeze(0).float(),  # (1, 1, H, W)
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .to(self.activation_map.dtype)
            )

        self.activation_map.copy_(values)

    # ---- Forward pass (inside compiled graph) -----------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Run wrapped attention + add steering vector.

        Diffusers attention blocks return (hidden_states,) tuples.
        """
        result = self.module(hidden_states, *args, **kwargs)
        out = result[0]
        C = out.shape[1]

        # steering = (scalar * spatial_mask) * direction_broadcasted
        steering = (self.strength * self.activation_map).unsqueeze(
            1
        ) * self.direction.view(1, C, 1, 1)
        out = out + steering

        return (out,)


# ===================================================================
# Manager — loads SAE weights, wraps UNet blocks, manages steering
# ===================================================================


class InlineSAEManager:
    """Manages SAE steering across multiple UNet attention blocks.

    At init time:
    1. Loads SAE decoder weights for each target block
    2. Wraps attention modules with SteeredModule
    3. Pre-caches mean activation stats (avoids GPU syncs at runtime)

    At runtime, two operations:
    - set_steering()    — called when feature IDs change (clears + sets all)
    - update_strengths() — called per-frame when only amplitudes change
    """

    # Maps short block names → UNet module paths
    DEFAULT_BLOCK_PATHS: dict[str, str] = {
        "down.2.1": "down_blocks.2.attentions.1",
        "mid.0": "mid_block.attentions.0",
        "up.0.0": "up_blocks.0.attentions.0",
        "up.0.1": "up_blocks.0.attentions.1",
    }

    def __init__(
        self,
        unet: nn.Module,
        sae_weights_dir: str,
        device: str | torch.device,
        dtype: torch.dtype,
        blocks: list[str] | None = None,
    ):
        """Load SAE decoders and wrap UNet attention blocks.

        Args:
            unet: SDXL UNet model (modified in-place).
            sae_weights_dir: Directory with per-block SAE weights.
            device: Target device.
            dtype: Target dtype (must match UNet).
            blocks: Block names to load (default: all 4).
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.steered_modules: dict[str, SteeredModule] = {}
        self.mean_stats: dict[str, torch.Tensor] = {}
        self.std_stats: dict[str, torch.Tensor] = {}
        self.block_order: list[str] = []

        # Pre-cached mean values: (block_name, feature_id) → float
        # Avoids .item() GPU syncs (~1-3ms each) during generation
        self._mean_cache: dict[tuple[str, int], float] = {}

        block_names = (
            blocks if blocks is not None else list(self.DEFAULT_BLOCK_PATHS.keys())
        )
        block_paths = {
            b: self.DEFAULT_BLOCK_PATHS[b]
            for b in block_names
            if b in self.DEFAULT_BLOCK_PATHS
        }
        self.block_order = list(block_paths.keys())

        logger.info(f"Initializing InlineSAEManager for blocks: {self.block_order}")

        for block_name, module_path in block_paths.items():
            block_dir = self._resolve_weights_dir(sae_weights_dir, block_name)
            if block_dir is None:
                logger.warning(f"SAE weights not found for block {block_name}")
                continue

            sae = SparseAutoencoder.load_from_disk(
                block_dir, device=self.device, dtype=self.dtype, decoder_only=True,
            )
            self._load_stats(block_name, block_dir)

            # Replace attention module with SteeredModule wrapper
            original = self._get_module(unet, module_path)
            wrapped = SteeredModule(original, decoder_weight=sae.decoder.weight)
            self._set_module(unet, module_path, wrapped)
            self.steered_modules[block_name] = wrapped

            logger.info(f"Wrapped {block_name} at {module_path}")

        self._precache_mean_values()
        logger.info(f"InlineSAEManager ready: {len(self.steered_modules)} blocks")

    # ---- Public API -------------------------------------------------------

    def set_steering(
        self,
        configs: dict[str, tuple[int, float]],
        use_mean_scaling: bool = True,
    ) -> None:
        """Set feature directions and strengths. Clears previous steering first.

        Called when feature IDs change (not every frame).

        Args:
            configs: {block_name: (feature_id, strength)}
            use_mean_scaling: Multiply strength by feature's mean activation.
        """
        for module in self.steered_modules.values():
            module.clear()

        for block_name, (feature_id, strength) in configs.items():
            if block_name not in self.steered_modules:
                continue
            self.steered_modules[block_name].set_feature(feature_id)
            self.steered_modules[block_name].set_strength(
                self._scaled_strength(
                    block_name, feature_id, strength, use_mean_scaling
                )
            )

    def update_strengths(
        self,
        configs: dict[str, tuple[int, float]],
        use_mean_scaling: bool = True,
    ) -> None:
        """Update only strength amplitudes (feature directions unchanged).

        Called per-frame for audio-reactive steering. Skips set_feature()
        since directions haven't changed — just fill_() the strength buffers.

        Args:
            configs: {block_name: (feature_id, strength)}
            use_mean_scaling: Multiply strength by feature's mean activation.
        """
        for block_name, (feature_id, strength) in configs.items():
            if block_name not in self.steered_modules:
                continue
            self.steered_modules[block_name].set_strength(
                self._scaled_strength(
                    block_name, feature_id, strength, use_mean_scaling
                )
            )

    def clear_hooks(self) -> None:
        """Zero all steering across all blocks."""
        for module in self.steered_modules.values():
            module.clear()

    # ---- Spatial activation maps ------------------------------------------

    def init_activation_maps(self, latent_h: int, latent_w: int) -> None:
        """Pre-allocate activation map buffers for all blocks.

        All steered blocks operate at latent/4 resolution in SDXL:
        down_blocks[0,1] downsample twice, so blocks 2/mid/up.0 see latent/4.
        """
        spatial_h = latent_h // 4
        spatial_w = latent_w // 4

        for block_name, module in self.steered_modules.items():
            module.init_activation_map(spatial_h, spatial_w)
            logger.debug(
                f"Initialized {block_name} activation map: {spatial_h}x{spatial_w}"
            )

    def update_activation_map(self, block_name: str, values: torch.Tensor) -> None:
        """Update one block's spatial mask in-place (compile-safe)."""
        if block_name in self.steered_modules:
            self.steered_modules[block_name].update_activation_map(values)

    # ---- Internal ---------------------------------------------------------

    def _scaled_strength(
        self,
        block_name: str,
        feature_id: int,
        strength: float,
        use_mean_scaling: bool,
    ) -> float:
        """Apply mean-activation scaling to a raw strength value.

        Mean scaling normalizes steering so that strength=1.0 produces
        roughly one standard deviation of that feature's natural activation.
        """
        if use_mean_scaling and block_name in self.mean_stats:
            mean_val = self._mean_cache.get((block_name, feature_id), 0.0)
            return strength * mean_val
        return strength

    def _precache_mean_values(self) -> None:
        """Bulk-transfer all mean stats to CPU floats at init time.

        Eliminates per-feature .item() calls (each triggers cudaDeviceSynchronize).
        """
        total = 0
        for block_name, mean_tensor in self.mean_stats.items():
            mean_cpu = mean_tensor.cpu().numpy()
            for fid in range(mean_cpu.shape[0]):
                self._mean_cache[(block_name, fid)] = float(mean_cpu[fid])
            total += mean_cpu.shape[0]
        logger.info(f"Pre-cached {total} mean values")

    def _resolve_weights_dir(self, base_dir: str, block_name: str) -> str | None:
        """Find the SAE weights directory for a block."""
        block_dir = find_sae_block_dir(base_dir, block_name)
        if block_dir is None:
            return None
        return str(block_dir)

    def _load_stats(self, block_name: str, block_dir: str) -> None:
        """Load mean/std activation statistics for a block."""
        mean_path = os.path.join(block_dir, "mean.pt")
        std_path = os.path.join(block_dir, "std.pt")
        if os.path.exists(mean_path):
            self.mean_stats[block_name] = torch.load(
                mean_path,
                map_location=self.device,
                weights_only=True,
            ).to(self.dtype)
        if os.path.exists(std_path):
            self.std_stats[block_name] = torch.load(
                std_path,
                map_location=self.device,
                weights_only=True,
            ).to(self.dtype)

    @staticmethod
    def _get_module(model: nn.Module, path: str) -> nn.Module:
        """Navigate model tree by dot-separated path."""
        module = model
        for attr in path.split("."):
            module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
        return module

    @staticmethod
    def _set_module(model: nn.Module, path: str, new_module: nn.Module) -> None:
        """Replace module at dot-separated path."""
        parts = path.split(".")
        parent = InlineSAEManager._get_module(model, ".".join(parts[:-1]))
        attr = parts[-1]
        if attr.isdigit():
            parent[int(attr)] = new_module
        else:
            setattr(parent, attr, new_module)
