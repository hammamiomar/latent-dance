"""Spatial mask manager for SAE steering.

Two modes:
  - draw: User-painted 16x16 binary grid, stored per-block
  - pitch_aligned: Dynamic pitch→Y mapping using 12 chromatic Gaussian bands
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import torch

from hambajuba2ba.audio.focus_config import get_base_stem
from hambajuba2ba.bridge.spatial import (
    generate_pitch_indexed_masks,
    generate_pitch_indexed_masks_gpu,
    pitch_to_mask_index,
)

if TYPE_CHECKING:
    from hambajuba2ba.audio.focus_config import BlockLinkConfig
    from hambajuba2ba.audio.sampler import AudioSampler

logger = logging.getLogger("uvicorn")

PRESET_MASKS: Dict[str, list[float]] = {
    "floor": [0.0] * 128 + [1.0] * 128,
    "ceiling": [1.0] * 128 + [0.0] * 128,
    "center": [0.0] * 64 + [1.0] * 128 + [0.0] * 64,
    "uniform": [1.0] * 256,
}


class SpatialManager:
    """Per-block 16x16 spatial masks. Pitch-aligned mode overrides per-frame from audio."""

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        latent_h: int,
        latent_w: int,
        **_kwargs,
    ):
        self.device = device
        self.dtype = dtype
        self.latent_h = latent_h
        self.latent_w = latent_w

        self._block_masks_gpu: Dict[str, torch.Tensor] = {}
        self._pitch_masks_gpu: Optional[torch.Tensor] = None
        self._pitch_masks_cpu: Optional[torch.Tensor] = None
        self._pitch_transfer_buf: Optional[torch.Tensor] = None

    def initialize(self) -> None:
        if self.device == "cuda":
            self._pitch_masks_gpu = generate_pitch_indexed_masks_gpu(
                self.latent_h, self.latent_w,
                device=self.device, dtype=self.dtype,
            )
        else:
            self._pitch_masks_cpu = generate_pitch_indexed_masks(
                self.latent_h, self.latent_w,
            )
            self._pitch_transfer_buf = torch.zeros(
                (self.latent_h, self.latent_w),
                device=self.device, dtype=self.dtype,
            )
        logger.info("Initialized spatial manager: 12 pitch masks + per-block draw buffers")

    def set_block_mask(self, block: str, mask_data: list[float]) -> None:
        t = torch.tensor(mask_data, dtype=self.dtype, device=self.device).reshape(16, 16)
        if block in self._block_masks_gpu:
            self._block_masks_gpu[block].copy_(t)
        else:
            self._block_masks_gpu[block] = t

    def update_masks(
        self,
        steering_manager,
        block_configs: Dict[str, "BlockLinkConfig"],
        audio_sampler: Optional["AudioSampler"] = None,
        audio_time: float = 0.0,
    ) -> None:
        if steering_manager is None:
            return

        for config in block_configs.values():
            if not config.enabled:
                continue

            if config.spatial_mode == "pitch_aligned" and audio_sampler is not None:
                self._update_pitch_aligned(steering_manager, config, audio_sampler, audio_time)
            else:
                mask = self._block_masks_gpu.get(config.block)
                if mask is not None:
                    steering_manager.update_activation_map(config.block, mask)

    def _update_pitch_aligned(
        self,
        steering_manager,
        config: "BlockLinkConfig",
        audio_sampler: "AudioSampler",
        audio_time: float,
    ) -> None:
        base_stem = get_base_stem(config.link_target)
        if not base_stem:
            return

        pitch_norm = audio_sampler.sample_pitch_normalized(base_stem, audio_time)
        pitch_idx = pitch_to_mask_index(pitch_norm)

        if self._pitch_masks_gpu is not None:
            steering_manager.update_activation_map(
                config.block, self._pitch_masks_gpu[pitch_idx]
            )
        elif self._pitch_masks_cpu is not None:
            mask_np = self._pitch_masks_cpu[pitch_idx]
            self._pitch_transfer_buf.copy_(torch.from_numpy(mask_np))
            steering_manager.update_activation_map(config.block, self._pitch_transfer_buf)

    def clear(self) -> None:
        self._block_masks_gpu.clear()
        self._pitch_masks_gpu = None
        self._pitch_masks_cpu = None
        self._pitch_transfer_buf = None
