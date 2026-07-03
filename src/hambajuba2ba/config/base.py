"""Base pipeline configuration - composes all sub-configs."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from hambajuba2ba.device import autodetect

from .server import ServerConfig
from .audio import AudioConfig
from .streaming import StreamingConfig
from .strategy import StrategyConfig
from .sae import SAEConfig


@dataclass
class PipelineConfig:
    """Master configuration - composes all sub-configs.

    Device and dtype are auto-detected by default:
    - CUDA: float16 (matches model weights, no conversion overhead)
    - MPS: float32 default; explicit float16 respected (validated Jul 2026)
    - CPU: float32 (compatibility)

    Override by passing explicit values.
    """

    # Nested configs
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)

    # === Model IDs ===
    sdxl_model_id: str = "stabilityai/sdxl-turbo"
    sdxl_vae_id: str = "madebyollin/taesdxl"

    # === Device and dtype ===
    # None = auto-detect
    device: Optional[Literal["cuda", "mps", "cpu"]] = None
    dtype: Optional[Literal["float16", "float32", "bfloat16"]] = None

    # === Generation settings ===
    height: int = 512  # SDXL-Turbo works best at 512
    width: int = 512
    num_inference_steps: int = 1  # SDXL-Turbo is 1-step
    seed: int = 42  # Default seed

    # === Optimizations ===
    use_tiny_vae: bool = True  # TinyVAE: 2x faster decode, smaller sync (worth the quality tradeoff)

    # === Execution ===
    cpu_workers: int = 4
    # 3 iterations for CUDA graph capture (CuBLAS benchmark → graph record → verify)
    warmup_iterations: int = 3

    def __post_init__(self):
        """Resolve auto-detected fields at construction."""
        self.resolve()

    def resolve(self) -> None:
        """Fill unset device/dtype and enforce device-dtype coherence.

        Runs at construction, and again from the config loader after
        env/YAML overrides land: an overridden device must re-derive a
        stale derived dtype (HAMBAJUBA_DEVICE=cpu on a CUDA box must
        not keep the float16 chosen at construction). Idempotent.
        """
        if self.device is None:
            self.device = autodetect()

        if self.dtype is None:
            if self.device == "cuda":
                # float16 matches SDXL-Turbo fp16 variant weights (no conversion)
                self.dtype = "float16"
            else:
                # Conservative default off-CUDA. Explicit float16 on MPS is
                # respected — validated Jul 2026 on torch 2.10 (M1 Max:
                # ~1.3x faster, clean output; TinyVAE is fp16-safe).
                self.dtype = "float32"

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.dtype]

    @property
    def latent_height(self) -> int:
        """Height in latent space (VAE downsamples by 8)."""
        return self.height // 8

    @property
    def latent_width(self) -> int:
        """Width in latent space (VAE downsamples by 8)."""
        return self.width // 8

    @property
    def variant(self) -> str:
        """HuggingFace checkpoint variant to download.

        Always the fp16 files: diffusers upcasts them to torch_dtype at
        load, so float32 runs (mps/cpu) get correct weights from half
        the download (~7GB fp16 vs ~13GB full-precision checkpoint).
        """
        return "fp16"

    # Convenience accessors for nested config values (backward compatibility)
    @property
    def fps(self) -> float:
        """Target frame rate (from streaming config)."""
        return self.streaming.fps

    @property
    def jpeg_quality(self) -> int:
        """JPEG compression quality (from streaming config)."""
        return self.streaming.jpeg_quality

    @property
    def audio_sample_rate(self) -> int:
        """Audio sample rate (from audio config)."""
        return self.audio.sample_rate
