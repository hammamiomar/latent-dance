"""Base pipeline configuration - composes all sub-configs."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

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
    - MPS: float32 (Apple Silicon requires this for stability)
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
        """Auto-detect device and dtype if not specified."""
        # Auto-detect device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Auto-detect dtype based on device
        if self.dtype is None:
            if self.device == "cuda":
                # float16 matches SDXL-Turbo fp16 variant weights (no conversion)
                self.dtype = "float16"
            else:
                # MPS and CPU need float32 for stability
                self.dtype = "float32"

        # Safety: MPS cannot use float16 reliably
        if self.device == "mps" and self.dtype == "float16":
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
    def variant(self) -> Optional[str]:
        """HuggingFace model variant for loading weights."""
        return "fp16" if self.dtype == "float16" else None

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
