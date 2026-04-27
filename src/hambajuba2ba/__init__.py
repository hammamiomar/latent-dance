"""HambaJuba2ba - Real-time audio-reactive SDXL-Turbo with SAE steering.

Subpackages:
- generation: ML inference (compiled CUDA graph), encoding, SAE steering
- bridge: Physics, spatial, composition, destinations, steering, clock
- audio: Offline DSP, feature extraction, stem separation
- config: Pipeline and server configuration
- presets: YAML preset files (envelope, brightness, physics, dual_layer)
"""

from .config import PipelineConfig, SAEConfig
from .generation.pipeline import SAESteerablePipeline
from .generation.sae import SparseAutoencoder, InlineSAEManager

__all__ = [
    # Configuration
    "PipelineConfig",
    "SAEConfig",
    # Pipelines
    "SAESteerablePipeline",
    # SAE steering
    "SparseAutoencoder",
    "InlineSAEManager",
]
