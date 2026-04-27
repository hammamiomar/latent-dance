"""Generation subpackage — ML inference, encoding, and SAE steering.

Modules:
- engine: Direct SDXL-Turbo inference (compiled CUDA graph)
- pipeline: Diffusers pipeline wrapper with SAE integration
- encoding: GPU→CPU tensor encoding (JPEG)
- sae/: Sparse autoencoder loading and inline steering
"""

from .pipeline import SAESteerablePipeline
from .encoding import gpu_to_cpu_tensor, encode_cpu_tensor
from .sae import SparseAutoencoder, InlineSAEManager, SteeredModule

__all__ = [
    "SAESteerablePipeline",
    "gpu_to_cpu_tensor",
    "encode_cpu_tensor",
    "SparseAutoencoder",
    "InlineSAEManager",
    "SteeredModule",
]
