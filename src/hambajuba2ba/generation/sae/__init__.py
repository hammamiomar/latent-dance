"""SAE (Sparse Autoencoder) module for interpretable feature steering.

Provides InlineSAEManager for torch.compile-safe steering of SDXL-Turbo
generation via learned sparse features. Each SAE feature corresponds to
an interpretable visual concept (e.g., "tiger stripes", "intense mood").
"""

from .model import SparseAutoencoder, unit_norm_decoder_
from .inline import InlineSAEManager, SteeredModule

__all__ = [
    "SparseAutoencoder",
    "unit_norm_decoder_",
    "InlineSAEManager",
    "SteeredModule",
]
