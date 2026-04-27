"""SAE (Sparse Autoencoder) configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class SAEConfig:
    """Configuration for SAE feature steering."""

    # Weights location
    weights_dir: str = "./data/sdxl/sae_weights"

    # Public artifact fallback. Local weights are preferred; if they are absent,
    # runtime can download the configured subtree from Hugging Face.
    auto_download_weights: bool = True
    artifact_repo_id: str = "surokpro2/sdxl-saes"
    artifact_repo_type: str = "model"
    artifact_weights_subdir: str = ""
    artifact_cache_dir: str = str(
        Path.home() / ".cache" / "hambajuba2ba" / "artifacts"
    )

    # Blocks to load (SDXL UNet attention blocks)
    blocks: List[str] = field(
        default_factory=lambda: ["down.2.1", "mid.0", "up.0.0", "up.0.1"]
    )

    # Whether to scale by mean activation
    use_mean_scaling: bool = True
