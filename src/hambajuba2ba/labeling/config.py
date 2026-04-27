"""Labeling pipeline configuration.

All paths and hyperparameters live here. Nothing depends on the
generation runtime — only shared constants are the block names
and SAE weight directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# Project root: src/hambajuba2ba/labeling/config.py → 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# UNet attention blocks in hookpoint order
BLOCKS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

# Block-appropriate aggregation metric for Stage 2 ranking.
# All three (mean, max, sum) are stored in Stage 1; this picks which to rank by.
BLOCK_RANKING_METRIC: dict[str, str] = {
    "down.2.1": "mean",     # Global compositional — pervasive activation = stronger signal
    "mid.0": "adaptive",    # Mixed: high spatial entropy → mean, low entropy → max
    "up.0.0": "max",        # Local details — strong local activation matters
    "up.0.1": "sum",        # Semi-global textures — rewards coverage + intensity
}

# Hand-labeled features — ground truth for validation.
# `also_accept` keywords soften matching: VLMs produce descriptive labels
# (e.g. "dark shadowy aesthetic") not terse shorthand ("dark/black").
KNOWN_FEATURES: dict[str, list[dict]] = {
    "down.2.1": [
        {"id": 2301, "label": "intense/evil", "category": "mood",
         "also_accept": ["evil", "monster", "villain", "dark", "sinister", "menacing", "demonic"]},
        {"id": 4074, "label": "anime style", "category": "style",
         "also_accept": ["anime", "manga", "japanese"]},
        {"id": 89, "label": "muscular", "category": "form",
         "also_accept": ["muscular", "hulk", "strong", "bodybuilder", "powerful", "buff", "muscles"]},
        {"id": 527, "label": "dark/black", "category": "color",
         "also_accept": ["dark", "black", "shadow"]},
        {"id": 4998, "label": "cartoon", "category": "style",
         "also_accept": ["cartoon", "animated", "3d render"]},
    ],
    "mid.0": [
        {"id": 1388, "label": "distortion", "category": "abstract",
         "also_accept": ["distort", "symmetr", "warp", "pattern", "abstract"]},
    ],
    "up.0.0": [
        {"id": 2937, "label": "shouting", "category": "expression",
         "also_accept": ["shout", "yell", "scream", "mouth", "tongue", "open mouth"]},
        {"id": 4161, "label": "smile", "category": "expression",
         "also_accept": ["smile", "smiling", "teeth", "grin", "happy"]},
        {"id": 2638, "label": "sunglasses", "category": "object",
         "also_accept": ["sunglass", "glasses", "eyewear", "spectacles"]},
        {"id": 4594, "label": "moustache", "category": "object",
         "also_accept": ["moustache", "mustache", "beard", "facial hair", "goatee"]},
        {"id": 775, "label": "buttons", "category": "texture",
         "also_accept": ["button", "vest", "fastener"]},
    ],
    "up.0.1": [
        {"id": 4977, "label": "tiger stripes", "category": "pattern",
         "also_accept": ["tiger", "stripe", "striped"]},
        {"id": 90, "label": "fur texture", "category": "texture",
         "also_accept": ["fur", "fuzzy", "furry", "hair texture"]},
        {"id": 1393, "label": "leopard spots", "category": "pattern",
         "also_accept": ["leopard", "spot", "animal print", "cheetah"]},
        {"id": 2615, "label": "twilight", "category": "lighting",
         "also_accept": ["twilight", "sunset", "dusk", "golden hour", "warm light"]},
        {"id": 3718, "label": "giraffe pattern", "category": "pattern",
         "also_accept": ["giraffe", "giraffe spot"]},
    ],
}


@dataclass(frozen=True)
class LabelingConfig:
    """Immutable labeling pipeline config."""

    # ── Input ──────────────────────────────────────────────────
    weights_dir: Path = Path(os.getenv("HAMBAJUBA_SAE_WEIGHTS_DIR", PROJECT_ROOT / "data" / "sdxl" / "sae_weights"))

    # ── Output base directories ────────────────────────────────
    labels_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels"
    images_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_images"
    clusters_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "clusters"

    # ── Stage 1: Image generation + activation logging ─────────
    generated_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_images" / "generated"
    activations_path: Path = PROJECT_ROOT / "data" / "labeling" / "sae_images" / "activations.jsonl"
    n_images: int = 50_000
    image_size: int = 512
    jpeg_quality: int = 90
    gen_chunk_size: int = 1_000

    # ── Stage 2: Feature ranking + image selection ─────────────
    feature_sets_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_images" / "feature_sets"
    top_k_images: int = 10
    n_off_images: int = 3
    n_mid_range: int = 3       # Extra mid-range images for flat-distribution features
    crop_radius: int = 2       # Spatial cells around argmax for up.0.0 cropping
    min_crop_px: int = 128     # Minimum crop size in pixels
    flat_cv_threshold: float = 0.5  # CV below this → flat distribution → add mid-range

    # ── Stage 3: VLM ensemble annotation ───────────────────────
    vlm_labels_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "vlm_labels"
    vlm_concurrency: int = 50
    vlm_timeout: float = 120.0
    vlm_max_retries: int = 3

    # ── Stage 4: Zero-cost supplements ─────────────────────────
    supplement_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "supplement"
    tfidf_top_n_images: int = 100  # Top-N images for prompt collection
    tfidf_top_n_terms: int = 10    # Top-N distinctive terms per feature

    # ── Stage 5: Label fusion ──────────────────────────────────
    final_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "final"
    embedding_model: str = "all-MiniLM-L6-v2"
    consensus_threshold: float = 0.7

    # ── Stage 6: Validation ────────────────────────────────────
    validation_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "validation"
    validation_sample_size: int = 200
    ground_truth_threshold: float = 0.75
    detection_threshold: float = 0.70

    # ── Stage 7: NMF factor grouping ─────────────────────────────
    factors_dir: Path = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "factors"
    nmf_n_factors: int = 150         # Number of NMF components per block
    nmf_top_features: int = 10       # Top-N features per factor to store
    nmf_max_iter: int = 500          # NMF iteration limit

    # ── Tier 0: HDBSCAN clustering (existing) ──────────────────
    min_cluster_size: int = 5
    min_samples: int = 3

    # ── Architecture constants ─────────────────────────────────
    n_features: int = 5120
    blocks: tuple[str, ...] = BLOCKS
    seed: int = 42

    def weight_path(self, block: str) -> Path:
        """Path to a block's SAE checkpoint directory."""
        return self.weights_dir / block / "final"

    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for d in (
            self.labels_dir, self.images_dir, self.clusters_dir,
            self.generated_dir, self.feature_sets_dir,
            self.vlm_labels_dir, self.supplement_dir,
            self.final_dir, self.validation_dir, self.factors_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
