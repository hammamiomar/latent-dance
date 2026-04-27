"""Tier 0: Decoder weight clustering.

Groups SAE features by cosine similarity of their decoder directions.
Pipeline: load weights → UMAP (1280d → 50d) → HDBSCAN → clusters.

Runs on CPU in ~30 seconds. Produces natural groupings that become
category assignments for the full labeling pipeline.

Why UMAP first: In 1280 dimensions, all unit-norm vectors are roughly
equidistant (mean cosine sim ~0.02). Density-based clustering can't
find structure. UMAP reveals the local manifold, then HDBSCAN works.

Usage:
    uv run python -m hambajuba2ba.labeling.clustering
    uv run python -m hambajuba2ba.labeling.clustering --block down.2.1
    uv run python -m hambajuba2ba.labeling.clustering --umap-dim 30
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.cluster import HDBSCAN

from hambajuba2ba.generation.sae.model import SparseAutoencoder
from hambajuba2ba.labeling.config import KNOWN_FEATURES, LabelingConfig


# ─── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterResult:
    """Clustering output for one block."""

    block: str
    n_features: int
    n_clusters: int
    n_noise: int
    assignments: list[int]            # feature_id → cluster_id (-1 = noise)
    clusters: dict[int, list[int]]    # cluster_id → list of feature_ids
    min_cluster_size: int
    min_samples: int
    umap_dim: int


# ─── Pure functions ──────────────────────────────────────────────


def load_decoder_weights(weight_path: Path) -> np.ndarray:
    """Load decoder directions as (n_features, d_model) unit-norm rows.

    The SAE decoder has weight shape (d_model, n_features).
    Transposing gives one row per feature direction on the unit sphere.
    """
    sae = SparseAutoencoder.load_from_disk(
        str(weight_path), device="cpu", dtype=torch.float32,
    )
    features = sae.decoder.weight.data.T.numpy()

    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"Decoder directions not unit norm: [{norms.min():.4f}, {norms.max():.4f}]"
    )
    return features


def reduce_dimensions(
    features: np.ndarray,
    *,
    n_components: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """UMAP: 1280d → n_components while preserving local cosine structure.

    Args:
        features: (n_features, d_model) array, unit-norm rows.
        n_components: Target dimensionality.
        n_neighbors: Local neighborhood size (higher = more global structure).
        min_dist: 0.0 packs clusters tightly, ideal for HDBSCAN downstream.
        seed: Reproducibility.

    Returns:
        (n_features, n_components) array in reduced space.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def cluster_features(
    features: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """HDBSCAN on reduced-dimensional features.

    Args:
        features: (n_features, n_components) array from UMAP.
        min_cluster_size: Minimum members for a valid cluster.
        min_samples: Core-point density threshold.

    Returns:
        Array of cluster labels, shape (n_features,). -1 = noise.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_jobs=-1,
    )
    return clusterer.fit_predict(features)


def build_result(
    block: str,
    labels: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    umap_dim: int,
) -> ClusterResult:
    """Package raw cluster labels into a structured result."""
    assignments = [int(label) for label in labels]

    clusters: dict[int, list[int]] = {}
    for feat_id, cid in enumerate(assignments):
        clusters.setdefault(cid, []).append(feat_id)

    n_clusters = len(set(assignments)) - (1 if -1 in assignments else 0)
    n_noise = int(np.sum(labels == -1))

    return ClusterResult(
        block=block,
        n_features=len(labels),
        n_clusters=n_clusters,
        n_noise=n_noise,
        assignments=assignments,
        clusters=clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_dim=umap_dim,
    )


# ─── Validation ──────────────────────────────────────────────────


def validate_known_features(result: ClusterResult) -> None:
    """Check that hand-labeled features cluster sensibly.

    Prints cluster assignments for our 16 known features and checks
    whether same-category features (e.g. tiger/leopard/giraffe) land
    in the same cluster.
    """
    known = KNOWN_FEATURES.get(result.block, [])
    if not known:
        return

    print("\n  Known features:")
    for feat in known:
        cid = result.assignments[feat["id"]]
        size = len(result.clusters.get(cid, []))
        tag = " [NOISE]" if cid == -1 else ""
        print(
            f"    {feat['label']:20s}  id={feat['id']:4d}"
            f"  → cluster {cid:3d}  (size {size}){tag}"
        )

    # Same-category validation
    by_cat: dict[str, list[tuple[str, int]]] = {}
    for feat in known:
        by_cat.setdefault(feat["category"], []).append(
            (feat["label"], result.assignments[feat["id"]])
        )

    for cat, members in by_cat.items():
        if len(members) < 2:
            continue
        cids = {cid for _, cid in members}
        names = ", ".join(label for label, _ in members)
        if len(cids) == 1 and -1 not in cids:
            print(f"    + {cat}: [{names}] share cluster {cids.pop()}")
        else:
            print(f"    ~ {cat}: [{names}] split across {cids}")


# ─── I/O ─────────────────────────────────────────────────────────


def save_result(result: ClusterResult, out_path: Path) -> None:
    """Serialize clustering result to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "block": result.block,
        "n_features": result.n_features,
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise,
        "assignments": result.assignments,
        "clusters": {str(k): v for k, v in sorted(result.clusters.items())},
        "params": {
            "min_cluster_size": result.min_cluster_size,
            "min_samples": result.min_samples,
            "umap_dim": result.umap_dim,
            "umap_metric": "cosine",
        },
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


# ─── CLI ─────────────────────────────────────────────────────────


def run_block(
    block: str,
    cfg: LabelingConfig,
    umap_dim: int = 50,
) -> ClusterResult:
    """Full pipeline for one block: load → UMAP → HDBSCAN → validate → save."""
    features = load_decoder_weights(cfg.weight_path(block))
    print(f"  {features.shape[0]} features x {features.shape[1]}d")

    print(f"  UMAP {features.shape[1]}d → {umap_dim}d ...", end=" ", flush=True)
    reduced = reduce_dimensions(features, n_components=umap_dim, seed=cfg.seed)
    print("done")

    labels = cluster_features(
        reduced,
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
    )
    result = build_result(
        block, labels, cfg.min_cluster_size, cfg.min_samples, umap_dim,
    )

    print(f"  Clusters: {result.n_clusters}")
    print(f"  Noise:    {result.n_noise} ({result.n_noise / result.n_features * 100:.1f}%)")

    # Show largest clusters
    sizes = sorted(
        [(k, len(v)) for k, v in result.clusters.items() if k != -1],
        key=lambda x: -x[1],
    )
    if sizes:
        print("  Largest:")
        for cid, size in sizes[:5]:
            print(f"    cluster {cid}: {size} features")

    validate_known_features(result)

    out_path = cfg.clusters_dir / f"{block}.json"
    save_result(result, out_path)
    print(f"  Saved → {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 0: UMAP + HDBSCAN clustering of SAE decoder weights",
    )
    parser.add_argument(
        "--block", type=str, default=None,
        help="Single block to cluster (default: all 4)",
    )
    parser.add_argument(
        "--umap-dim", type=int, default=50,
        help="UMAP target dimensionality (default: 50)",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="HDBSCAN min_cluster_size (default: from config)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=None,
        help="HDBSCAN min_samples (default: from config)",
    )
    args = parser.parse_args()

    cfg = LabelingConfig()
    if args.min_cluster_size is not None:
        cfg = LabelingConfig(min_cluster_size=args.min_cluster_size)
    if args.min_samples is not None:
        cfg = LabelingConfig(
            min_cluster_size=cfg.min_cluster_size,
            min_samples=args.min_samples,
        )
    cfg.ensure_dirs()

    blocks = (args.block,) if args.block else cfg.blocks

    for block in blocks:
        print(f"\n{'=' * 60}")
        print(f"  {block}")
        print(f"{'=' * 60}")
        run_block(block, cfg, umap_dim=args.umap_dim)

    print(f"\nDone. Results in {cfg.clusters_dir}")


if __name__ == "__main__":
    main()
