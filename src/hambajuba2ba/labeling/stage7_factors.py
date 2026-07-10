"""Stage 7: NMF Factor Grouping — group co-occurring features into concepts.

Individual SAE features are too granular for users. No single feature
encodes "cat" — there are separate features for cat eyes, cat fur,
cat face. NMF discovers co-activation patterns and groups them into
~100-200 higher-level "factors" per block.

Inspired by Goodfire's Paint With Ember research. Unlike Goodfire
(per-image NMF for painting), we want global factors that persist
across frames for the music visualizer.

Input:
    data/labeling/sae_images/activations.jsonl  (from Stage 1)
    data/labeling/sae_labels/final/{block}.json (from Stage 5)

Output:
    data/labeling/sae_labels/factors/{block}.json

Usage:
    uv run python -m hambajuba2ba.labeling.stage7_factors
    uv run python -m hambajuba2ba.labeling.stage7_factors --block down.2.1
    uv run python -m hambajuba2ba.labeling.stage7_factors --n-factors 200
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF

from hambajuba2ba.labeling.config import (
    BLOCK_RANKING_METRIC,
    LabelingConfig,
)

logger = logging.getLogger(__name__)


# ─── Activation matrix construction ──────────────────────────────


def build_activation_matrix(
    activations_path: Path,
    block: str,
    n_features: int,
    metric: str = "mean",
) -> tuple[sparse.csr_matrix, int]:
    """Build (n_images, n_features) activation matrix from JSONL.

    Scans activations.jsonl once, constructing a sparse matrix where
    entry (i, j) = aggregated activation of feature j for image i.

    Returns:
        (activation_matrix, n_images)
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_images = 0

    # Adaptive metric → use mean for matrix construction
    effective_metric = "mean" if metric == "adaptive" else metric

    with open(activations_path) as f:
        for line in f:
            record = json.loads(line)
            image_id: int = record["id"]
            n_images = max(n_images, image_id + 1)

            if block not in record:
                continue

            for entry in record[block]:
                fid: int = entry["f"]
                val: float = entry[effective_metric]
                rows.append(image_id)
                cols.append(fid)
                vals.append(val)

    mat = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_images, n_features),
        dtype=np.float32,
    )
    return mat, n_images


# ─── Feature filtering ──────────────────────────────────────────


def load_spatial_feature_ids(
    final_dir: Path, block: str,
) -> set[int]:
    """Load feature IDs labeled as spatial patterns (from Stage 5).

    Spatial features waste NMF factors on positional information.
    Returns empty set if no final labels exist.
    """
    path = final_dir / f"{block}.json"
    if not path.exists():
        return set()
    with open(path) as f:
        features = json.load(f)
    return {
        f["feature_id"]
        for f in features
        if f.get("label", "").startswith("spatial:")
    }


def load_feature_labels(
    final_dir: Path, block: str,
) -> dict[int, dict]:
    """Load Stage 5 final labels for factor naming.

    Returns {feature_id: {label, category, confidence, ...}}.
    """
    path = final_dir / f"{block}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        features = json.load(f)
    return {f["feature_id"]: f for f in features}


# ─── NMF fitting ────────────────────────────────────────────────


def fit_nmf(
    activation_matrix: sparse.csr_matrix,
    n_components: int,
    exclude_features: set[int] | None = None,
    seed: int = 42,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Fit NMF and return factor-feature decomposition.

    Args:
        activation_matrix: (n_images, n_features) sparse matrix.
        n_components: Number of NMF factors.
        exclude_features: Feature IDs to exclude before fitting.
        seed: Random seed for reproducibility.
        max_iter: Maximum NMF iterations.

    Returns:
        (W, H, kept_feature_ids)
        W: (n_images, n_components) — image-to-factor weights
        H: (n_components, n_kept_features) — factor-to-feature weights
        kept_feature_ids: maps H columns back to original feature IDs
    """
    n_features = activation_matrix.shape[1]

    if exclude_features:
        kept = [i for i in range(n_features) if i not in exclude_features]
    else:
        kept = list(range(n_features))

    if len(kept) < n_components:
        raise ValueError(
            f"Only {len(kept)} features after filtering, but requested "
            f"{n_components} NMF components. Reduce --n-factors."
        )

    mat = activation_matrix[:, kept]

    # Clip negatives (SAE ReLU guarantees non-negative, but guard against float issues)
    if sparse.issparse(mat):
        mat.data = np.maximum(mat.data, 0)

    print(f"    Fitting NMF: {mat.shape[0]} images × {mat.shape[1]} features "
          f"→ {n_components} factors")

    nmf = NMF(
        n_components=n_components,
        init="nndsvd",
        random_state=seed,
        max_iter=max_iter,
    )
    W = nmf.fit_transform(mat)
    H = nmf.components_

    print(f"    Reconstruction error: {nmf.reconstruction_err_:.2f}")
    print(f"    Iterations: {nmf.n_iter_}")

    return W, H, kept


# ─── Factor labeling ────────────────────────────────────────────


_STOPWORDS = frozenset({
    "the", "and", "with", "for", "from", "that", "this", "into",
    "over", "under", "between", "through", "about", "after", "before",
    "looking", "images", "image", "first", "last", "reference",
    "feature", "active", "not", "are", "was", "were", "has", "have",
    "but", "also", "very", "more", "most", "some", "other", "each",
    "all", "both", "such", "than", "when", "where", "which", "while",
    "being", "having", "their", "these", "those", "what", "will",
    "can", "may", "its", "various", "multiple", "different", "similar",
    "visible", "present", "showing", "including", "versus",
})


def _is_cot_label(label: str) -> bool:
    """Detect chain-of-thought reasoning that leaked through as a label.

    Some VLMs (especially kimi) output reasoning text like
    "Looking at the first 3 images..." instead of concise labels.
    """
    lower = label.lower()
    return (
        lower.startswith("looking at")
        or "i need to" in lower
        or "i can see" in lower
        or "the first 3" in lower
        or "reference images" in lower
        or len(label) > 80  # Real labels are ≤10 words (~60 chars)
    )


def _find_common_theme(labels: list[str], min_count: int = 3) -> str | None:
    """Find a common theme word across feature labels.

    If 3+ labels share a meaningful word, use the most frequent one.
    Filters stopwords and VLM boilerplate ("Looking at the first 3 images...").
    E.g., ["cat eyes", "cat fur", "cat face"] → "cat".
    """
    if len(labels) < min_count:
        return None

    word_counts: dict[str, int] = defaultdict(int)
    for label in labels:
        seen_words: set[str] = set()
        for word in label.lower().split():
            clean = word.strip(".,;:()\"'-/")
            if len(clean) > 2 and clean not in seen_words and clean not in _STOPWORDS:
                word_counts[clean] += 1
                seen_words.add(clean)

    common = [(w, c) for w, c in word_counts.items() if c >= min_count]
    if not common:
        return None

    # Most frequent, break ties by longest word
    common.sort(key=lambda x: (-x[1], -len(x[0])))
    return common[0][0]


# ── Factor label strategy: design notes ─────────────────────────
#
# Current approach: use the highest-weighted feature's label, or
# the common theme word if 3+ constituent labels share one.
#
# This is a design decision with meaningful trade-offs:
#
# Option A (current): Simple heuristic — common word or top feature label.
#   + Zero cost, instant, deterministic
#   - Misses semantic relationships ("warm", "golden", "sunset" → "warm lighting")
#
# Option B: Sentence embedding centroid — embed all constituent labels,
#   find the one closest to the centroid.
#   + Captures semantic similarity without shared words
#   - Still limited to existing vocabulary (can't synthesize "cat" from "cat eyes")
#
# Option C: VLM summarization — send top-5 constituent labels to a VLM,
#   ask for a one-word concept name.
#   + Best quality, handles abstraction ("cat eyes" + "cat fur" → "cat")
#   - Costs money, adds API dependency to an offline stage
#
# For now, Option A is good enough. The constituent list is always
# preserved, so factor labels can be refined later.


def extract_factors(
    H: np.ndarray,
    W: np.ndarray,
    kept_feature_ids: list[int],
    feature_labels: dict[int, dict],
    top_n_features: int = 10,
) -> list[dict]:
    """Extract named factors from NMF decomposition.

    For each factor, finds the top-weighted features, looks up their
    labels from Stage 5, and picks a representative label.
    """
    n_factors = H.shape[0]
    factors: list[dict] = []

    for factor_idx in range(n_factors):
        weights = H[factor_idx]

        # Top features by NMF weight
        top_local = np.argsort(weights)[::-1][:top_n_features]
        top_weights = weights[top_local]

        # Map back to original feature IDs
        constituents: list[dict] = []
        labels_for_naming: list[str] = []

        for local_idx, weight in zip(top_local, top_weights):
            if weight < 1e-6:
                break
            fid = kept_feature_ids[local_idx]
            info = feature_labels.get(fid, {})
            label = info.get("label", f"feature_{fid}")

            constituents.append({
                "feature_id": fid,
                "weight": round(float(weight), 4),
                "label": label,
                "category": info.get("category", "unknown"),
            })
            if label != "unlabeled" and not _is_cot_label(label):
                labels_for_naming.append(label)

        if not constituents:
            continue

        # Try common theme, fall back to first non-CoT constituent label
        theme = _find_common_theme(labels_for_naming)
        fallback = next(
            (c["label"] for c in constituents if not _is_cot_label(c["label"])),
            constituents[0]["label"],
        )
        factor_label = theme if theme else fallback
        factor_category = constituents[0]["category"]

        # Factor activation statistics
        factor_col = W[:, factor_idx]
        factor_mean = float(factor_col.mean())
        factor_max = float(factor_col.max())
        n_active = int(np.sum(factor_col > 0))

        factors.append({
            "factor_id": factor_idx,
            "label": factor_label,
            "category": factor_category,
            "n_constituents": len(constituents),
            "constituents": constituents,
            "constituent_labels": labels_for_naming[:5],
            "mean_activation": round(factor_mean, 4),
            "max_activation": round(factor_max, 4),
            "n_active_images": n_active,
        })

    # Sort by mean activation (most active factors first)
    factors.sort(key=lambda f: f["mean_activation"], reverse=True)

    # Re-number after sorting
    for i, f in enumerate(factors):
        f["factor_id"] = i

    return factors


# ─── Block-level processing ─────────────────────────────────────


def _cache_path(factors_dir: Path, block: str, suffix: str) -> Path:
    return factors_dir / f".nmf_cache_{block}_{suffix}.npy"


def process_block(
    block: str,
    cfg: LabelingConfig,
    n_factors: int,
    refit: bool = True,
) -> list[dict]:
    """Run NMF factor grouping for one block.

    If refit=False and cached W/H exist, skips the expensive NMF fit
    and just re-runs factor naming (useful for iterating on labels).
    """
    # Load feature labels for factor naming
    feature_labels = load_feature_labels(cfg.final_dir, block)
    n_labeled = sum(
        1 for v in feature_labels.values() if v.get("label") != "unlabeled"
    )
    print(f"    {n_labeled} features have labels")

    # Filter spatial features (primarily mid.0)
    exclude = load_spatial_feature_ids(cfg.final_dir, block)
    if exclude:
        print(f"    Filtering {len(exclude)} spatial features")

    w_cache = _cache_path(cfg.factors_dir, block, "W")
    h_cache = _cache_path(cfg.factors_dir, block, "H")
    kept_cache = _cache_path(cfg.factors_dir, block, "kept")

    if not refit and w_cache.exists() and h_cache.exists() and kept_cache.exists():
        print("    Loading cached NMF matrices (use --refit to recompute)")
        W = np.load(w_cache)
        H = np.load(h_cache)
        kept = np.load(kept_cache).tolist()
    else:
        metric = BLOCK_RANKING_METRIC.get(block, "mean")
        print(f"  Building activation matrix ({metric} metric)...")
        mat, n_images = build_activation_matrix(
            cfg.activations_path, block, cfg.n_features, metric,
        )
        print(f"    {n_images} images, {mat.nnz} non-zero entries "
              f"({mat.nnz / (n_images * cfg.n_features) * 100:.1f}% fill)")

        available = cfg.n_features - len(exclude)
        effective_n = min(n_factors, available - 1)

        W, H, kept = fit_nmf(
            mat, effective_n, exclude_features=exclude,
            seed=cfg.seed, max_iter=cfg.nmf_max_iter,
        )

        # Cache for fast re-naming
        np.save(w_cache, W)
        np.save(h_cache, H)
        np.save(kept_cache, np.array(kept))

    factors = extract_factors(
        H, W, kept, feature_labels,
        top_n_features=cfg.nmf_top_features,
    )

    return factors


# ─── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7: NMF factor grouping for SAE features",
    )
    parser.add_argument(
        "--block", type=str, default=None,
        help="Single block to process (default: all 4)",
    )
    parser.add_argument(
        "--n-factors", type=int, default=None,
        help="Number of NMF factors per block (default: from config)",
    )
    parser.add_argument(
        "--refit", action="store_true",
        help="Force NMF refit even if cached matrices exist",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    cfg = LabelingConfig()
    cfg.ensure_dirs()
    blocks = (args.block,) if args.block else cfg.blocks
    n_factors = args.n_factors if args.n_factors is not None else cfg.nmf_n_factors

    for block in blocks:
        print(f"\n{'=' * 60}")
        print(f"  Stage 7: NMF Factor Grouping — {block}")
        print(f"{'=' * 60}")

        factors = process_block(block, cfg, n_factors, refit=args.refit)

        # Save
        out_path = cfg.factors_dir / f"{block}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(factors, f, indent=2)

        # Summary
        n_features_covered = len({
            c["feature_id"]
            for factor in factors
            for c in factor["constituents"]
        })
        print(f"\n  Factors: {len(factors)}")
        print(f"  Features covered: {n_features_covered}/{cfg.n_features}")
        print(f"  -> {out_path}")

    print(f"\nDone. Factors in {cfg.factors_dir}")


if __name__ == "__main__":
    main()
