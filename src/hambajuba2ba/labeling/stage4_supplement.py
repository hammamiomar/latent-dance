"""Stage 4: Zero-cost supplementary labeling methods.

Three independent signals that require no API calls:

    4A: Prompt-Activation TF-IDF
        For each feature, collect prompts of top-100 activating images.
        TF-IDF against full 50K corpus, extract top-10 distinctive terms.

    4B: Spatial Activation Analysis (mid.0 only)
        Average spatial activation maps across top-100 images.
        Classify pattern: center, border, corners, bands, uniform.

Usage:
    uv run python -m hambajuba2ba.labeling.stage4_supplement
    uv run python -m hambajuba2ba.labeling.stage4_supplement --block down.2.1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from hambajuba2ba.labeling.config import BLOCKS, LabelingConfig


# ─── Data loading ─────────────────────────────────────────────────


def load_activations(
    activations_path: Path,
    blocks: tuple[str, ...],
) -> tuple[dict[str, dict[int, list[tuple[int, float]]]], list[str], int]:
    """Load activations.jsonl and build per-feature ranking + prompt corpus.

    Returns:
        (rankings, prompts, n_images)
        rankings: {block -> {feature_id -> [(image_id, ranking_value)]}}
        prompts: list of prompt strings indexed by image_id
        n_images: total images
    """
    block_set = set(blocks)
    # Use mean for TF-IDF ranking (we just need top-N, metric doesn't matter much)
    rankings: dict[str, dict[int, list[tuple[int, float]]]] = {
        b: defaultdict(list) for b in blocks
    }
    prompts: list[str] = []
    n_images = 0

    with open(activations_path) as f:
        for line in f:
            record = json.loads(line)
            image_id: int = record["id"]
            prompts.append(record.get("prompt", ""))
            n_images += 1

            for block in block_set:
                if block not in record:
                    continue
                for entry in record[block]:
                    fid: int = entry["f"]
                    rankings[block][fid].append((image_id, entry["mean"]))

    rankings = {b: dict(feats) for b, feats in rankings.items()}
    print(f"  Loaded {n_images} images, {len(prompts)} prompts")
    return rankings, prompts, n_images


# ─── 4A: TF-IDF prompt analysis ──────────────────────────────────


def compute_tfidf_terms(
    rankings: dict[int, list[tuple[int, float]]],
    prompts: list[str],
    n_features: int,
    top_n_images: int = 100,
    top_n_terms: int = 10,
) -> dict[int, list[str]]:
    """Compute distinctive TF-IDF terms for each feature's top-activating images.

    For each feature, collects prompts of its top-N activating images,
    then uses TF-IDF against the full corpus to find distinctive terms.

    Returns:
        {feature_id: [term1, term2, ...]} for features that have activations.
    """
    # Build the TF-IDF vectorizer on the full corpus
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(prompts)
    feature_names = vectorizer.get_feature_names_out()

    results: dict[int, list[str]] = {}

    for fid in range(n_features):
        records = rankings.get(fid, [])
        if not records:
            continue

        # Sort by activation value, take top-N image IDs
        sorted_records = sorted(records, key=lambda x: x[1], reverse=True)
        top_ids = [r[0] for r in sorted_records[:top_n_images]]

        # Average TF-IDF vectors for this feature's top images
        if not top_ids:
            continue

        # Get the TF-IDF rows for these images
        subset = tfidf_matrix[top_ids]
        mean_tfidf = np.asarray(subset.mean(axis=0)).flatten()

        # Top-N distinctive terms
        top_indices = mean_tfidf.argsort()[::-1][:top_n_terms]
        terms = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]

        if terms:
            results[fid] = terms

    return results


# ─── 4B: Spatial activation analysis (mid.0 only) ─────────────────


def compute_spatial_patterns(
    rankings: dict[int, list[tuple[int, float]]],
    activations_path: Path,
    n_features: int,
    top_n_images: int = 100,
    spatial_h: int = 16,
    spatial_w: int = 16,
) -> dict[int, str | None]:
    """Classify spatial activation patterns for mid.0 features.

    For each feature, builds an average spatial heatmap from its
    top-activating images, then classifies the pattern.

    Returns:
        {feature_id: pattern_string_or_None}
    """
    # First, identify the top-N images per feature
    top_images_per_feature: dict[int, set[int]] = {}
    for fid in range(n_features):
        records = rankings.get(fid, [])
        if not records:
            continue
        sorted_records = sorted(records, key=lambda x: x[1], reverse=True)
        top_images_per_feature[fid] = {r[0] for r in sorted_records[:top_n_images]}

    if not top_images_per_feature:
        return {}

    # Build spatial heatmaps by scanning activations.jsonl for mid.0 entries
    # that include row/col positions
    heatmaps: dict[int, np.ndarray] = {
        fid: np.zeros((spatial_h, spatial_w), dtype=np.float32)
        for fid in top_images_per_feature
    }
    counts: dict[int, int] = {fid: 0 for fid in top_images_per_feature}

    with open(activations_path) as f:
        for line in f:
            record = json.loads(line)
            image_id: int = record["id"]

            if "mid.0" not in record:
                continue

            for entry in record["mid.0"]:
                fid: int = entry["f"]
                if fid not in top_images_per_feature:
                    continue
                if image_id not in top_images_per_feature[fid]:
                    continue

                row = entry.get("row", 0)
                col = entry.get("col", 0)
                val = entry.get("mean", 0.0)

                if 0 <= row < spatial_h and 0 <= col < spatial_w:
                    heatmaps[fid][row, col] += val
                    counts[fid] += 1

    # Classify each heatmap
    results: dict[int, str | None] = {}
    for fid in range(n_features):
        if fid not in heatmaps or counts.get(fid, 0) == 0:
            results[fid] = None
            continue
        results[fid] = _classify_spatial_pattern(heatmaps[fid], spatial_h, spatial_w)

    return results


def _classify_spatial_pattern(
    heatmap: np.ndarray,
    spatial_h: int = 16,
    spatial_w: int = 16,
) -> str | None:
    """Classify a spatial heatmap into a named pattern.

    Divides the 16x16 grid into regions and checks which has
    the highest mean activation relative to the total.
    """
    total = heatmap.sum()
    if total < 1e-6:
        return None

    # Normalize
    h = heatmap / total

    # Define regions (fractional boundaries)
    mid_h = spatial_h // 2
    mid_w = spatial_w // 2
    quarter_h = spatial_h // 4
    quarter_w = spatial_w // 4

    # Region activations (fraction of total)
    center = h[quarter_h:spatial_h - quarter_h, quarter_w:spatial_w - quarter_w].sum()
    top = h[:quarter_h, :].sum()
    bottom = h[spatial_h - quarter_h:, :].sum()
    left = h[:, :quarter_w].sum()
    right = h[:, spatial_w - quarter_w:].sum()
    top_band = h[:mid_h, :].sum()
    bottom_band = h[mid_h:, :].sum()
    left_band = h[:, :mid_w].sum()
    right_band = h[:, mid_w:].sum()

    # Border = everything except center
    border = 1.0 - center

    # Corners
    tl = h[:quarter_h, :quarter_w].sum()
    tr = h[:quarter_h, spatial_w - quarter_w:].sum()
    bl = h[spatial_h - quarter_h:, :quarter_w].sum()
    br = h[spatial_h - quarter_h:, spatial_w - quarter_w:].sum()
    corners = tl + tr + bl + br

    # Check uniformity: if max region fraction is < 0.35, it's uniform
    region_fractions = [center, top, bottom, left, right, corners]
    if max(region_fractions) < 0.35:
        return None  # uniform, no clear pattern

    # Classify by strongest signal
    if center > 0.5:
        return "center"
    if corners > 0.4:
        return "corners"
    if top > 0.35:
        return "top"
    if bottom > 0.35:
        return "bottom"
    if left > 0.35:
        return "left"
    if right > 0.35:
        return "right"

    # Check band patterns
    h_asymmetry = abs(top_band - bottom_band)
    v_asymmetry = abs(left_band - right_band)

    if h_asymmetry > 0.3:
        return "horizontal-band"
    if v_asymmetry > 0.3:
        return "vertical-band"

    if border > 0.65:
        return "border"

    return None  # No clear pattern


# ─── Output ───────────────────────────────────────────────────────


def save_supplements(
    block: str,
    tfidf_terms: dict[int, list[str]],
    spatial_patterns: dict[int, str | None],
    n_features: int,
    output_dir: Path,
) -> None:
    """Save supplement data as JSON list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for fid in range(n_features):
        entries.append({
            "feature_id": fid,
            "tfidf_top_terms": tfidf_terms.get(fid, []),
            "spatial_pattern": spatial_patterns.get(fid),
        })

    out_path = output_dir / f"{block}.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Saved {len(entries)} entries -> {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Zero-cost supplementary labeling methods",
    )
    parser.add_argument(
        "--block", type=str, default=None,
        help="Single block to process (default: all 4)",
    )
    args = parser.parse_args()

    cfg = LabelingConfig()
    cfg.ensure_dirs()
    blocks = (args.block,) if args.block else BLOCKS

    print("\nLoading activations...")
    rankings, prompts, n_images = load_activations(cfg.activations_path, blocks)

    for block in blocks:
        print(f"\n{'=' * 60}")
        print(f"  {block}")
        print(f"{'=' * 60}")

        # 4A: TF-IDF
        print("  Computing TF-IDF terms...")
        tfidf = compute_tfidf_terms(
            rankings[block], prompts, cfg.n_features,
            top_n_images=cfg.tfidf_top_n_images,
            top_n_terms=cfg.tfidf_top_n_terms,
        )
        n_with_terms = sum(1 for v in tfidf.values() if v)
        print(f"    {n_with_terms} features have TF-IDF terms")

        # 4B: Spatial analysis (mid.0 only)
        spatial: dict[int, str | None] = {}
        if block == "mid.0":
            print("  Computing spatial patterns...")
            spatial = compute_spatial_patterns(
                rankings[block], cfg.activations_path, cfg.n_features,
                top_n_images=cfg.tfidf_top_n_images,
            )
            n_patterned = sum(1 for v in spatial.values() if v is not None)
            print(f"    {n_patterned} features have spatial patterns")

        save_supplements(block, tfidf, spatial, cfg.n_features, cfg.supplement_dir)

    print(f"\nDone. Supplements in {cfg.supplement_dir}")


if __name__ == "__main__":
    main()
