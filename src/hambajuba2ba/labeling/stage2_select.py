"""Stage 2: Feature ranking + image selection.

For each of 20,480 SAE features (5,120 per block x 4 blocks):
  1. Rank all 50K images by block-appropriate aggregation metric
  2. Select top-10 as ON set + 3 random non-activating as OFF contrast
  3. For up.0.0: crop ON images to spatial region of max activation

Input:
    data/labeling/sae_images/activations.jsonl  (one JSON line per image)
    data/labeling/sae_images/generated/{id:05d}.jpg

Output:
    data/labeling/sae_images/feature_sets/{block}/{fid}/metadata.json
    data/labeling/sae_images/feature_sets/up.0.0/{fid}/crop_{rank}.jpg
    data/labeling/sae_images/feature_sets/download_manifest.json

Usage:
    uv run python -m hambajuba2ba.labeling.stage2_select
    uv run python -m hambajuba2ba.labeling.stage2_select --block down.2.1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hambajuba2ba.labeling.config import (
    BLOCKS,
    BLOCK_RANKING_METRIC,
    LabelingConfig,
)
from hambajuba2ba.labeling.utils import load_activation_stats


# ─── Types ────────────────────────────────────────────────────────

# Per-image activation record for one feature:
# (image_id, mean, max, sum, row, col)
ActivationRecord = tuple[int, float, float, float, int, int]

# Inverted index: {block -> {feature_id -> [ActivationRecord, ...]}}
InvertedIndex = dict[str, dict[int, list[ActivationRecord]]]


# ─── Spatial cropping ─────────────────────────────────────────────


def crop_to_activation(
    image_path: Path,
    row: int,
    col: int,
    *,
    spatial_h: int = 16,
    spatial_w: int = 16,
    image_size: int = 512,
    crop_radius: int = 2,
    min_crop_px: int = 128,
) -> Image.Image:
    """Crop image to region around max activation for up.0.0 features.

    Each spatial cell covers image_size / spatial_h pixels. We take a
    window of crop_radius cells around the argmax, enforcing a minimum
    crop size, then resize to 512x512 for consistent VLM input.

    Args:
        image_path: Path to the source JPEG.
        row: Spatial row of the max activation.
        col: Spatial column of the max activation.
        spatial_h: Height of the feature activation grid.
        spatial_w: Width of the feature activation grid.
        image_size: Original image dimension in pixels (assumed square).
        crop_radius: Number of cells around argmax in each direction.
        min_crop_px: Minimum crop dimension in pixels.

    Returns:
        Cropped and resized 512x512 PIL Image.
    """
    cell_h = image_size // spatial_h
    cell_w = image_size // spatial_w

    # Region bounds in cells
    r_min = max(0, row - crop_radius)
    r_max = min(spatial_h, row + crop_radius + 1)
    c_min = max(0, col - crop_radius)
    c_max = min(spatial_w, col + crop_radius + 1)

    # Convert to pixels
    top = r_min * cell_h
    bottom = r_max * cell_h
    left = c_min * cell_w
    right = c_max * cell_w

    # Ensure minimum crop size
    if (bottom - top) < min_crop_px:
        mid = (top + bottom) // 2
        top = max(0, mid - min_crop_px // 2)
        bottom = min(image_size, top + min_crop_px)
    if (right - left) < min_crop_px:
        mid = (left + right) // 2
        left = max(0, mid - min_crop_px // 2)
        right = min(image_size, left + min_crop_px)

    img = Image.open(image_path)
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((512, 512), Image.LANCZOS)


# ─── Inverted index construction ─────────────────────────────────


def build_inverted_index(
    activations_path: Path,
    blocks: tuple[str, ...],
) -> tuple[InvertedIndex, int]:
    """Single-pass scan of activations.jsonl into an inverted index.

    Reads one JSON line per image. Each line contains per-block lists
    of (feature_id, mean, max, sum, row, col) records. We invert this
    into {block -> {feature_id -> [(image_id, mean, max, sum, row, col)]}}.

    Args:
        activations_path: Path to activations.jsonl.
        blocks: Which blocks to index.

    Returns:
        (inverted_index, n_images) where n_images is total lines read.
    """
    block_set = set(blocks)
    index: InvertedIndex = {b: defaultdict(list) for b in blocks}
    n_images = 0

    with open(activations_path) as f:
        for line in f:
            record = json.loads(line)
            image_id: int = record["id"]
            n_images += 1

            for block in block_set:
                if block not in record:
                    continue
                for entry in record[block]:
                    fid: int = entry["f"]
                    index[block][fid].append((
                        image_id,
                        entry["mean"],
                        entry["max"],
                        entry["sum"],
                        entry.get("row", 0),
                        entry.get("col", 0),
                    ))

    # Convert defaultdicts to regular dicts
    index = {b: dict(feats) for b, feats in index.items()}

    print(f"  Indexed {n_images} images across {len(blocks)} blocks")
    for block in blocks:
        n_feats = len(index[block])
        n_entries = sum(len(v) for v in index[block].values())
        print(f"    {block}: {n_feats} features, {n_entries} activation entries")

    return index, n_images


# ─── Ranking metric helpers ───────────────────────────────────────


def _sort_key_mean(rec: ActivationRecord) -> float:
    return rec[1]


def _sort_key_max(rec: ActivationRecord) -> float:
    return rec[2]


def _sort_key_sum(rec: ActivationRecord) -> float:
    return rec[3]


def _sort_key_adaptive(rec: ActivationRecord) -> float:
    """Adaptive: use mean if feature fires broadly, else max.

    Approximates spatial entropy from the mean/max ratio. When a feature
    fires at many positions, mean ~ max. When localized, mean << max.
    """
    mean_val, max_val = rec[1], rec[2]
    if max_val > 0 and mean_val > 0.5 * max_val:
        return mean_val  # Broad spatial spread — mean is meaningful
    return max_val  # Localized — max captures the spike


SORT_KEY_MAP = {
    "mean": _sort_key_mean,
    "max": _sort_key_max,
    "sum": _sort_key_sum,
    "adaptive": _sort_key_adaptive,
}


# ─── Feature selection ────────────────────────────────────────────


def select_feature_images(
    feature_records: list[ActivationRecord],
    metric: str,
    all_image_ids: set[int],
    rng: np.random.Generator,
    *,
    top_k: int = 10,
    n_off: int = 3,
    n_mid_range: int = 3,
    cv: float = 1.0,
    flat_cv_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank and select ON, OFF, and optional mid-range images for one feature.

    Args:
        feature_records: All activation records for this feature.
        metric: Ranking metric name (mean/max/sum/adaptive).
        all_image_ids: Full set of image IDs for OFF sampling.
        rng: NumPy random generator for reproducible OFF selection.
        top_k: Number of top images to select as ON set.
        n_off: Number of non-activating images for contrast.
        n_mid_range: Number of mid-range images (if CV is flat).
        cv: Coefficient of variation for this feature.
        flat_cv_threshold: CV below this triggers mid-range selection.

    Returns:
        (on_images, off_images, mid_range_images) as lists of metadata dicts.
    """
    sort_key = SORT_KEY_MAP[metric]
    sorted_records = sorted(feature_records, key=sort_key, reverse=True)

    # ON set: top-k
    on_images: list[dict[str, Any]] = []
    for rank, rec in enumerate(sorted_records[:top_k]):
        image_id, mean, max_val, sum_val, row, col = rec
        on_images.append({
            "image_id": image_id,
            "mean": mean,
            "max": max_val,
            "sum": sum_val,
            "row": row,
            "col": col,
            "rank": rank,
        })

    # OFF set: random images where this feature did NOT fire
    activating_ids = {rec[0] for rec in feature_records}
    non_activating = list(all_image_ids - activating_ids)

    off_images: list[dict[str, Any]] = []
    if non_activating:
        n_sample = min(n_off, len(non_activating))
        chosen = rng.choice(non_activating, size=n_sample, replace=False)
        for image_id in chosen:
            off_images.append({"image_id": int(image_id), "rank": -1})

    # Mid-range set: 40th-60th percentile (only for flat-distribution features)
    mid_range_images: list[dict[str, Any]] = []
    if cv < flat_cv_threshold and len(sorted_records) > top_k:
        n_total = len(sorted_records)
        p40 = int(n_total * 0.4)
        p60 = int(n_total * 0.6)
        mid_slice = sorted_records[p40:p60]
        if mid_slice:
            n_sample = min(n_mid_range, len(mid_slice))
            indices = rng.choice(len(mid_slice), size=n_sample, replace=False)
            for idx in indices:
                rec = mid_slice[idx]
                image_id, mean, max_val, sum_val, row, col = rec
                mid_range_images.append({
                    "image_id": image_id,
                    "mean": mean,
                    "max": max_val,
                    "sum": sum_val,
                    "row": row,
                    "col": col,
                    "rank": p40 + int(idx),
                })

    return on_images, off_images, mid_range_images


# ─── Per-block processing ─────────────────────────────────────────


def process_block(
    block: str,
    index: dict[int, list[ActivationRecord]],
    all_image_ids: set[int],
    cv_array: np.ndarray,
    cfg: LabelingConfig,
) -> set[int]:
    """Process all features for one block: rank, select, save metadata + crops.

    Returns:
        Set of all referenced image IDs (for the download manifest).
    """
    metric = BLOCK_RANKING_METRIC[block]
    rng = np.random.default_rng(cfg.seed)
    block_dir = cfg.feature_sets_dir / block
    referenced_ids: set[int] = set()
    is_up00 = block == "up.0.0"

    n_processed = 0
    n_empty = 0

    for fid in range(cfg.n_features):
        records = index.get(fid, [])

        if not records:
            n_empty += 1
            continue

        cv = float(cv_array[fid]) if fid < len(cv_array) else 1.0

        on_images, off_images, mid_range_images = select_feature_images(
            records, metric, all_image_ids, rng,
            top_k=cfg.top_k_images,
            n_off=cfg.n_off_images,
            n_mid_range=cfg.n_mid_range,
            cv=cv,
            flat_cv_threshold=cfg.flat_cv_threshold,
        )

        # Collect referenced image IDs
        for entry in on_images:
            referenced_ids.add(entry["image_id"])
        for entry in off_images:
            referenced_ids.add(entry["image_id"])
        for entry in mid_range_images:
            referenced_ids.add(entry["image_id"])

        # Save crops for up.0.0 ON images
        if is_up00:
            feat_dir = block_dir / str(fid)
            feat_dir.mkdir(parents=True, exist_ok=True)
            for entry in on_images:
                image_path = cfg.generated_dir / f"{entry['image_id']:05d}.jpg"
                if image_path.exists():
                    cropped = crop_to_activation(
                        image_path, entry["row"], entry["col"],
                        image_size=cfg.image_size,
                        crop_radius=cfg.crop_radius,
                        min_crop_px=cfg.min_crop_px,
                    )
                    crop_path = feat_dir / f"crop_{entry['rank']}.jpg"
                    cropped.save(crop_path, quality=90)

        # Write metadata
        metadata = {
            "block": block,
            "feature_id": fid,
            "on_images": on_images,
            "off_images": off_images,
            "mid_range_images": mid_range_images,
            "ranking_metric": metric,
            "activation_cv": round(cv, 4),
        }

        feat_dir = block_dir / str(fid)
        feat_dir.mkdir(parents=True, exist_ok=True)
        with open(feat_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        n_processed += 1

    print(f"    Processed: {n_processed} features")
    print(f"    Empty (never fired): {n_empty} features")
    print(f"    Referenced images: {len(referenced_ids)}")

    return referenced_ids


# ─── Activation CV computation ────────────────────────────────────


def compute_cv(weight_path: Path) -> np.ndarray:
    """Compute coefficient of variation (std/mean) per feature from SAE stats.

    Features with mean near zero get CV = 1.0 (treated as normal).
    """
    mean, std = load_activation_stats(weight_path)
    safe_mean = np.where(mean > 0, mean, 1.0)
    return std / safe_mean


# ─── Manifest ─────────────────────────────────────────────────────


def save_manifest(all_referenced: set[int], output_path: Path) -> None:
    """Save download manifest: sorted list of all referenced image IDs."""
    manifest = {
        "n_images": len(all_referenced),
        "image_ids": sorted(all_referenced),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {len(all_referenced)} unique images -> {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Feature ranking + image selection for SAE labeling",
    )
    parser.add_argument(
        "--block", type=str, default=None,
        help="Single block to process (default: all 4)",
    )
    args = parser.parse_args()

    cfg = LabelingConfig()
    blocks = (args.block,) if args.block else BLOCKS

    for block in blocks:
        if block not in BLOCK_RANKING_METRIC:
            parser.error(f"Unknown block '{block}'. Valid: {', '.join(BLOCKS)}")

    cfg.ensure_dirs()

    # Build inverted index
    print("\nBuilding inverted index...")
    index, n_images = build_inverted_index(cfg.activations_path, blocks)
    all_image_ids = set(range(n_images))

    # Process each block
    all_referenced: set[int] = set()

    for block in blocks:
        print(f"\n{'=' * 60}")
        print(f"  {block}  (metric: {BLOCK_RANKING_METRIC[block]})")
        print(f"{'=' * 60}")

        weight_path = cfg.weight_path(block)
        cv_array = compute_cv(weight_path)
        n_flat = int(np.sum(cv_array < cfg.flat_cv_threshold))
        print(f"    CV stats: mean={cv_array.mean():.3f}, "
              f"flat (<{cfg.flat_cv_threshold}): {n_flat}/{len(cv_array)}")

        block_referenced = process_block(
            block, index[block], all_image_ids, cv_array, cfg,
        )
        all_referenced |= block_referenced

    # Save download manifest
    manifest_path = cfg.feature_sets_dir / "download_manifest.json"
    save_manifest(all_referenced, manifest_path)
    print(f"\nDone. Feature sets in {cfg.feature_sets_dir}")


if __name__ == "__main__":
    main()
