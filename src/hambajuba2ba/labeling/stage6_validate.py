"""Stage 6: Validation — ground-truth gate, free-tier prompt iteration,
and detection scoring.

Sub-commands:
    free-iter  — Run 16 ground-truth + 84 random features on free VLMs ($0)
    gate       — Run 16 known features on paid model, abort if <75% ($0.01)
    detection  — Show 5 ON + 5 random shuffled, ask VLM to identify (~$0.50)
    roundtrip  — Not implemented in this public runtime release

Usage:
    uv run python -m hambajuba2ba.labeling.stage6_validate --mode free-iter
    uv run python -m hambajuba2ba.labeling.stage6_validate --mode gate
    uv run python -m hambajuba2ba.labeling.stage6_validate --mode detection
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random

import numpy as np
from sentence_transformers import SentenceTransformer

from hambajuba2ba.labeling.config import (
    BLOCKS,
    KNOWN_FEATURES,
    LabelingConfig,
)
from hambajuba2ba.labeling.openrouter import (
    FREE_MODELS,
    PAID_MODELS,
    OpenRouterClient,
    encode_image_b64,
)
from hambajuba2ba.labeling.stage3_annotate import BLOCK_PROMPTS, build_image_list

logger = logging.getLogger(__name__)


# ─── Label matching ───────────────────────────────────────────────


def _label_matches(
    predicted: str,
    expected: str,
    embedder: SentenceTransformer | None = None,
    sim_threshold: float = 0.6,
    also_accept: list[str] | None = None,
) -> bool:
    """Check if a predicted label semantically matches ground truth.

    Matching cascade (first match wins):
        1. Exact match (case-insensitive)
        2. Substring containment (either direction)
        3. also_accept keyword hit (substring in predicted)
        4. Embedding cosine similarity > threshold
    """
    pred = predicted.lower().strip()
    exp = expected.lower().strip()

    if pred == exp:
        return True
    if exp in pred or pred in exp:
        return True

    # Keyword expansion — catches VLM paraphrases of terse ground-truth labels
    if also_accept:
        for keyword in also_accept:
            if keyword.lower() in pred:
                return True

    if embedder is not None:
        vecs = embedder.encode([pred, exp], convert_to_numpy=True)
        sim = float(np.dot(vecs[0], vecs[1]) / (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-8
        ))
        return sim > sim_threshold

    return False


def _load_feature_meta(block: str, fid: int, cfg: LabelingConfig) -> dict | None:
    """Load metadata.json for a single feature."""
    path = cfg.feature_sets_dir / block / str(fid) / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── 6A: Free-tier prompt iteration ──────────────────────────────


async def run_free_iter(cfg: LabelingConfig) -> None:
    """Run 16 ground-truth + 84 random features through free VLMs.

    Gate: ≥12/16 ground-truth correct before spending money.
    """
    print(f"\n{'=' * 60}")
    print("  Stage 6A: Free-tier prompt iteration ($0)")
    print(f"{'=' * 60}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect ground-truth features (with also_accept keyword lists)
    gt_features: list[tuple[str, int, str, list[str] | None]] = []
    for block, known_list in KNOWN_FEATURES.items():
        for known in known_list:
            gt_features.append((
                block, known["id"], known["label"], known.get("also_accept"),
            ))

    # Collect 84 random features (that have metadata)
    rng = random.Random(42)
    random_features: list[tuple[str, int]] = []
    for block in BLOCKS:
        available = [
            fid for fid in range(cfg.n_features)
            if _load_feature_meta(block, fid, cfg) is not None
        ]
        if available:
            sample_n = min(21, len(available))  # ~21 per block × 4 ≈ 84
            random_features.extend(
                (block, fid) for fid in rng.sample(available, sample_n)
            )

    for model_name, model_id in FREE_MODELS.items():
        print(f"\n  --- {model_name} ---")

        n_correct = 0
        n_total = len(gt_features)

        async with OpenRouterClient(
            concurrency=10,  # Conservative for free tier
            timeout=cfg.vlm_timeout,
        ) as client:
            # Ground-truth features
            for block, fid, expected, also_accept in gt_features:
                meta = _load_feature_meta(block, fid, cfg)
                if meta is None:
                    print(f"    {block}/{fid}: no metadata, skipping")
                    continue

                try:
                    images_b64 = build_image_list(meta, cfg)
                    resp = await client.annotate(
                        model=model_id,
                        system_prompt=BLOCK_PROMPTS[block],
                        images_b64=images_b64,
                    )
                    matches = _label_matches(
                        resp.label, expected, embedder,
                        also_accept=also_accept,
                    )
                    status = "OK" if matches else "MISS"
                    if matches:
                        n_correct += 1
                    print(f"    [{status}] {block}/{fid}: "
                          f"expected='{expected}' got='{resp.label}'")
                except Exception as e:
                    print(f"    [ERR] {block}/{fid}: {e}")

            # Random features (just print labels for eyeballing)
            print("\n  Random features (first 20):")
            for block, fid in random_features[:20]:
                meta = _load_feature_meta(block, fid, cfg)
                if meta is None:
                    continue
                try:
                    images_b64 = build_image_list(meta, cfg)
                    resp = await client.annotate(
                        model=model_id,
                        system_prompt=BLOCK_PROMPTS[block],
                        images_b64=images_b64,
                    )
                    print(f"    {block}/{fid}: '{resp.label}' "
                          f"[{resp.category}, {resp.confidence}]")
                except Exception as e:
                    print(f"    {block}/{fid}: error: {e}")

        print(f"\n  Ground-truth: {n_correct}/{n_total} correct")
        if n_correct >= 12:
            print("  PASS — prompts are ready for paid run")
        else:
            print("  FAIL — iterate on prompts before spending money")


# ─── 6B: Ground-truth gate ────────────────────────────────────────


async def run_gate(cfg: LabelingConfig, model_key: str = "qwen3_vl_235b") -> None:
    """Run 16 known features on paid model. Abort if <75% accuracy."""
    print(f"\n{'=' * 60}")
    print("  Stage 6B: Ground-truth gate")
    print(f"{'=' * 60}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    model_id = PAID_MODELS[model_key]

    n_correct = 0
    n_total = 0

    async with OpenRouterClient(
        concurrency=cfg.vlm_concurrency,
        timeout=cfg.vlm_timeout,
    ) as client:
        for block, known_list in KNOWN_FEATURES.items():
            for known in known_list:
                fid = known["id"]
                expected = known["label"]
                meta = _load_feature_meta(block, fid, cfg)
                if meta is None:
                    print(f"  {block}/{fid}: no metadata, skipping")
                    continue

                try:
                    images_b64 = build_image_list(meta, cfg)
                    resp = await client.annotate(
                        model=model_id,
                        system_prompt=BLOCK_PROMPTS[block],
                        images_b64=images_b64,
                    )
                    matches = _label_matches(
                        resp.label, expected, embedder,
                        also_accept=known.get("also_accept"),
                    )
                    n_total += 1
                    if matches:
                        n_correct += 1
                    status = "OK" if matches else "MISS"
                    print(f"  [{status}] {block}/{fid}: "
                          f"expected='{expected}' got='{resp.label}'")
                except Exception as e:
                    print(f"  [ERR] {block}/{fid}: {e}")
                    n_total += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print(f"\n  Result: {n_correct}/{n_total} ({accuracy:.1%})")

    if accuracy >= cfg.ground_truth_threshold:
        print("  PASS — proceed with full run")
    else:
        print(f"  FAIL — accuracy {accuracy:.1%} < "
              f"threshold {cfg.ground_truth_threshold:.1%}")
        print("  DO NOT proceed with the $50+ paid run. Fix prompts first.")


# ─── 6C: Detection scoring ───────────────────────────────────────


async def run_detection(
    cfg: LabelingConfig, model_key: str = "qwen3_vl_235b",
) -> None:
    """Detection scoring: show 5 ON + 5 random shuffled, ask VLM to identify."""
    print(f"\n{'=' * 60}")
    print("  Stage 6C: Detection scoring")
    print(f"{'=' * 60}")

    # Load final labels
    all_features: list[dict] = []
    for block in BLOCKS:
        final_path = cfg.final_dir / f"{block}.json"
        if not final_path.exists():
            print(f"  Warning: no final labels for {block}")
            continue
        with open(final_path) as f:
            features = json.load(f)
        all_features.extend(features)

    if not all_features:
        print("  No final labels found. Run stage5_fuse first.")
        return

    # Sample 200 features (50 per block)
    rng = random.Random(42)
    sampled: list[dict] = []
    for block in BLOCKS:
        block_feats = [f for f in all_features if f["block"] == block]
        if len(block_feats) > 50:
            sampled.extend(rng.sample(block_feats, 50))
        else:
            sampled.extend(block_feats)

    model_id = PAID_MODELS[model_key]
    n_correct_on = 0
    n_correct_off = 0
    n_total = 0

    async with OpenRouterClient(
        concurrency=cfg.vlm_concurrency,
        timeout=cfg.vlm_timeout,
    ) as client:
        for feat in sampled:
            block = feat["block"]
            fid = feat["feature_id"]
            label = feat["label"]

            meta = _load_feature_meta(block, fid, cfg)
            if meta is None:
                continue

            # Get 5 ON images + 5 random OFF images
            on_ids = [o["image_id"] for o in meta["on_images"][:5]]
            off_ids = [o["image_id"] for o in meta["off_images"][:5]]

            if len(on_ids) < 5 or len(off_ids) < 3:
                continue

            # Pad OFF to 5 with more random images
            all_ids = set(range(50_000))
            activating = set(on_ids) | set(off_ids)
            extra = rng.sample(list(all_ids - activating), max(0, 5 - len(off_ids)))
            off_ids = off_ids + extra
            off_ids = off_ids[:5]

            # Shuffle and track which are ON
            indexed = [(i, "on") for i in on_ids] + [(i, "off") for i in off_ids]
            rng.shuffle(indexed)

            images_b64 = []
            for img_id, _ in indexed:
                path = cfg.generated_dir / f"{img_id:05d}.jpg"
                if path.exists():
                    images_b64.append(encode_image_b64(path))

            if len(images_b64) < 10:
                continue

            prompt = (
                f"I'm showing you 10 images numbered 1-10. "
                f"Which of these images match the description '{label}'? "
                f"List only the matching image numbers, comma-separated."
            )

            try:
                resp = await client.annotate(
                    model=model_id,
                    system_prompt="You are evaluating image-label matches.",
                    images_b64=images_b64,
                    user_text=prompt,
                    max_tokens=100,
                )

                # Parse response for numbers
                predicted_nums = set()
                for word in resp.raw.replace(",", " ").split():
                    word = word.strip(".,;")
                    if word.isdigit() and 1 <= int(word) <= 10:
                        predicted_nums.add(int(word))

                # Check accuracy
                for pos, (_, label_type) in enumerate(indexed, 1):
                    if label_type == "on" and pos in predicted_nums:
                        n_correct_on += 1
                    elif label_type == "off" and pos not in predicted_nums:
                        n_correct_off += 1

                n_total += 1
            except Exception as e:
                logger.warning("Detection error for %s/%d: %s", block, fid, e)

            if n_total % 50 == 0:
                print(f"  Progress: {n_total}/{len(sampled)}")

    if n_total > 0:
        sens = n_correct_on / (n_total * 5) if n_total > 0 else 0
        spec = n_correct_off / (n_total * 5) if n_total > 0 else 0
        balanced = (sens + spec) / 2

        print(f"\n  Features tested: {n_total}")
        print(f"  Sensitivity (ON correct): {sens:.1%}")
        print(f"  Specificity (OFF correct): {spec:.1%}")
        print(f"  Balanced accuracy: {balanced:.1%}")

        if balanced >= cfg.detection_threshold:
            print(f"  PASS (>= {cfg.detection_threshold:.1%})")
        else:
            print(f"  FAIL (< {cfg.detection_threshold:.1%})")
    else:
        print("  No features successfully tested.")


# ─── 6D: Round-trip generation test ──────────────────────────────


async def run_roundtrip(cfg: LabelingConfig) -> None:
    """Placeholder for round-trip generation test."""
    print(f"\n{'=' * 60}")
    print("  Stage 6D: Round-trip generation test")
    print(f"{'=' * 60}")
    print("  This test is not implemented in the public runtime release.")
    print("  For each sampled feature, uses its label as an SDXL-Turbo prompt,")
    print("  generates 20 images, and checks if the target feature activates.")
    print()
    print("  Stage 1 Modal reproduction code is included separately for")
    print("  image generation + activation capture.")
    print()
    print("  Expected: high recall for down.2.1, lower for mid.0 (by design).")


# ─── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: Validation for SAE feature labeling",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["free-iter", "gate", "detection", "roundtrip"],
        help="Validation mode to run",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3_vl_235b",
        help="Paid model key for gate/detection (default: qwen3_vl_235b)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = LabelingConfig()
    cfg.ensure_dirs()

    if args.mode == "free-iter":
        asyncio.run(run_free_iter(cfg))
    elif args.mode == "gate":
        asyncio.run(run_gate(cfg, model_key=args.model))
    elif args.mode == "detection":
        asyncio.run(run_detection(cfg, model_key=args.model))
    elif args.mode == "roundtrip":
        asyncio.run(run_roundtrip(cfg))


if __name__ == "__main__":
    main()
