"""Stage 3: VLM Ensemble Annotation via OpenRouter.

For each of 20,480 SAE features, send max-activating ON images + OFF
contrast images to VLMs for labeling. Uses a tiered model strategy:

    Pass 1 (qwen):  Qwen3-VL-235B on all 20,480 features
    Pass 2 (glm):   GLM-4.6V on all 20,480 features
    Pass 3 (kimi):  Kimi K2.5 on disagreements only (qwen != glm)

Each pass writes JSONL per block. Stage 5 does the real semantic
fusion; this stage just collects raw VLM opinions.

Usage:
    uv run python -m hambajuba2ba.labeling.stage3_annotate --pass qwen
    uv run python -m hambajuba2ba.labeling.stage3_annotate --pass glm --block down.2.1
    uv run python -m hambajuba2ba.labeling.stage3_annotate --pass kimi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from hambajuba2ba.labeling.config import BLOCKS, LabelingConfig
from hambajuba2ba.labeling.openrouter import (
    PAID_MODELS,
    OpenRouterClient,
    VLMResponse,
    encode_image_b64,
)

logger = logging.getLogger(__name__)


# ─── Block-specific VLM prompts ──────────────────────────────────

BLOCK_PROMPTS: dict[str, str] = {
    "down.2.1": """You are labeling visual features from a neural network.
You will see 13 images. The FIRST 3 images are reference images where a specific feature is NOT active.
The LAST 10 images show the feature at increasing activation strength.

What compositional element, object, scene type, or mood is present
in the last 10 images but absent or reduced in the first 3?

Respond with ONLY:
- label: a concise description (≤10 words)
- category: one of [object, scene, mood, color, composition, character]
- confidence: high / medium / low""",

    "up.0.1": """You are labeling STYLE and TEXTURE features from a neural network.
IGNORE the subject matter of the images. Focus ONLY on visual properties.

You will see 13 images. The FIRST 3 are reference images where a specific feature is NOT active.
The LAST 10 show the feature at increasing activation strength.

What shared artistic style, color palette, lighting quality, surface
texture, or visual pattern appears in the last 10 but not the first 3?

Respond with ONLY:
- label: a concise description of the visual property (≤10 words)
- category: one of [texture, pattern, color, lighting, style, material]
- confidence: high / medium / low""",

    "up.0.0": """You are labeling DETAIL features from a neural network.
These features encode specific local visual elements like facial features,
accessories, body parts, or object details.

You will see 13 images. The FIRST 3 are uncropped reference images where the feature is NOT active.
The LAST 10 are CROPPED regions showing where the feature is active, at increasing strength.

What specific visual detail appears across the cropped regions?

Respond with ONLY:
- label: a concise description (≤10 words)
- category: one of [face, body, accessory, object_detail, edge, shape]
- confidence: high / medium / low""",

    "mid.0": """You are labeling STRUCTURAL features from a neural network.
These features may encode abstract properties like spatial arrangement,
borders, symmetry, depth, or image structure — NOT specific objects.

You will see 13 images. The FIRST 3 are reference images where the feature is NOT active.
The LAST 10 show the feature at increasing activation strength.

Describe any shared STRUCTURAL property: spatial arrangement, symmetry,
depth, borders, framing, density, contrast pattern, or visual rhythm.
It's OK to say "unclear" if no pattern is visible.

Respond with ONLY:
- label: a concise description (≤10 words)
- category: one of [spatial, symmetry, border, depth, density, contrast, unclear]
- confidence: high / medium / low""",
}

# ─── Model pass configuration ────────────────────────────────────

PASS_CONFIG: dict[str, dict[str, str]] = {
    "qwen": {
        "model_key": "qwen3_vl_235b",
        "model_id": PAID_MODELS["qwen3_vl_235b"],
    },
    "glm": {
        "model_key": "glm_4_6v",
        "model_id": PAID_MODELS["glm_4_6v"],
    },
    "kimi": {
        "model_key": "kimi_k2_5",
        "model_id": PAID_MODELS["kimi_k2_5"],
    },
}


# ─── Image list construction ─────────────────────────────────────


def build_image_list(feature_meta: dict, cfg: LabelingConfig) -> list[str]:
    """Build base64 image list for VLM: OFF first, then ON ascending.

    Image ordering:
        [0..2]  OFF images — references where the feature is inactive
        [3..12] ON images — weakest activation first, strongest last
    """
    images_b64: list[str] = []

    # OFF images (3 references where feature is NOT active)
    for off in feature_meta["off_images"]:
        path = cfg.generated_dir / f"{off['image_id']:05d}.jpg"
        images_b64.append(encode_image_b64(path))

    # ON images in ascending activation (weakest first → strongest last)
    on_sorted = sorted(
        feature_meta["on_images"],
        key=lambda x: x["rank"],
        reverse=True,
    )
    for on in on_sorted:
        # For up.0.0, prefer cropped images that isolate the detail region
        crop_path = (
            cfg.feature_sets_dir
            / feature_meta["block"]
            / str(feature_meta["feature_id"])
            / f"crop_{on['rank']}.jpg"
        )
        if feature_meta["block"] == "up.0.0" and crop_path.exists():
            images_b64.append(encode_image_b64(crop_path))
        else:
            path = cfg.generated_dir / f"{on['image_id']:05d}.jpg"
            images_b64.append(encode_image_b64(path))

    return images_b64


# ─── Feature metadata loading ────────────────────────────────────


def load_feature_meta(
    block: str, feature_id: int, cfg: LabelingConfig,
) -> dict | None:
    """Load metadata.json for a single feature. Returns None if missing."""
    meta_path = cfg.feature_sets_dir / block / str(feature_id) / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


# ─── Disagreement detection for kimi pass ─────────────────────────


def load_pass_labels(
    model_key: str, block: str, cfg: LabelingConfig,
) -> dict[int, str]:
    """Load labels from a completed VLM pass as {feature_id: label}."""
    path = cfg.vlm_labels_dir / model_key / f"{block}.jsonl"
    if not path.exists():
        return {}
    labels: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            labels[entry["feature_id"]] = entry["label"]
    return labels


def find_disagreements(
    block: str, cfg: LabelingConfig, sim_threshold: float = 0.65,
) -> set[int]:
    """Find feature IDs where qwen and glm labels semantically disagree.

    Uses sentence embedding cosine similarity — NOT exact string matching,
    since two VLMs will almost never produce identical label strings even
    when they agree on the concept.
    """
    qwen = load_pass_labels("qwen3_vl_235b", block, cfg)
    glm = load_pass_labels("glm_4_6v", block, cfg)

    if not qwen or not glm:
        logger.warning(
            "Cannot compute disagreements for %s: qwen=%d, glm=%d",
            block, len(qwen), len(glm),
        )
        return set()

    from sentence_transformers import SentenceTransformer
    import numpy as np

    common_fids = sorted(set(qwen) & set(glm))
    if not common_fids:
        return set()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    qwen_texts = [qwen[fid] for fid in common_fids]
    glm_texts = [glm[fid] for fid in common_fids]

    # Batch encode for speed (5K labels in ~2 seconds on CPU)
    qwen_vecs = model.encode(qwen_texts, show_progress_bar=False, convert_to_numpy=True)
    glm_vecs = model.encode(glm_texts, show_progress_bar=False, convert_to_numpy=True)

    # Row-wise cosine similarity
    dots = np.sum(qwen_vecs * glm_vecs, axis=1)
    norms = np.linalg.norm(qwen_vecs, axis=1) * np.linalg.norm(glm_vecs, axis=1)
    sims = dots / np.maximum(norms, 1e-8)

    disagreements = {
        fid for fid, sim in zip(common_fids, sims)
        if sim < sim_threshold
    }

    n_agree = len(common_fids) - len(disagreements)
    logger.info(
        "%s: %d agree (sim >= %.2f), %d disagree → kimi tiebreaker",
        block, n_agree, sim_threshold, len(disagreements),
    )
    return disagreements


# ─── Resume support ──────────────────────────────────────────────


def load_existing_feature_ids(path: Path) -> set[int]:
    """Load feature IDs already annotated in a JSONL file."""
    if not path.exists():
        return set()
    ids: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["feature_id"])
    return ids


# ─── Core annotation ─────────────────────────────────────────────


async def annotate_block(
    block: str,
    model_key: str,
    model_id: str,
    cfg: LabelingConfig,
    feature_ids: list[int] | None = None,
) -> None:
    """Annotate all features in a block with one VLM model.

    Writes JSONL output with one line per feature. Supports resume.
    """
    out_dir = cfg.vlm_labels_dir / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{block}.jsonl"

    done_ids = load_existing_feature_ids(out_path)
    if done_ids:
        logger.info("Resuming %s/%s: %d already done", model_key, block, len(done_ids))

    if feature_ids is None:
        feature_ids = list(range(cfg.n_features))

    pending = [fid for fid in feature_ids if fid not in done_ids]
    total = len(pending)

    if total == 0:
        print(f"  {block}: all features already annotated, skipping")
        return

    print(f"  {block}: annotating {total} features with {model_key}")

    prompt = BLOCK_PROMPTS[block]
    buffer: list[str] = []

    async with OpenRouterClient(
        concurrency=cfg.vlm_concurrency,
        timeout=cfg.vlm_timeout,
        max_retries=cfg.vlm_max_retries,
    ) as client:

        async def _annotate_one(fid: int) -> str | None:
            meta = load_feature_meta(block, fid, cfg)
            if meta is None:
                return None

            try:
                images_b64 = build_image_list(meta, cfg)
            except FileNotFoundError as e:
                logger.warning("Missing image for %s/%d: %s", block, fid, e)
                return None

            try:
                resp: VLMResponse = await client.annotate(
                    model=model_id,
                    system_prompt=prompt,
                    images_b64=images_b64,
                )
            except Exception as e:
                logger.error("VLM error for %s/%d: %s", block, fid, e)
                return None

            entry = {
                "feature_id": fid,
                "label": resp.label,
                "category": resp.category,
                "confidence": resp.confidence,
                "raw": resp.raw,
            }
            return json.dumps(entry, ensure_ascii=False)

        # Process in chunks of 100 for progress + flush
        completed = 0
        for chunk_start in range(0, total, 100):
            chunk = pending[chunk_start : chunk_start + 100]
            tasks = [_annotate_one(fid) for fid in chunk]
            results = await asyncio.gather(*tasks)

            lines = [r for r in results if r is not None]
            buffer.extend(lines)
            completed += len(chunk)

            if buffer:
                with open(out_path, "a") as f:
                    for line in buffer:
                        f.write(line + "\n")
                buffer.clear()

            print(f"    {block}: {completed}/{total} ({completed * 100 // total}%)")

    print(f"  {block}: done -> {out_path}")


# ─── Pass orchestration ──────────────────────────────────────────


async def run_pass(
    pass_name: str,
    blocks: tuple[str, ...],
    cfg: LabelingConfig,
    limit: int | None = None,
) -> None:
    """Run a full annotation pass across specified blocks."""
    pass_cfg = PASS_CONFIG[pass_name]
    model_key = pass_cfg["model_key"]
    model_id = pass_cfg["model_id"]

    print(f"\n{'=' * 60}")
    print(f"  Stage 3: {pass_name} pass ({model_key})")
    if limit is not None:
        print(f"  DRY-RUN: capped at {limit} features per block")
    print(f"{'=' * 60}")

    for block in blocks:
        feature_ids: list[int] | None = None

        if pass_name == "kimi":
            disagreements = find_disagreements(block, cfg)
            if not disagreements:
                print(f"  {block}: no disagreements found, skipping")
                continue
            feature_ids = sorted(disagreements)
            print(f"  {block}: {len(feature_ids)} disagreements to resolve")

        if limit is not None:
            if feature_ids is None:
                feature_ids = list(range(cfg.n_features))
            feature_ids = feature_ids[:limit]

        await annotate_block(
            block=block,
            model_key=model_key,
            model_id=model_id,
            cfg=cfg,
            feature_ids=feature_ids,
        )

    print(f"\nPass '{pass_name}' complete. Labels in {cfg.vlm_labels_dir / model_key}")


# ─── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: VLM ensemble annotation via OpenRouter",
    )
    parser.add_argument(
        "--pass", dest="pass_name", required=True,
        choices=["qwen", "glm", "kimi"],
        help="Which VLM pass to run",
    )
    parser.add_argument(
        "--block", type=str, default=None,
        help="Single block to annotate (default: all 4)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap features per block (for cheap dry-runs before full commitment)",
    )
    args = parser.parse_args()

    cfg = LabelingConfig()
    cfg.ensure_dirs()
    blocks = (args.block,) if args.block else BLOCKS

    asyncio.run(run_pass(args.pass_name, blocks, cfg, limit=args.limit))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
