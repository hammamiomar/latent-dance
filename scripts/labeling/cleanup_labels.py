"""Clean up LLM artifact labels by falling back to qwen3_vl_235b.

Fixes two issues:
  1. kimi_k2_5 leaked chain-of-thought into labels (~2,566 across all blocks)
  2. mid.0 spatial analysis was too aggressive, discarding semantic labels (~4,771)

For each corrupted label, substitutes qwen3's label/category/confidence
(which was 100% clean across all blocks).

Writes cleaned versions to both:
  - data/labeling/sae_labels/final/{block}.json  (in-place)
  - frontend/public/data/features/{block}.json  (compact frontend format)

Usage:
    python scripts/labeling/cleanup_labels.py [--dry-run]
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "labeling" / "sae_labels"
FINAL_DIR = DATA_ROOT / "final"
QWEN3_DIR = DATA_ROOT / "vlm_labels" / "qwen3_vl_235b"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "public" / "data" / "features"
BLOCKS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

LLM_ARTIFACT = re.compile(
    r"^(Looking at|I need to|Let me|I'll |Based on|Analyzing|The (image|feature))",
    re.IGNORECASE,
)

CATEGORY_FIXES: dict[str, str] = {
    "** lighting": "lighting",
    "setting/environment": "setting",
}


def load_qwen3_index(block: str) -> dict[int, dict]:
    path = QWEN3_DIR / f"{block}.jsonl"
    index = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            index[entry["feature_id"]] = entry
    return index


def is_bad_label(feat: dict, block: str) -> bool:
    label = feat.get("label", "")
    if LLM_ARTIFACT.search(label):
        return True
    if block == "mid.0" and label.startswith("spatial:"):
        return True
    if not label or label == "unlabeled":
        return True
    return False


def cleanup_block(block: str, dry_run: bool = False) -> dict:
    final_path = FINAL_DIR / f"{block}.json"
    with open(final_path) as f:
        features = json.load(f)

    qwen3 = load_qwen3_index(block)

    stats = {"total": len(features), "replaced": 0, "missing_qwen3": 0}
    reasons: dict[str, int] = {}

    for feat in features:
        if not is_bad_label(feat, block):
            continue

        fid = feat["feature_id"]
        q = qwen3.get(fid)

        if not q or not q.get("label"):
            stats["missing_qwen3"] += 1
            continue

        old_label = feat["label"]
        reason = "spatial" if old_label.startswith("spatial:") else "llm_artifact"
        reasons[reason] = reasons.get(reason, 0) + 1

        feat["label"] = q["label"]
        feat["category"] = q.get("category", feat.get("category", "unknown"))
        feat["confidence"] = q.get("confidence", "medium")
        feat["method"] = f"qwen3_fallback (was: {feat.get('method', '?')})"
        stats["replaced"] += 1

    stats["reasons"] = reasons

    if not dry_run:
        with open(final_path, "w") as f:
            json.dump(features, f, indent=2)

        FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
        compact = []
        for feat in features:
            cat = feat.get("category", "unknown")
            cat = CATEGORY_FIXES.get(cat, cat)
            compact.append({
                "id": feat["feature_id"],
                "label": feat.get("label", "unlabeled"),
                "category": cat,
                "confidence": feat.get("confidence", "low"),
                "meanActivation": round(feat.get("mean_activation", 0), 2),
            })
        compact.sort(key=lambda x: x["id"])

        out_path = FRONTEND_DIR / f"{block}.json"
        with open(out_path, "w") as f:
            json.dump(compact, f, separators=(",", ":"))

    return stats


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN — no files will be modified\n")

    if not FINAL_DIR.exists():
        print(f"Error: {FINAL_DIR} not found")
        print("Run from the repo root where data/labeling/ exists.")
        sys.exit(1)

    total_replaced = 0
    for block in BLOCKS:
        stats = cleanup_block(block, dry_run=dry_run)
        total_replaced += stats["replaced"]

        reasons_str = ", ".join(f"{k}={v}" for k, v in stats["reasons"].items())
        print(
            f"{block}: {stats['replaced']}/{stats['total']} replaced "
            f"({reasons_str})"
        )
        if stats["missing_qwen3"]:
            print(f"  warning: {stats['missing_qwen3']} features missing qwen3 label")

    print(f"\nTotal: {total_replaced} labels cleaned up")
    if dry_run:
        print("(dry run — run without --dry-run to apply)")


if __name__ == "__main__":
    main()
