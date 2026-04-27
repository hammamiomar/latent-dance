"""Transform SAE labels into compact frontend-friendly JSON.

Reads final labels from data/labeling/sae_labels/final/{block}.json and outputs
compact arrays to frontend/public/data/features/{block}.json.

Strips VLM provenance, TF-IDF terms, and other pipeline metadata.
Normalizes malformed categories from VLM artifacts.

Usage:
    uv run python scripts/labeling/transform_labels_for_frontend.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "labeling" / "sae_labels" / "final"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data" / "features"
BLOCKS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

# Normalize malformed categories from VLM artifacts
CATEGORY_FIXES: dict[str, str] = {
    "** lighting": "lighting",
    "setting/environment": "setting",
}


def transform_block(block: str) -> list[dict]:
    """Read final labels for one block, output compact format."""
    input_path = INPUT_DIR / f"{block}.json"
    with open(input_path) as f:
        features = json.load(f)

    compact = []
    for feat in features:
        category = feat.get("category", "unknown")
        category = CATEGORY_FIXES.get(category, category)

        compact.append({
            "id": feat["feature_id"],
            "label": feat.get("label", "unlabeled"),
            "category": category,
            "confidence": feat.get("confidence", "low"),
            "meanActivation": round(feat.get("mean_activation", 0), 2),
        })

    # Sort by feature ID for consistent ordering
    compact.sort(key=lambda x: x["id"])
    return compact


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for block in BLOCKS:
        features = transform_block(block)
        output_path = OUTPUT_DIR / f"{block}.json"
        with open(output_path, "w") as f:
            json.dump(features, f, separators=(",", ":"))

        size_kb = output_path.stat().st_size / 1024
        n_high = sum(1 for f in features if f["confidence"] == "high")
        print(f"{block}: {len(features)} features, {n_high} high-confidence, {size_kb:.0f} KB")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
