"""Stage 5: Label Fusion — merge VLM labels + supplements into final labels.

Combines signals from Stages 3 (VLM ensemble) and 4 (zero-cost supplements)
into a single label per feature with confidence scoring.

Fusion rules (priority order):
    1. 2+ VLMs agree (cosine sim > threshold): accept, confidence=high
    2. VLMs disagree but TF-IDF aligns with one: accept aligned, confidence=medium
    3. All disagree: best-guess from strongest model, confidence=low
    4. mid.0 spatial features: structural label overrides VLM output

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim, CPU) for
semantic similarity between VLM labels.

Usage:
    uv run python -m hambajuba2ba.labeling.stage5_fuse
    uv run python -m hambajuba2ba.labeling.stage5_fuse --block down.2.1
    uv run python -m hambajuba2ba.labeling.stage5_fuse --calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from hambajuba2ba.labeling.config import (
    KNOWN_FEATURES,
    LabelingConfig,
)
from hambajuba2ba.labeling.utils import load_activation_stats

logger = logging.getLogger(__name__)

# VLM model keys in preference order (strongest first for tie-breaking)
VLM_MODEL_KEYS: tuple[str, ...] = ("kimi_k2_5", "qwen3_vl_235b", "glm_4_6v")

# Spatial patterns that indicate a structural feature (mid.0)
_STRUCTURAL_PATTERNS = frozenset({
    "center", "top", "bottom", "left", "right",
    "top-left", "top-right", "bottom-left", "bottom-right",
    "horizontal-band", "vertical-band", "border", "corners",
})


# ─── Data structures ─────────────────────────────────────────────


@dataclass
class VLMLabel:
    """Parsed VLM label from a JSONL file."""
    feature_id: int
    label: str
    category: str
    confidence: str
    raw: str


@dataclass
class SupplementData:
    """Zero-cost supplement data for one feature."""
    feature_id: int
    tfidf_top_terms: list[str]
    spatial_pattern: str | None


# ─── I/O helpers ──────────────────────────────────────────────────


def load_vlm_labels(
    vlm_labels_dir: Path, model_key: str, block: str,
) -> dict[int, VLMLabel]:
    """Load VLM labels from JSONL. Returns empty dict if missing."""
    path = vlm_labels_dir / model_key / f"{block}.jsonl"
    if not path.exists():
        return {}
    labels: dict[int, VLMLabel] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            labels[data["feature_id"]] = VLMLabel(
                feature_id=data["feature_id"],
                label=data.get("label", ""),
                category=data.get("category", "unknown"),
                confidence=data.get("confidence", "low"),
                raw=data.get("raw", ""),
            )
    return labels


def load_supplements(
    supplement_dir: Path, block: str,
) -> dict[int, SupplementData]:
    """Load zero-cost supplement data for a block."""
    path = supplement_dir / f"{block}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {
        e["feature_id"]: SupplementData(
            feature_id=e["feature_id"],
            tfidf_top_terms=e.get("tfidf_top_terms", []),
            spatial_pattern=e.get("spatial_pattern"),
        )
        for e in entries
    }


# ─── Sentence embedding similarity ───────────────────────────────


class LabelEmbedder:
    """Sentence-transformer wrapper for comparing VLM labels semantically."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.model.encode(
                text, show_progress_bar=False, convert_to_numpy=True,
            )
        return self._cache[text]

    def cosine_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        vec_a = self.encode(a)
        vec_b = self.encode(b)
        dot = float(np.dot(vec_a, vec_b))
        norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        return dot / norm if norm > 0 else 0.0

    def clear_cache(self) -> None:
        self._cache.clear()


# ─── Fusion logic ─────────────────────────────────────────────────


def _tfidf_aligns_with_label(
    tfidf_terms: list[str], vlm_label: str, top_n: int = 5,
) -> bool:
    """Check if TF-IDF top terms overlap with a VLM label's words."""
    if not tfidf_terms or not vlm_label:
        return False
    label_lower = vlm_label.lower()
    label_words = set(label_lower.split())
    for term in (t.lower() for t in tfidf_terms[:top_n]):
        if term in label_words or term in label_lower:
            return True
        for word in label_words:
            if word in term:
                return True
    return False


def fuse_feature(
    feature_id: int,
    block: str,
    vlm_labels: dict[str, VLMLabel | None],
    supplement: SupplementData | None,
    embedder: LabelEmbedder,
    threshold: float,
    mean_act: float,
    cv: float,
) -> dict:
    """Apply fusion rules to produce a single label for one feature."""
    available: dict[str, str] = {}
    available_cats: dict[str, str] = {}
    for key in VLM_MODEL_KEYS:
        vlm = vlm_labels.get(key)
        if vlm is not None and vlm.label:
            available[key] = vlm.label
            available_cats[key] = vlm.category

    tfidf_terms = supplement.tfidf_top_terms if supplement else []
    spatial = supplement.spatial_pattern if supplement else None

    vlm_out: dict[str, str | None] = {
        key: vlm_labels[key].label if vlm_labels.get(key) else None
        for key in VLM_MODEL_KEYS
    }

    base = {
        "block": block,
        "feature_id": feature_id,
        "vlm_labels": vlm_out,
        "tfidf_top5": tfidf_terms[:5],
        "spatial_pattern": spatial,
        "mean_activation": round(mean_act, 2),
        "activation_cv": round(cv, 4),
        "n_activating_images": 0,
    }

    # Rule 4 (first): mid.0 spatial override
    if block == "mid.0" and spatial and spatial.strip().lower() in _STRUCTURAL_PATTERNS:
        return {**base, "label": f"spatial: {spatial}", "category": "spatial",
                "confidence": "high", "method": "spatial_analysis"}

    if not available:
        return {**base, "label": "unlabeled", "category": "unknown",
                "confidence": "low", "method": "no_vlm_data"}

    # Rule 1: pairwise VLM consensus
    keys = list(available.keys())
    agreeing_pairs = [
        (k1, k2, embedder.cosine_similarity(available[k1], available[k2]))
        for k1, k2 in combinations(keys, 2)
    ]
    agreeing_pairs = [(k1, k2, s) for k1, k2, s in agreeing_pairs if s > threshold]

    if agreeing_pairs:
        agreeing_pairs.sort(key=lambda x: -x[2])
        best_k1, best_k2, _ = agreeing_pairs[0]

        n_agreeing = len(set().union(*(
            {k1, k2} for k1, k2, _ in agreeing_pairs
        )))
        method = "vlm_consensus_3way" if n_agreeing >= 3 and len(keys) >= 3 else "vlm_consensus_2way"

        # Pick label from stronger model in the pair
        chosen_key = best_k1
        for preferred in VLM_MODEL_KEYS:
            if preferred in (best_k1, best_k2):
                chosen_key = preferred
                break

        return {**base, "label": available[chosen_key],
                "category": available_cats.get(chosen_key, "unknown"),
                "confidence": "high", "method": method}

    # Rule 2: TF-IDF alignment
    if tfidf_terms:
        for preferred in VLM_MODEL_KEYS:
            if preferred in available and _tfidf_aligns_with_label(tfidf_terms, available[preferred]):
                return {**base, "label": available[preferred],
                        "category": available_cats.get(preferred, "unknown"),
                        "confidence": "medium", "method": "vlm_tfidf_aligned"}

    # Rule 3: best guess from strongest model
    for preferred in VLM_MODEL_KEYS:
        if preferred in available:
            return {**base, "label": available[preferred],
                    "category": available_cats.get(preferred, "unknown"),
                    "confidence": "low", "method": "best_guess"}

    return {**base, "label": "unlabeled", "category": "unknown",
            "confidence": "low", "method": "fallback"}


# ─── Block-level fusion ──────────────────────────────────────────


def fuse_block(
    block: str, cfg: LabelingConfig, embedder: LabelEmbedder,
    threshold: float | None = None,
) -> list[dict]:
    """Fuse all features in one block. Returns list of dicts for JSON."""
    threshold = threshold if threshold is not None else cfg.consensus_threshold

    vlm_by_model: dict[str, dict[int, VLMLabel]] = {
        key: load_vlm_labels(cfg.vlm_labels_dir, key, block)
        for key in VLM_MODEL_KEYS
    }
    supplements = load_supplements(cfg.supplement_dir, block)
    mean_acts, std_acts = load_activation_stats(cfg.weight_path(block))

    results: list[dict] = []
    for fid in range(cfg.n_features):
        vlm_for_feature = {key: vlm_by_model[key].get(fid) for key in VLM_MODEL_KEYS}
        supplement = supplements.get(fid)
        mean_act = float(mean_acts[fid])
        std_act = float(std_acts[fid])
        cv = std_act / mean_act if mean_act > 0 else 0.0

        results.append(fuse_feature(
            fid, block, vlm_for_feature, supplement, embedder,
            threshold, mean_act, cv,
        ))

    return results


# ─── Threshold calibration ───────────────────────────────────────


def calibrate_threshold(
    cfg: LabelingConfig, embedder: LabelEmbedder,
    thresholds: list[float] | None = None,
) -> float:
    """Run fusion on ground-truth features at multiple thresholds."""
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'=' * 60}")
    print("  Threshold Calibration")
    print(f"{'=' * 60}")

    best_threshold = cfg.consensus_threshold
    best_accuracy = 0.0

    for thresh in thresholds:
        n_correct = 0
        n_total = 0

        for block, known_list in KNOWN_FEATURES.items():
            vlm_by_model = {
                key: load_vlm_labels(cfg.vlm_labels_dir, key, block)
                for key in VLM_MODEL_KEYS
            }
            supplements = load_supplements(cfg.supplement_dir, block)
            mean_acts, std_acts = load_activation_stats(cfg.weight_path(block))

            for known in known_list:
                fid = known["id"]
                expected = known["label"]
                vlm_for_feature = {key: vlm_by_model[key].get(fid) for key in VLM_MODEL_KEYS}
                supplement = supplements.get(fid)
                mean_act = float(mean_acts[fid])
                std_act = float(std_acts[fid])
                cv = std_act / mean_act if mean_act > 0 else 0.0

                fused = fuse_feature(
                    fid, block, vlm_for_feature, supplement, embedder,
                    thresh, mean_act, cv,
                )

                pred = fused["label"].lower().strip()
                exp = expected.lower().strip()
                also_accept = known.get("also_accept", [])
                keyword_hit = any(kw.lower() in pred for kw in also_accept)
                if (exp in pred or pred in exp or keyword_hit
                        or embedder.cosine_similarity(pred, exp) > 0.6):
                    n_correct += 1
                n_total += 1

        acc = n_correct / n_total if n_total > 0 else 0.0
        print(f"  threshold={thresh:.1f}  accuracy={n_correct}/{n_total} ({acc:.1%})")
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thresh

    print(f"\n  Best: {best_threshold:.1f} ({best_accuracy:.1%})")
    return best_threshold


# ─── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Fuse VLM labels + supplements into final labels",
    )
    parser.add_argument("--block", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = LabelingConfig()
    cfg.ensure_dirs()

    print("Loading sentence-transformer...")
    embedder = LabelEmbedder(cfg.embedding_model)

    if args.calibrate:
        best = calibrate_threshold(cfg, embedder)
        if args.threshold is None:
            args.threshold = best

    threshold = args.threshold if args.threshold is not None else cfg.consensus_threshold
    blocks = (args.block,) if args.block else cfg.blocks

    for block in blocks:
        print(f"\n{'=' * 60}")
        print(f"  Fusing {block}")
        print(f"{'=' * 60}")

        results = fuse_block(block, cfg, embedder, threshold)

        by_conf: dict[str, int] = {}
        by_method: dict[str, int] = {}
        for r in results:
            by_conf[r["confidence"]] = by_conf.get(r["confidence"], 0) + 1
            by_method[r["method"]] = by_method.get(r["method"], 0) + 1

        print(f"  Total: {len(results)}")
        for conf in ("high", "medium", "low"):
            c = by_conf.get(conf, 0)
            print(f"    {conf}: {c} ({c / len(results):.1%})")
        for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
            print(f"    {method}: {count}")

        out_path = cfg.final_dir / f"{block}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {out_path}")

        embedder.clear_cache()

    print(f"\nDone. Final labels in {cfg.final_dir}")


if __name__ == "__main__":
    main()
