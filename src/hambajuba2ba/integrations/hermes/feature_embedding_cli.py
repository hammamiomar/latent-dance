"""Offline builder for Hermes feature embedding artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from hambajuba2ba.integrations.hermes.embeddings import safe_block_key, source_hash
from hambajuba2ba.integrations.hermes.features import BLOCKS, default_catalog_dir
from hambajuba2ba.integrations.hermes.query_embedder import (
    DEFAULT_QUERY_EMBEDDING_MODEL,
)

DEFAULT_MODEL = DEFAULT_QUERY_EMBEDDING_MODEL


def build_feature_embeddings(
    *,
    catalog_dir: Path,
    output_path: Path,
    manifest_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    dtype: str = "float16",
    blocks: tuple[str, ...] = BLOCKS,
) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Feature embedding generation requires sentence-transformers. "
            "Install with `uv sync --extra hermes-semantic` or use the existing "
            "`labeling` extra before running this CLI."
        ) from exc

    model = SentenceTransformer(model_name)
    arrays: dict[str, np.ndarray] = {}
    source_hashes: dict[str, str] = {}
    dimensions: set[int] = set()

    for block in blocks:
        labels, feature_ids = _load_block_labels(catalog_dir, block)
        vectors = model.encode(
            labels,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        dimensions.add(int(vectors.shape[1]))
        if dtype == "float16":
            vectors = vectors.astype(np.float16)
        elif dtype == "float32":
            vectors = vectors.astype(np.float32)
        else:
            raise ValueError("dtype must be float16 or float32")

        safe = safe_block_key(block)
        arrays[f"{safe}_ids"] = np.asarray(feature_ids, dtype=np.int32)
        arrays[f"{safe}_vectors"] = vectors
        source_hashes[block] = source_hash(catalog_dir / f"{block}.json")

    if len(dimensions) != 1:
        raise ValueError(f"Expected one embedding dimension, got {sorted(dimensions)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    manifest = {
        "format_version": 1,
        "model": model_name,
        "dimension": dimensions.pop(),
        "dtype": dtype,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blocks": list(blocks),
        "source_hashes": source_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build precomputed Hermes SAE feature label embeddings."
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=default_catalog_dir(),
        help="Directory containing frontend/public/data/features/{block}.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Defaults to feature_embeddings.npz in catalog-dir.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest path. Defaults to feature_embeddings.manifest.json.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--block",
        action="append",
        choices=BLOCKS,
        dest="blocks",
        help="Block to include. Repeat to build a subset. Defaults to all blocks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or args.catalog_dir / "feature_embeddings.npz"
    manifest = args.manifest or args.catalog_dir / "feature_embeddings.manifest.json"
    build_feature_embeddings(
        catalog_dir=args.catalog_dir,
        output_path=output,
        manifest_path=manifest,
        model_name=args.model,
        batch_size=args.batch_size,
        dtype=args.dtype,
        blocks=tuple(args.blocks or BLOCKS),
    )


def _load_block_labels(catalog_dir: Path, block: str) -> tuple[list[str], list[int]]:
    raw_entries: list[dict[str, Any]] = json.loads(
        (catalog_dir / f"{block}.json").read_text()
    )
    return (
        [str(entry["label"]) for entry in raw_entries],
        [int(entry["id"]) for entry in raw_entries],
    )


if __name__ == "__main__":
    main()
