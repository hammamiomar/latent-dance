"""Optional precomputed feature embedding index for Hermes retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BLOCKS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


@dataclass(frozen=True)
class EmbeddingHit:
    feature_id: int
    score: float


@dataclass(frozen=True)
class _EmbeddingBlock:
    feature_ids: np.ndarray
    vectors: np.ndarray


class EmbeddingIndex:
    def __init__(
        self,
        blocks: dict[str, _EmbeddingBlock],
        *,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        if not blocks:
            raise ValueError("Embedding index has no blocks")
        self._blocks = blocks
        self.manifest = manifest or {}
        self.model_name = (
            self.manifest.get("model")
            or self.manifest.get("model_name")
            or self.manifest.get("embedding_model")
        )

    @classmethod
    def load(
        cls,
        npz_path: Path,
        *,
        manifest_path: Path | None = None,
        catalog_dir: Path | None = None,
        validate_source_hashes: bool = True,
    ) -> "EmbeddingIndex":
        manifest = _read_manifest(manifest_path)
        if validate_source_hashes and catalog_dir is not None:
            _validate_source_hashes(manifest, catalog_dir)

        with np.load(npz_path) as artifact:
            blocks: dict[str, _EmbeddingBlock] = {}
            for block in BLOCKS:
                safe = safe_block_key(block)
                ids_key = f"{safe}_ids"
                vectors_key = f"{safe}_vectors"
                if ids_key not in artifact.files or vectors_key not in artifact.files:
                    continue
                feature_ids = np.asarray(artifact[ids_key], dtype=np.int32)
                vectors = np.asarray(artifact[vectors_key], dtype=np.float32)
                blocks[block] = _EmbeddingBlock(
                    feature_ids=_validate_ids(feature_ids, block),
                    vectors=_normalize_vectors(
                        _validate_vectors(vectors, feature_ids, block)
                    ),
                )

        return cls(blocks, manifest=manifest)

    @classmethod
    def load_optional(
        cls,
        npz_path: Path,
        *,
        manifest_path: Path | None = None,
        catalog_dir: Path | None = None,
    ) -> "EmbeddingIndex | None":
        if not npz_path.exists():
            return None
        try:
            return cls.load(
                npz_path,
                manifest_path=manifest_path,
                catalog_dir=catalog_dir,
                validate_source_hashes=True,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def search(
        self,
        block: str,
        query_vector: Any,
        *,
        limit: int = 64,
    ) -> list[EmbeddingHit]:
        try:
            embedding_block = self._blocks[block]
        except KeyError as exc:
            raise ValueError(f"No embedding vectors for SAE block: {block}") from exc

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1:
            raise ValueError("Query vector must be one-dimensional")
        if query.shape[0] != embedding_block.vectors.shape[1]:
            raise ValueError(
                "Query vector dimension does not match embedding index "
                f"({query.shape[0]} != {embedding_block.vectors.shape[1]})"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm

        scores = embedding_block.vectors @ query
        count = max(1, min(int(limit), len(scores)))
        order = np.argsort(-scores, kind="stable")[:count]
        return [
            EmbeddingHit(
                feature_id=int(embedding_block.feature_ids[index]),
                score=float(scores[index]),
            )
            for index in order
        ]


def safe_block_key(block: str) -> str:
    return block.replace(".", "_")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _validate_source_hashes(manifest: dict[str, Any], catalog_dir: Path) -> None:
    source_hashes = manifest.get("source_hashes") or manifest.get("sourceHashes") or {}
    if not source_hashes:
        raise ValueError("Embedding manifest missing source_hashes")
    for block, expected in source_hashes.items():
        path = catalog_dir / f"{block}.json"
        if not path.exists():
            raise ValueError(f"Cannot validate missing catalog file: {path}")
        if source_hash(path) != expected:
            raise ValueError(f"Embedding source hash mismatch for {block}")


def _validate_ids(feature_ids: np.ndarray, block: str) -> np.ndarray:
    if feature_ids.ndim != 1:
        raise ValueError(f"Embedding IDs for {block} must be one-dimensional")
    if np.any(feature_ids < 0) or np.any(feature_ids > 5119):
        raise ValueError(f"Embedding IDs for {block} must be in [0, 5119]")
    if len(np.unique(feature_ids)) != len(feature_ids):
        raise ValueError(f"Embedding IDs for {block} must be unique")
    return feature_ids


def _validate_vectors(
    vectors: np.ndarray,
    feature_ids: np.ndarray,
    block: str,
) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError(f"Embedding vectors for {block} must be two-dimensional")
    if vectors.shape[0] != feature_ids.shape[0]:
        raise ValueError(
            f"Embedding vector count for {block} does not match feature IDs"
        )
    return vectors


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms
