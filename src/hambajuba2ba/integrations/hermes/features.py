"""Feature catalog exploration for Hermes visual planning."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from functools import lru_cache
from typing import Any, Iterable, Protocol

from hambajuba2ba.integrations.hermes.contracts import (
    FeatureCandidate,
    FeatureEntry,
    FeatureSearchResponse,
)
from hambajuba2ba.integrations.hermes.embeddings import EmbeddingIndex
from hambajuba2ba.integrations.hermes.query_embedder import create_default_query_embedder

BLOCKS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_RESULTS = 50
DEFAULT_SAMPLE_COUNT = 8
NEAR_DUPLICATE_JACCARD = 0.86
LEXICAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


class QueryEmbedder(Protocol):
    def encode(self, text: str) -> Any: ...


def default_catalog_dir() -> Path:
    env_path = os.getenv("HAMBA_FEATURE_CATALOG_DIR")
    if env_path:
        return Path(env_path)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "frontend" / "public" / "data" / "features"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def _token_variants(token: str) -> tuple[str, ...]:
    variants = [token]
    if len(token) > 3 and token.endswith("s"):
        variants.append(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        variants.append(token[:-3])
    return tuple(dict.fromkeys(variants))


def _index_tokens(text: str) -> tuple[str, ...]:
    expanded: list[str] = []
    for token in tokenize(text):
        if token in LEXICAL_STOPWORDS:
            continue
        expanded.extend(_token_variants(token))
    return tuple(expanded)


def _stable_seed(*parts: object) -> int:
    seed_text = "\0".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _clamp_count(value: int, *, default: int = DEFAULT_SAMPLE_COUNT) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = default
    return max(1, min(count, MAX_RESULTS))


def _normalize_category(category: str | None) -> str:
    normalized = (category or "unknown").strip().lower()
    return normalized or "unknown"


def _normalize_label(label: str) -> str:
    return " ".join(tokenize(label))


def _confidence_score(confidence: str | None) -> float:
    return {"high": 1.0, "medium": 0.72, "low": 0.45}.get(
        (confidence or "").lower(), 0.35
    )


def _activation_score(mean_activation: float | None) -> float:
    if mean_activation is None:
        return 0.4
    return max(0.0, min(float(mean_activation) / 50.0, 1.0))


def _entry_quality(entry: FeatureEntry) -> float:
    return (0.72 * _confidence_score(entry.confidence)) + (
        0.28 * _activation_score(entry.mean_activation)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _round_score(value: float) -> float:
    return round(max(0.0, float(value)), 6)


@dataclass(frozen=True)
class _IndexedEntry:
    block: str
    entry: FeatureEntry
    tokens: tuple[str, ...]
    token_counts: Counter[str]
    label_key: str
    label_tokens: frozenset[str]
    quality: float

    @property
    def id(self) -> int:
        return self.entry.id

    @property
    def category(self) -> str:
        return _normalize_category(self.entry.category)


@dataclass(frozen=True)
class _BlockIndex:
    block: str
    entries: tuple[_IndexedEntry, ...]
    by_category: dict[str, tuple[_IndexedEntry, ...]]
    document_frequency: Counter[str]
    average_document_length: float

    @classmethod
    def build(cls, block: str, entries: Iterable[FeatureEntry]) -> "_BlockIndex":
        indexed_entries: list[_IndexedEntry] = []
        by_category: defaultdict[str, list[_IndexedEntry]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        total_tokens = 0

        for entry in entries:
            tokens = _index_tokens(f"{entry.label} {entry.category}")
            indexed = _IndexedEntry(
                block=block,
                entry=entry,
                tokens=tokens,
                token_counts=Counter(tokens),
                label_key=_normalize_label(entry.label),
                label_tokens=frozenset(tokenize(entry.label)),
                quality=_entry_quality(entry),
            )
            indexed_entries.append(indexed)
            by_category[indexed.category].append(indexed)
            document_frequency.update(set(tokens))
            total_tokens += len(tokens)

        frozen_categories = {
            category: tuple(items) for category, items in by_category.items()
        }
        average_length = total_tokens / max(len(indexed_entries), 1)
        return cls(
            block=block,
            entries=tuple(indexed_entries),
            by_category=frozen_categories,
            document_frequency=document_frequency,
            average_document_length=average_length,
        )


class FeatureCatalog:
    def __init__(
        self,
        entries: dict[str, tuple[FeatureEntry, ...]],
        *,
        embedding_index: EmbeddingIndex | None = None,
        query_embedder: QueryEmbedder | None = None,
    ):
        self._entries = entries
        self.explorer = FeatureExplorer(
            entries,
            embedding_index=embedding_index,
            query_embedder=query_embedder,
        )

    @classmethod
    def load(
        cls,
        catalog_dir: Path | None = None,
        *,
        embedding_index: EmbeddingIndex | None = None,
        query_embedder: QueryEmbedder | None = None,
        load_embeddings: bool = True,
    ) -> "FeatureCatalog":
        root = catalog_dir or default_catalog_dir()
        entries = {
            block: tuple(
                FeatureEntry.model_validate(item)
                for item in json.loads((root / f"{block}.json").read_text())
            )
            for block in BLOCKS
        }
        resolved_embedding_index = embedding_index
        if resolved_embedding_index is None and load_embeddings:
            embeddings_path = os.getenv("HAMBA_FEATURE_EMBEDDINGS_PATH")
            embedding_npz = (
                Path(embeddings_path)
                if embeddings_path
                else root / "feature_embeddings.npz"
            )
            manifest_path = (
                embedding_npz.with_suffix(".manifest.json")
                if embeddings_path
                else root / "feature_embeddings.manifest.json"
            )
            resolved_embedding_index = EmbeddingIndex.load_optional(
                embedding_npz,
                manifest_path=manifest_path,
                catalog_dir=root,
            )
        resolved_query_embedder = query_embedder
        if resolved_query_embedder is None and resolved_embedding_index is not None:
            resolved_query_embedder = create_default_query_embedder(
                model_name=resolved_embedding_index.model_name
            )
        return cls(
            entries,
            embedding_index=resolved_embedding_index,
            query_embedder=resolved_query_embedder,
        )

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def browse(
        self,
        block: str,
        *,
        category: str | None = None,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        seed: str | None = None,
        temperature: float = 0.6,
        avoid_feature_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        return self.explorer.browse(
            block=block,
            category=category,
            sample_count=sample_count,
            seed=seed,
            temperature=temperature,
            avoid_feature_ids=avoid_feature_ids,
        )

    def search(
        self,
        block: str,
        query: str,
        *,
        category: str | None = None,
        limit: int = 8,
        seed: str | None = None,
        temperature: float = 0.35,
        semantic: bool = True,
        avoid_feature_ids: list[int] | None = None,
    ) -> FeatureSearchResponse:
        details = self.search_details(
            block=block,
            query=query,
            category=category,
            limit=limit,
            seed=seed,
            temperature=temperature,
            semantic=semantic,
            avoid_feature_ids=avoid_feature_ids,
        )
        ranked = [
            FeatureCandidate(
                block=candidate["block"],
                id=candidate["id"],
                label=candidate["label"],
                category=candidate["category"],
                confidence=candidate.get("confidence"),
                mean_activation=candidate.get("mean_activation"),
                score=candidate["score"],
            )
            for candidate in details["candidates"]
        ]
        return FeatureSearchResponse(
            block=block,
            query=query,
            category=category,
            candidates=ranked,
        )

    def search_details(
        self,
        block: str,
        query: str,
        *,
        category: str | None = None,
        limit: int = 8,
        seed: str | None = None,
        temperature: float = 0.35,
        semantic: bool = True,
        avoid_feature_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        return self.explorer.search(
            block=block,
            query=query,
            category=category,
            limit=limit,
            seed=seed,
            temperature=temperature,
            semantic=semantic,
            avoid_feature_ids=avoid_feature_ids,
        )


class FeatureExplorer:
    """Indexed local feature explorer for browse, lexical, and optional semantic search."""

    def __init__(
        self,
        entries: dict[str, tuple[FeatureEntry, ...]],
        *,
        embedding_index: EmbeddingIndex | None = None,
        query_embedder: QueryEmbedder | None = None,
    ) -> None:
        self._indexes = {
            block: _BlockIndex.build(block, block_entries)
            for block, block_entries in entries.items()
        }
        self._embedding_index = embedding_index
        self._query_embedder = query_embedder

    def browse(
        self,
        block: str,
        *,
        category: str | None = None,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        seed: str | None = None,
        temperature: float = 0.6,
        avoid_feature_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        index = self._require_block(block)
        count = _clamp_count(sample_count)

        if category:
            category_entries = self._entries_for_category(index, category)
            samples = self._sample_entries(
                category_entries,
                limit=count,
                seed=seed,
                temperature=temperature,
                avoid_feature_ids=avoid_feature_ids,
                context=(block, "browse", category),
            )
            return {
                "block": block,
                "category": category,
                "count": len(category_entries),
                "seed": seed,
                "temperature": temperature,
                "samples": [self._sample_dict(item) for item in samples],
            }

        categories: dict[str, dict[str, Any]] = {}
        per_category_count = min(count, 3)
        for category_name, category_entries in sorted(
            index.by_category.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            samples = self._sample_entries(
                category_entries,
                limit=per_category_count,
                seed=seed,
                temperature=temperature,
                avoid_feature_ids=avoid_feature_ids,
                context=(block, "browse", category_name),
            )
            categories[category_name] = {
                "count": len(category_entries),
                "samples": [self._sample_dict(item) for item in samples],
            }

        return {
            "block": block,
            "total_features": len(index.entries),
            "seed": seed,
            "temperature": temperature,
            "categories": categories,
        }

    def search(
        self,
        block: str,
        query: str,
        *,
        category: str | None = None,
        limit: int = 12,
        seed: str | None = None,
        temperature: float = 0.35,
        semantic: bool = True,
        avoid_feature_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        index = self._require_block(block)
        if not query.strip() and not category:
            raise ValueError("Feature search needs a query or category")

        count = _clamp_count(limit, default=12)
        query_tokens = _index_tokens(query)
        category_entries = self._entries_for_category(index, category)
        semantic_scores, semantic_metadata = self._semantic_scores(
            block=block,
            query=query,
            enabled=semantic,
            limit=max(count * 12, 64),
        )
        avoid_ids = set(avoid_feature_ids or ())

        raw_candidates = []
        max_lexical = 0.0
        for indexed in category_entries:
            if indexed.id in avoid_ids:
                continue
            lexical_score = self._lexical_score(index, indexed, query, query_tokens)
            max_lexical = max(max_lexical, lexical_score)
            semantic_score = semantic_scores.get(indexed.id, 0.0)
            if query_tokens and lexical_score <= 0.0 and semantic_score <= 0.0:
                continue
            raw_candidates.append(
                {
                    "indexed": indexed,
                    "lexical_raw": lexical_score,
                    "semantic": semantic_score,
                }
            )

        scored_candidates = []
        for item in raw_candidates:
            lexical = item["lexical_raw"] / max_lexical if max_lexical else 0.0
            semantic_score = max(0.0, float(item["semantic"]))
            quality = item["indexed"].quality
            if query_tokens:
                score = (0.72 * lexical) + (0.2 * semantic_score) + (0.08 * quality)
            else:
                score = quality
            scored_candidates.append(
                {
                    "indexed": item["indexed"],
                    "score": score,
                    "scores": {
                        "lexical": _round_score(lexical),
                        "semantic": _round_score(semantic_score),
                        "quality": _round_score(quality),
                        "diversity": 0.0,
                    },
                }
            )

        selected = self._select_diverse(
            scored_candidates,
            limit=count,
            seed=seed,
            temperature=temperature,
            context=(block, "search", query, category),
        )
        return {
            "block": block,
            "query": query,
            "category": category,
            "seed": seed,
            "temperature": temperature,
            "retrieval": semantic_metadata,
            "candidates": [self._candidate_dict(item) for item in selected],
        }

    def _require_block(self, block: str) -> _BlockIndex:
        try:
            return self._indexes[block]
        except KeyError as exc:
            raise ValueError(f"Unknown SAE block: {block}") from exc

    def _entries_for_category(
        self,
        index: _BlockIndex,
        category: str | None,
    ) -> tuple[_IndexedEntry, ...]:
        if category is None:
            return index.entries
        requested = category.strip().lower()
        exact = index.by_category.get(requested)
        if exact is not None:
            return exact
        return tuple(
            indexed
            for indexed in index.entries
            if requested in indexed.entry.category.lower()
        )

    def _lexical_score(
        self,
        index: _BlockIndex,
        indexed: _IndexedEntry,
        query: str,
        query_tokens: Iterable[str],
    ) -> float:
        tokens = tuple(dict.fromkeys(query_tokens))
        if not tokens:
            return 0.0

        score = 0.0
        doc_len = max(len(indexed.tokens), 1)
        average_len = max(index.average_document_length, 1.0)
        total_docs = max(len(index.entries), 1)
        k1 = 1.5
        b = 0.75

        for token in tokens:
            frequency = indexed.token_counts.get(token, 0)
            if frequency <= 0:
                continue
            doc_frequency = index.document_frequency[token]
            idf = math.log(
                1.0 + ((total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            )
            denominator = frequency + k1 * (1.0 - b + b * (doc_len / average_len))
            score += idf * ((frequency * (k1 + 1.0)) / denominator)

        query_phrase = query.strip().lower()
        haystack = f"{indexed.entry.label} {indexed.entry.category}".lower()
        if query_phrase and query_phrase in haystack:
            score += 1.5
        return score

    def _semantic_scores(
        self,
        *,
        block: str,
        query: str,
        enabled: bool,
        limit: int,
    ) -> tuple[dict[int, float], dict[str, Any]]:
        metadata = {
            "lexical": True,
            "semantic": False,
            "embedding_model": None,
            "fallback": None,
        }
        if not enabled:
            metadata["fallback"] = "semantic disabled"
            return {}, metadata
        if self._embedding_index is None:
            metadata["fallback"] = "embedding artifact absent"
            return {}, metadata
        if self._query_embedder is None:
            metadata["embedding_model"] = self._embedding_index.model_name
            metadata["fallback"] = "query embedder absent"
            return {}, metadata
        if not query.strip():
            metadata["embedding_model"] = self._embedding_index.model_name
            metadata["fallback"] = "empty query"
            return {}, metadata

        try:
            query_vector = self._embed_query(query)
            hits = self._embedding_index.search(block, query_vector, limit=limit)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            metadata["embedding_model"] = self._embedding_index.model_name
            metadata["fallback"] = f"semantic unavailable: {exc.__class__.__name__}"
            return {}, metadata

        metadata["semantic"] = True
        metadata["embedding_model"] = self._embedding_index.model_name
        return {hit.feature_id: hit.score for hit in hits}, metadata

    def _embed_query(self, query: str) -> Any:
        embedder = self._query_embedder
        if embedder is None:
            raise RuntimeError("query embedder absent")
        if hasattr(embedder, "embed_query"):
            return embedder.embed_query(query)
        if hasattr(embedder, "encode"):
            return embedder.encode(query)
        if callable(embedder):
            return embedder(query)
        raise TypeError("query embedder must be callable or expose encode/embed_query")

    def _sample_entries(
        self,
        entries: tuple[_IndexedEntry, ...],
        *,
        limit: int,
        seed: str | None,
        temperature: float,
        avoid_feature_ids: list[int] | None,
        context: tuple[object, ...],
    ) -> list[_IndexedEntry]:
        avoid_ids = set(avoid_feature_ids or ())
        scored = [
            {"indexed": indexed, "score": indexed.quality, "scores": {}}
            for indexed in entries
            if indexed.id not in avoid_ids
        ]
        return [
            item["indexed"]
            for item in self._select_diverse(
                scored,
                limit=limit,
                seed=seed,
                temperature=temperature,
                context=context,
            )
        ]

    def _select_diverse(
        self,
        candidates: list[dict[str, Any]],
        *,
        limit: int,
        seed: str | None,
        temperature: float,
        context: tuple[object, ...],
    ) -> list[dict[str, Any]]:
        pool = sorted(
            candidates,
            key=lambda item: (-float(item["score"]), item["indexed"].id),
        )
        if not pool:
            return []

        rng = (
            random.Random(_stable_seed(seed, *context))
            if seed is not None
            else random.Random()
        )
        selected: list[dict[str, Any]] = []
        label_keys: set[str] = set()
        label_token_sets: list[frozenset[str]] = []
        target_count = _clamp_count(limit)

        while pool and len(selected) < target_count:
            index = 0
            if temperature > 0.0 and len(pool) > 1:
                index = self._weighted_index(pool, rng=rng, temperature=temperature)
            item = pool.pop(index)
            indexed: _IndexedEntry = item["indexed"]
            if indexed.label_key in label_keys:
                continue
            if any(
                _jaccard(indexed.label_tokens, existing) >= NEAR_DUPLICATE_JACCARD
                for existing in label_token_sets
            ):
                continue
            label_keys.add(indexed.label_key)
            label_token_sets.append(indexed.label_tokens)
            item["scores"]["diversity"] = _round_score(
                1.0 - (len(selected) / target_count)
            )
            selected.append(item)

        return selected

    def _weighted_index(
        self,
        pool: list[dict[str, Any]],
        *,
        rng: random.Random,
        temperature: float,
    ) -> int:
        max_score = max(float(item["score"]) for item in pool)
        denominator = max(float(temperature), 0.05)
        weights = [
            math.exp((float(item["score"]) - max_score) / denominator) for item in pool
        ]
        total = sum(weights)
        threshold = rng.random() * total
        running = 0.0
        for index, weight in enumerate(weights):
            running += weight
            if running >= threshold:
                return index
        return len(pool) - 1

    @staticmethod
    def _sample_dict(indexed: _IndexedEntry) -> dict[str, Any]:
        return {
            "id": indexed.id,
            "label": indexed.entry.label,
            "category": indexed.entry.category,
            "confidence": indexed.entry.confidence,
            "mean_activation": indexed.entry.mean_activation,
        }

    @staticmethod
    def _candidate_dict(item: dict[str, Any]) -> dict[str, Any]:
        indexed: _IndexedEntry = item["indexed"]
        return {
            "block": indexed.block,
            "id": indexed.id,
            "label": indexed.entry.label,
            "category": indexed.entry.category,
            "confidence": indexed.entry.confidence,
            "mean_activation": indexed.entry.mean_activation,
            "score": _round_score(item["score"]),
            "scores": item["scores"],
        }


@lru_cache(maxsize=1)
def get_default_catalog() -> FeatureCatalog:
    return FeatureCatalog.load()
