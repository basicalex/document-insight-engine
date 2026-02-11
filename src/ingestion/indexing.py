from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from src.config.settings import Settings


class EmbeddingTier(str, Enum):
    TIER1 = "tier1"
    TIER4 = "tier4"


@dataclass(frozen=True)
class EmbeddingProfile:
    tier: EmbeddingTier
    index_name: str
    provider: str
    model_name: str
    dimension: int


@dataclass(frozen=True)
class IndexSchema:
    index_name: str
    dimension: int
    distance_metric: str
    fields: dict[str, str]


@dataclass(frozen=True)
class IndexRecord:
    record_id: str
    document_id: str
    chunk_id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_chunk_id: str | None = None
    section_path: tuple[str, ...] = ()
    page_refs: list[int] = field(default_factory=list)
    embedding_version: str = "v1"


@dataclass(frozen=True)
class QueryMatch:
    record_id: str
    score: float
    payload: dict[str, Any]


class IndexingError(Exception):
    pass


class IndexBackend(Protocol):
    def ensure_index(self, schema: IndexSchema) -> None: ...

    def upsert(
        self,
        index_name: str,
        record_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None: ...

    def query(
        self,
        index_name: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[QueryMatch]: ...


class HybridVectorIndexStore:
    def __init__(
        self,
        cfg: Settings,
        backend: IndexBackend | None = None,
        tier1_dimension: int = 384,
        tier4_dimension: int = 3072,
    ) -> None:
        self.cfg = cfg
        self.backend = backend or RedisVLIndexBackend(redis_url=cfg.redis_url)
        self._profiles = {
            EmbeddingTier.TIER1: EmbeddingProfile(
                tier=EmbeddingTier.TIER1,
                index_name=cfg.redis_index_tier1,
                provider="local",
                model_name=cfg.local_embedding_model,
                dimension=tier1_dimension,
            ),
            EmbeddingTier.TIER4: EmbeddingProfile(
                tier=EmbeddingTier.TIER4,
                index_name=cfg.redis_index_tier4,
                provider="cloud",
                model_name=cfg.cloud_embedding_model,
                dimension=tier4_dimension,
            ),
        }

    def bootstrap_indices(self) -> None:
        for profile in self._profiles.values():
            self.backend.ensure_index(
                IndexSchema(
                    index_name=profile.index_name,
                    dimension=profile.dimension,
                    distance_metric="cosine",
                    fields={
                        "record_id": "tag",
                        "document_id": "tag",
                        "chunk_id": "tag",
                        "parent_chunk_id": "tag",
                        "text": "text",
                        "section_path": "tag",
                        "page_refs": "tag",
                        "tags": "tag",
                        "metadata": "text",
                        "embedding_provider": "tag",
                        "embedding_model": "tag",
                        "embedding_version": "tag",
                        "tier": "tag",
                    },
                )
            )

    def persist_records(self, tier: EmbeddingTier, records: list[IndexRecord]) -> None:
        profile = self._profiles[tier]
        for record in records:
            self._validate_vector_dimension(
                vector=record.vector,
                expected_dimension=profile.dimension,
                context=f"record {record.record_id}",
            )
            payload = {
                "record_id": record.record_id,
                "document_id": record.document_id,
                "chunk_id": record.chunk_id,
                "parent_chunk_id": record.parent_chunk_id,
                "text": record.text,
                "section_path": "/".join(record.section_path),
                "page_refs": [str(ref) for ref in record.page_refs],
                "tags": record.tags,
                "metadata": record.metadata,
                "embedding_provider": profile.provider,
                "embedding_model": profile.model_name,
                "embedding_version": record.embedding_version,
                "tier": profile.tier.value,
            }
            self.backend.upsert(
                index_name=profile.index_name,
                record_id=record.record_id,
                vector=record.vector,
                payload=payload,
            )

    def query(
        self,
        tier: EmbeddingTier,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryMatch]:
        if top_k <= 0:
            raise IndexingError("top_k must be positive")

        profile = self._profiles[tier]
        self._validate_vector_dimension(
            vector=vector,
            expected_dimension=profile.dimension,
            context="query vector",
        )
        return self.backend.query(
            index_name=profile.index_name,
            vector=vector,
            top_k=top_k,
            filters=filters,
        )

    def _validate_vector_dimension(
        self,
        vector: list[float],
        expected_dimension: int,
        context: str,
    ) -> None:
        if len(vector) != expected_dimension:
            raise IndexingError(
                f"{context} dimension {len(vector)} does not match expected {expected_dimension}"
            )


class RedisVLIndexBackend:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._indices: dict[str, Any] = {}

    def ensure_index(self, schema: IndexSchema) -> None:
        if schema.index_name in self._indices:
            return

        try:
            from redisvl.index import SearchIndex
            from redisvl.schema import IndexSchema as RedisVLIndexSchema
        except ModuleNotFoundError as exc:
            raise IndexingError(
                "RedisVL backend requires the 'redisvl' package to be installed"
            ) from exc

        redis_schema = RedisVLIndexSchema.from_dict(
            {
                "index": {
                    "name": schema.index_name,
                    "prefix": f"{schema.index_name}:",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "record_id", "type": "tag"},
                    {"name": "document_id", "type": "tag"},
                    {"name": "chunk_id", "type": "tag"},
                    {"name": "parent_chunk_id", "type": "tag"},
                    {"name": "text", "type": "text"},
                    {"name": "section_path", "type": "tag"},
                    {"name": "page_refs", "type": "tag"},
                    {"name": "tags", "type": "tag"},
                    {"name": "metadata", "type": "text"},
                    {"name": "embedding_provider", "type": "tag"},
                    {"name": "embedding_model", "type": "tag"},
                    {"name": "embedding_version", "type": "tag"},
                    {"name": "tier", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "hnsw",
                            "dims": schema.dimension,
                            "distance_metric": schema.distance_metric,
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )
        index = SearchIndex(schema=redis_schema, redis_url=self.redis_url)
        index.create(overwrite=False)
        self._indices[schema.index_name] = index

    def upsert(
        self,
        index_name: str,
        record_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        index = self._indices.get(index_name)
        if index is None:
            raise IndexingError(f"index '{index_name}' has not been bootstrapped")

        record = dict(payload)
        record["id"] = record_id
        record["embedding"] = vector
        index.load([record], id_field="id")

    def query(
        self,
        index_name: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[QueryMatch]:
        index = self._indices.get(index_name)
        if index is None:
            raise IndexingError(f"index '{index_name}' has not been bootstrapped")

        try:
            from redisvl.query import VectorQuery
        except ModuleNotFoundError as exc:
            raise IndexingError(
                "RedisVL backend requires the 'redisvl' package to be installed"
            ) from exc

        return_fields = [
            "record_id",
            "document_id",
            "chunk_id",
            "parent_chunk_id",
            "text",
            "section_path",
            "page_refs",
            "tags",
            "metadata",
            "embedding_provider",
            "embedding_model",
            "embedding_version",
            "tier",
        ]
        query = VectorQuery(
            vector=vector,
            vector_field_name="embedding",
            return_fields=return_fields,
            num_results=top_k,
            filter_expression=_redis_filter_expression(filters),
        )
        results = index.query(query)

        matches: list[QueryMatch] = []
        for item in results:
            row = item if isinstance(item, dict) else item.__dict__
            score = float(row.get("vector_distance", row.get("score", 0.0)))
            record_id = str(row.get("record_id", ""))
            payload = {key: row.get(key) for key in return_fields}
            matches.append(
                QueryMatch(record_id=record_id, score=score, payload=payload)
            )
        return matches


class InMemoryIndexBackend:
    def __init__(self) -> None:
        self.schemas: dict[str, IndexSchema] = {}
        self.records: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}

    def ensure_index(self, schema: IndexSchema) -> None:
        self.schemas[schema.index_name] = schema
        self.records.setdefault(schema.index_name, {})

    def upsert(
        self,
        index_name: str,
        record_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        if index_name not in self.schemas:
            raise IndexingError(f"index '{index_name}' has not been bootstrapped")
        self.records[index_name][record_id] = (vector, payload)

    def query(
        self,
        index_name: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[QueryMatch]:
        if index_name not in self.schemas:
            raise IndexingError(f"index '{index_name}' has not been bootstrapped")

        candidates: list[QueryMatch] = []
        for record_id, (candidate_vector, payload) in self.records[index_name].items():
            if not _matches_filters(payload, filters):
                continue
            similarity = _cosine_similarity(vector, candidate_vector)
            candidates.append(
                QueryMatch(record_id=record_id, score=similarity, payload=payload)
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]


def _redis_filter_expression(filters: dict[str, Any] | None) -> str | None:
    if not filters:
        return None
    expressions: list[str] = []
    for key, value in filters.items():
        if isinstance(value, list):
            joined = "|".join(str(item) for item in value)
            expressions.append(f"@{key}:{{{joined}}}")
        else:
            expressions.append(f"@{key}:{{{value}}}")
    return " ".join(expressions)


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        current = payload.get(key)
        if isinstance(expected, list):
            if isinstance(current, list):
                if not any(item in current for item in expected):
                    return False
            elif current not in expected:
                return False
        else:
            if isinstance(current, list):
                if expected not in current:
                    return False
            elif current != expected:
                return False
    return True


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise IndexingError("cannot compare vectors with different dimensions")

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
