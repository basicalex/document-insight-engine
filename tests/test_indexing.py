from __future__ import annotations

from src.config.settings import Settings
from src.ingestion.indexing import (
    EmbeddingTier,
    HybridVectorIndexStore,
    InMemoryIndexBackend,
    IndexRecord,
    IndexingError,
)


def _store() -> tuple[HybridVectorIndexStore, InMemoryIndexBackend]:
    backend = InMemoryIndexBackend()
    store = HybridVectorIndexStore(
        cfg=Settings(),
        backend=backend,
        tier1_dimension=4,
        tier4_dimension=6,
    )
    store.bootstrap_indices()
    return store, backend


def test_bootstrap_creates_tier_indices_with_expected_dimensions() -> None:
    store, backend = _store()

    assert store is not None
    assert backend.schemas["tier1_idx"].dimension == 4
    assert backend.schemas["tier4_idx"].dimension == 6


def test_persist_records_adds_provider_model_and_tags_metadata() -> None:
    store, backend = _store()

    store.persist_records(
        tier=EmbeddingTier.TIER1,
        records=[
            IndexRecord(
                record_id="r1",
                document_id="doc-1",
                chunk_id="chunk-1",
                text="invoice total 123",
                vector=[1.0, 0.0, 0.0, 0.0],
                metadata={"source": "upload"},
                tags=["invoice", "finance"],
                section_path=("Invoice", "Totals"),
                page_refs=[1],
            )
        ],
    )

    _, payload = backend.records["tier1_idx"]["r1"]
    assert payload["embedding_provider"] == "local"
    assert payload["embedding_model"] == "nomic-embed-text:v1.5"
    assert payload["tier"] == "tier1"
    assert payload["tags"] == ["invoice", "finance"]
    assert payload["section_path"] == "Invoice/Totals"


def test_query_returns_filtered_matches() -> None:
    store, _ = _store()
    store.persist_records(
        tier=EmbeddingTier.TIER1,
        records=[
            IndexRecord(
                record_id="a",
                document_id="doc-1",
                chunk_id="chunk-a",
                text="alpha",
                vector=[1.0, 0.0, 0.0, 0.0],
                tags=["policy"],
            ),
            IndexRecord(
                record_id="b",
                document_id="doc-2",
                chunk_id="chunk-b",
                text="beta",
                vector=[0.9, 0.1, 0.0, 0.0],
                tags=["invoice"],
            ),
        ],
    )

    matches = store.query(
        tier=EmbeddingTier.TIER1,
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"tags": ["invoice"]},
    )

    assert len(matches) == 1
    assert matches[0].record_id == "b"


def test_persist_rejects_dimension_mismatch() -> None:
    store, _ = _store()

    try:
        store.persist_records(
            tier=EmbeddingTier.TIER1,
            records=[
                IndexRecord(
                    record_id="bad",
                    document_id="doc-1",
                    chunk_id="chunk-bad",
                    text="wrong dims",
                    vector=[1.0, 2.0],
                )
            ],
        )
    except IndexingError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("dimension mismatch should raise IndexingError")


def test_query_rejects_dimension_mismatch() -> None:
    store, _ = _store()

    try:
        store.query(
            tier=EmbeddingTier.TIER4,
            vector=[0.0, 1.0],
            top_k=3,
        )
    except IndexingError as exc:
        assert "expected 6" in str(exc)
    else:
        raise AssertionError("query dimension mismatch should raise IndexingError")
