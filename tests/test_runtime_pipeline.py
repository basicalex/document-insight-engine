from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import ChildChunk, ChunkingResult, ParentChunk
from src.ingestion.extraction import ExtractionResult, PageText
from src.ingestion.runtime_pipeline import (
    BestEffortParser,
    BestEffortTextExtractor,
    HashingIngestionEmbedder,
)
from src.ingestion.vectorize import hashing_vector


class RaisingExtractor:
    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult:
        del file_path, mime_type
        raise RuntimeError("missing extraction dependency")


class RaisingParser:
    def parse(self, document_id: str, file_path: Path) -> object:
        del document_id, file_path
        raise RuntimeError("docling not available")


class FixedExtractor:
    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult:
        del file_path, mime_type
        return ExtractionResult(
            text="Total Due: 1234.00 USD",
            method="fallback",
            pages=[
                PageText(
                    page_number=1, text="Total Due: 1234.00 USD", method="fallback"
                )
            ],
        )


def test_hashing_vector_is_stable_and_normalized() -> None:
    first = hashing_vector("total due invoice", dimension=16)
    second = hashing_vector("total due invoice", dimension=16)

    assert first == second
    assert len(first) == 16
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_best_effort_extractor_falls_back_when_delegate_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"fallback text content")
    extractor = BestEffortTextExtractor(delegate=RaisingExtractor())

    result = extractor.extract(file_path=file_path, mime_type="application/pdf")

    assert result.method == "fallback"
    assert "fallback text content" in result.text


def test_best_effort_parser_falls_back_when_docling_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "invoice.pdf"
    file_path.write_bytes(b"ignored")
    parser = BestEffortParser(
        docling_parser=RaisingParser(),
        fallback_extractor=FixedExtractor(),
    )

    parsed = parser.parse(document_id="doc-1", file_path=file_path)

    assert parsed.document_id == "doc-1"
    assert parsed.blocks
    assert parsed.blocks[0].markdown == "Total Due: 1234.00 USD"


def test_hashing_ingestion_embedder_creates_tier_records() -> None:
    parent = ParentChunk(
        chunk_id="par-1",
        document_id="doc-1",
        order=1,
        text="Parent clause text",
        token_count=3,
        block_ids=["b1"],
        section_path=("Section",),
        page_refs=[1],
    )
    child = ChildChunk(
        chunk_id="chd-1",
        document_id="doc-1",
        parent_chunk_id="par-1",
        order=1,
        text="Child clause text",
        token_count=3,
        section_path=("Section",),
        page_refs=[1],
        block_ids=["b1"],
    )
    chunks = ChunkingResult(parent_chunks=[parent], child_chunks=[child])
    embedder = HashingIngestionEmbedder(tier1_dimension=8, tier4_dimension=12)

    bundle = embedder.embed(document_id="doc-1", chunks=chunks)

    assert len(bundle.tier1_records) == 1
    assert len(bundle.tier4_records) == 1
    assert len(bundle.tier1_records[0].vector) == 8
    assert len(bundle.tier4_records[0].vector) == 12
