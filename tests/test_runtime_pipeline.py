from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import ChildChunk, ChunkingResult, ParentChunk
from src.ingestion.embeddings import TextEmbedding
from src.ingestion.extraction import ExtractionResult, PageText
from src.ingestion.parsing import ParsedBlock, ParsedMarkdownDocument
from src.ingestion.runtime_pipeline import (
    BestEffortParser,
    BestEffortTextExtractor,
    HashingIngestionEmbedder,
    ProviderIngestionEmbedder,
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


class CountingParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, document_id: str, file_path: Path) -> object:
        del document_id, file_path
        self.calls += 1
        raise RuntimeError("should not be called when docling disabled")


class SuccessfulParser:
    def __init__(self, parser_name: str) -> None:
        self.parser_name = parser_name
        self.calls = 0

    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
        self.calls += 1
        return ParsedMarkdownDocument(
            document_id=document_id,
            source_path=file_path,
            markdown=f"# Parsed by {self.parser_name}",
            blocks=[
                ParsedBlock(
                    block_id=f"{self.parser_name}-1",
                    block_type="paragraph",
                    markdown=f"{self.parser_name} output",
                    section_path=("root",),
                    page_number=1,
                )
            ],
            parser_name=self.parser_name,
        )


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


class MultiPageExtractor:
    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult:
        del file_path, mime_type
        return ExtractionResult(
            text="Page one text\n\nPage two text",
            method="fallback",
            pages=[
                PageText(page_number=1, text="Page one text", method="fallback"),
                PageText(page_number=2, text="Page two text", method="fallback"),
            ],
        )


class StubEmbeddingClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        version: str,
        dimension: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.dimension = dimension

    def embed_text(self, text: str) -> TextEmbedding:
        weight = float(max(len(text), 1))
        vector = [weight for _ in range(self.dimension)]
        return TextEmbedding(
            vector=vector,
            provider=self.provider,
            model=self.model,
            version=self.version,
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
        google_enabled=False,
    )

    parsed = parser.parse(document_id="doc-1", file_path=file_path)

    assert parsed.document_id == "doc-1"
    assert parsed.blocks
    assert parsed.blocks[0].markdown == "Total Due: 1234.00 USD"


def test_best_effort_parser_uses_page_level_blocks_on_fallback(tmp_path: Path) -> None:
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"ignored")
    parser = BestEffortParser(
        docling_parser=RaisingParser(),
        fallback_extractor=MultiPageExtractor(),
        google_enabled=False,
    )

    parsed = parser.parse(document_id="doc-2", file_path=file_path)

    assert len(parsed.blocks) == 2
    assert [block.page_number for block in parsed.blocks] == [1, 2]
    assert parsed.blocks[0].section_path == ("paper.pdf", "Page 1")
    assert parsed.blocks[1].section_path == ("paper.pdf", "Page 2")


def test_best_effort_parser_skips_docling_when_disabled(tmp_path: Path) -> None:
    file_path = tmp_path / "note.pdf"
    file_path.write_bytes(b"ignored")
    docling_parser = CountingParser()
    parser = BestEffortParser(
        docling_parser=docling_parser,
        fallback_extractor=FixedExtractor(),
        docling_enabled=False,
        google_enabled=False,
    )

    parsed = parser.parse(document_id="doc-3", file_path=file_path)

    assert docling_parser.calls == 0
    assert parsed.blocks
    assert parsed.blocks[0].markdown == "Total Due: 1234.00 USD"


def test_best_effort_parser_uses_google_when_docling_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "route.pdf"
    file_path.write_bytes(b"ignored")
    google_parser = SuccessfulParser("google")
    parser = BestEffortParser(
        docling_parser=RaisingParser(),
        google_parser=google_parser,
        fallback_extractor=FixedExtractor(),
    )

    parsed = parser.parse(document_id="doc-4", file_path=file_path)

    assert google_parser.calls == 1
    assert parsed.parser_name == "google"


def test_best_effort_parser_respects_routing_order_override(tmp_path: Path) -> None:
    file_path = tmp_path / "route-order.pdf"
    file_path.write_bytes(b"ignored")
    docling_parser = SuccessfulParser("docling")
    google_parser = SuccessfulParser("google")
    parser = BestEffortParser(
        docling_parser=docling_parser,
        google_parser=google_parser,
        fallback_extractor=FixedExtractor(),
        parser_order=("google", "docling", "fallback"),
    )

    parsed = parser.parse(document_id="doc-5", file_path=file_path)

    assert google_parser.calls == 1
    assert docling_parser.calls == 0
    assert parsed.parser_name == "google"


def test_best_effort_parser_persists_parsed_markdown_artifact(tmp_path: Path) -> None:
    file_path = tmp_path / "persist.pdf"
    file_path.write_bytes(b"ignored")
    parsed_dir = tmp_path / "parsed"
    parser = BestEffortParser(
        docling_parser=SuccessfulParser("docling"),
        fallback_extractor=FixedExtractor(),
        google_enabled=False,
        parsed_dir=parsed_dir,
    )

    parsed = parser.parse(document_id="doc-6", file_path=file_path)

    artifact_path = parsed_dir / "doc-6.md"
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == parsed.markdown


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


def test_provider_ingestion_embedder_stamps_provider_metadata() -> None:
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
    embedder = ProviderIngestionEmbedder(
        tier1_client=StubEmbeddingClient(
            provider="ollama",
            model="all-minilm",
            version="ollama:all-minilm:v1",
            dimension=8,
        ),
        tier4_client=StubEmbeddingClient(
            provider="gemini",
            model="gemini-embedding-001",
            version="gemini:gemini-embedding-001:v1",
            dimension=12,
        ),
    )

    bundle = embedder.embed(document_id="doc-1", chunks=chunks)

    assert bundle.tier1_records[0].embedding_provider == "ollama"
    assert bundle.tier1_records[0].embedding_model == "all-minilm"
    assert bundle.tier4_records[0].embedding_provider == "gemini"
    assert bundle.tier4_records[0].embedding_model == "gemini-embedding-001"
