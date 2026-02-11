from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings
from src.ingestion.chunking import ChildChunk, ChunkingResult, ParentChunk
from src.ingestion.extraction import ExtractionResult, PageText
from src.ingestion.indexing import (
    EmbeddingTier,
    HybridVectorIndexStore,
    InMemoryIndexBackend,
    IndexRecord,
)
from src.ingestion.orchestration import (
    EmbeddingBundle,
    IngestionOrchestrator,
    PipelineError,
    RetryableStageError,
)
from src.ingestion.parsing import ParsedBlock, ParsedMarkdownDocument
from src.models.schemas import IngestionStatus


class StubExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult:
        self.calls += 1
        return ExtractionResult(
            text="sample text",
            method="pymupdf",
            pages=[PageText(page_number=1, text="sample text", method="pymupdf")],
        )


class StubParser:
    def __init__(self, fail_once_retryable: bool = False) -> None:
        self.calls = 0
        self.fail_once_retryable = fail_once_retryable

    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
        self.calls += 1
        if self.fail_once_retryable and self.calls == 1:
            raise RetryableStageError("parse_busy", "temporary parser outage")
        return ParsedMarkdownDocument(
            document_id=document_id,
            source_path=file_path,
            markdown="# Header\n\nBody",
            blocks=[
                ParsedBlock(
                    block_id="b1",
                    block_type="paragraph",
                    markdown="Body",
                    section_path=("Header",),
                    page_number=1,
                )
            ],
        )


class StubChunker:
    def __init__(self) -> None:
        self.calls = 0

    def chunk_document(self, parsed: ParsedMarkdownDocument) -> ChunkingResult:
        self.calls += 1
        parent = ParentChunk(
            chunk_id="par_1",
            document_id=parsed.document_id,
            order=1,
            text="Body",
            token_count=1,
            block_ids=["b1"],
            section_path=("Header",),
            page_refs=[1],
        )
        child = ChildChunk(
            chunk_id="chd_1",
            document_id=parsed.document_id,
            parent_chunk_id=parent.chunk_id,
            order=1,
            text="Body",
            token_count=1,
            section_path=("Header",),
            page_refs=[1],
            block_ids=["b1"],
        )
        return ChunkingResult(parent_chunks=[parent], child_chunks=[child])


class StubEmbedder:
    def __init__(self, fail_at_embed: bool = False) -> None:
        self.calls = 0
        self.fail_at_embed = fail_at_embed

    def embed(self, document_id: str, chunks: ChunkingResult) -> EmbeddingBundle:
        self.calls += 1
        if self.fail_at_embed:
            raise PipelineError("embed_failed", "embedding provider unavailable")

        return EmbeddingBundle(
            tier1_records=[
                IndexRecord(
                    record_id="tier1-1",
                    document_id=document_id,
                    chunk_id=chunks.child_chunks[0].chunk_id,
                    text=chunks.child_chunks[0].text,
                    vector=[1.0, 0.0, 0.0, 0.0],
                    section_path=chunks.child_chunks[0].section_path,
                    page_refs=chunks.child_chunks[0].page_refs,
                )
            ],
            tier4_records=[
                IndexRecord(
                    record_id="tier4-1",
                    document_id=document_id,
                    chunk_id=chunks.parent_chunks[0].chunk_id,
                    text=chunks.parent_chunks[0].text,
                    vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    section_path=chunks.parent_chunks[0].section_path,
                    page_refs=chunks.parent_chunks[0].page_refs,
                )
            ],
        )


def _orchestrator(
    parser: StubParser | None = None,
    embedder: StubEmbedder | None = None,
) -> tuple[
    IngestionOrchestrator,
    StubExtractor,
    StubParser,
    StubChunker,
    StubEmbedder,
    InMemoryIndexBackend,
]:
    backend = InMemoryIndexBackend()
    store = HybridVectorIndexStore(
        cfg=Settings(),
        backend=backend,
        tier1_dimension=4,
        tier4_dimension=6,
    )
    store.bootstrap_indices()

    extractor = StubExtractor()
    parser_instance = parser or StubParser()
    chunker = StubChunker()
    embedder_instance = embedder or StubEmbedder()
    orchestrator = IngestionOrchestrator(
        extractor=extractor,
        parser=parser_instance,
        chunker=chunker,
        embedder=embedder_instance,
        index_store=store,
        max_retries_per_stage=2,
        retry_backoff_seconds=0.0,
    )
    return orchestrator, extractor, parser_instance, chunker, embedder_instance, backend


def test_orchestration_happy_path_reaches_indexed_with_all_stages() -> None:
    orchestrator, _, _, _, _, backend = _orchestrator()

    record = orchestrator.process(
        document_id="doc-happy",
        file_path=Path("invoice.pdf"),
        mime_type="application/pdf",
    )

    assert record.status == IngestionStatus.INDEXED
    assert record.completed_stages == ["extract", "parse", "chunk", "embed", "index"]
    assert "tier1-1" in backend.records["tier1_idx"]
    assert "tier4-1" in backend.records["tier4_idx"]


def test_orchestration_retries_retryable_stage_then_succeeds() -> None:
    parser = StubParser(fail_once_retryable=True)
    orchestrator, _, parser_ref, _, _, _ = _orchestrator(parser=parser)

    record = orchestrator.process(
        document_id="doc-retry",
        file_path=Path("invoice.pdf"),
        mime_type="application/pdf",
    )

    assert record.status == IngestionStatus.INDEXED
    assert parser_ref.calls == 2
    assert record.stage_attempts["parse"] == 2


def test_orchestration_marks_partial_on_late_stage_failure() -> None:
    embedder = StubEmbedder(fail_at_embed=True)
    orchestrator, _, _, _, _, _ = _orchestrator(embedder=embedder)

    record = orchestrator.process(
        document_id="doc-partial",
        file_path=Path("invoice.pdf"),
        mime_type="application/pdf",
    )

    assert record.status == IngestionStatus.PARTIAL
    assert record.error_code == "embed_failed"
    assert record.completed_stages == ["extract", "parse", "chunk"]


def test_orchestration_is_idempotent_after_successful_ingest() -> None:
    orchestrator, extractor, parser, chunker, embedder, _ = _orchestrator()

    first = orchestrator.process(
        document_id="doc-idem",
        file_path=Path("invoice.pdf"),
        mime_type="application/pdf",
        idempotency_key="idem-1",
    )
    second = orchestrator.process(
        document_id="doc-idem",
        file_path=Path("invoice.pdf"),
        mime_type="application/pdf",
        idempotency_key="idem-1",
    )

    assert first is second
    assert first.status == IngestionStatus.INDEXED
    assert extractor.calls == 1
    assert parser.calls == 1
    assert chunker.calls == 1
    assert embedder.calls == 1


def test_orchestration_marks_failed_if_first_stage_exhausts_retries() -> None:
    class AlwaysFailParser(StubParser):
        def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
            self.calls += 1
            raise RetryableStageError("parse_busy", "temporary parser outage")

    orchestrator, _, parser, _, _, _ = _orchestrator(parser=AlwaysFailParser())

    record = orchestrator.process(
        document_id="doc-failed",
        file_path=Path("invoice.pdf"),
        mime_type="application/pdf",
    )

    assert parser.calls == 3
    assert record.status == IngestionStatus.PARTIAL
    assert record.error_code == "parse_retries_exhausted"
