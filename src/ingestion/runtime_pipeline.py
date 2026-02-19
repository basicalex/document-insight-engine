from __future__ import annotations

import mimetypes
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from src.ingestion.chunking import ChunkingResult
from src.ingestion.embeddings import HashingEmbeddingClient, TextEmbeddingClient
from src.ingestion.extraction import ExtractionResult, PageText, Tier1TextExtractor
from src.ingestion.google_parser import Tier2GoogleParser
from src.ingestion.indexing import IndexRecord
from src.ingestion.orchestration import EmbeddingBundle
from src.ingestion.parsing import (
    ParsedBlock,
    ParsedMarkdownDocument,
    Tier2DoclingParser,
)


class BestEffortTextExtractor:
    def __init__(self, delegate: "TextExtractorDelegate | None" = None) -> None:
        self._delegate = delegate or Tier1TextExtractor()

    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult:
        try:
            return self._delegate.extract(file_path=file_path, mime_type=mime_type)
        except Exception:
            fallback_text = _read_text_fallback(file_path)
            return ExtractionResult(
                text=fallback_text,
                method="fallback",
                pages=[PageText(page_number=1, text=fallback_text, method="fallback")],
            )


class BestEffortParser:
    def __init__(
        self,
        *,
        docling_parser: "MarkdownParserDelegate | None" = None,
        google_parser: "MarkdownParserDelegate | None" = None,
        fallback_extractor: "TextExtractorDelegate | None" = None,
        docling_enabled: bool = True,
        google_enabled: bool = True,
        parser_order: tuple[str, ...] | None = None,
        parsed_dir: Path | None = None,
    ) -> None:
        self._docling_parser = docling_parser or Tier2DoclingParser()
        self._google_parser = google_parser or Tier2GoogleParser()
        self._fallback_extractor = fallback_extractor or BestEffortTextExtractor()
        self._parsed_dir = parsed_dir
        self._parser_order = _resolve_parser_order(
            parser_order=parser_order,
            docling_enabled=docling_enabled,
            google_enabled=google_enabled,
        )

    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
        attempts: list[tuple[str, str]] = []
        for parser_name in self._parser_order:
            try:
                if parser_name == "docling":
                    parsed = self._docling_parser.parse(
                        document_id=document_id,
                        file_path=file_path,
                    )
                    normalized = _normalize_parsed_document(
                        parsed=parsed,
                        parser_name="docling",
                    )
                    self._persist_parsed_markdown(normalized)
                    return normalized

                if parser_name == "google":
                    parsed = self._google_parser.parse(
                        document_id=document_id,
                        file_path=file_path,
                    )
                    normalized = _normalize_parsed_document(
                        parsed=parsed,
                        parser_name="google",
                    )
                    self._persist_parsed_markdown(normalized)
                    return normalized

                if parser_name == "fallback":
                    parsed = self._fallback_parse(
                        document_id=document_id, file_path=file_path
                    )
                    self._persist_parsed_markdown(parsed)
                    return parsed
            except Exception as exc:
                attempts.append((parser_name, str(exc)))

        raise RuntimeError(f"No parser succeeded. attempts={attempts}")

    def _fallback_parse(
        self, *, document_id: str, file_path: Path
    ) -> ParsedMarkdownDocument:
        mime_type = _guess_mime(file_path)
        extraction = self._fallback_extractor.extract(
            file_path=file_path,
            mime_type=mime_type,
        )
        page_blocks = _fallback_blocks_from_pages(
            document_id=document_id,
            file_name=file_path.name,
            extraction=extraction,
        )
        if page_blocks:
            markdown_sections = [
                f"## Page {block.page_number}\n\n{block.markdown}"
                for block in page_blocks
                if block.page_number is not None
            ]
            markdown_body = "\n\n".join(markdown_sections)
            markdown = f"# {file_path.name}\n\n{markdown_body}".strip()
            return ParsedMarkdownDocument(
                document_id=document_id,
                source_path=file_path,
                markdown=markdown,
                blocks=page_blocks,
                parser_name="fallback",
            )

        text = extraction.text.strip() or f"document {file_path.name}"
        markdown = f"# {file_path.name}\n\n{text}"
        block = ParsedBlock(
            block_id=_fallback_block_id(document_id=document_id, text=markdown),
            block_type="paragraph",
            markdown=text,
            section_path=(file_path.name,),
            page_number=1,
        )
        return ParsedMarkdownDocument(
            document_id=document_id,
            source_path=file_path,
            markdown=markdown,
            blocks=[block],
            parser_name="fallback",
        )

    def _persist_parsed_markdown(self, parsed: ParsedMarkdownDocument) -> None:
        if self._parsed_dir is None:
            return

        safe_document_id = Path(parsed.document_id).name
        if safe_document_id != parsed.document_id:
            raise RuntimeError("invalid document_id for parsed artifact path")

        self._parsed_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self._parsed_dir / f"{safe_document_id}.md"
        artifact_path.write_text(parsed.markdown, encoding="utf-8")


class ProviderIngestionEmbedder:
    def __init__(
        self,
        *,
        tier1_client: TextEmbeddingClient,
        tier4_client: TextEmbeddingClient,
    ) -> None:
        self.tier1_client = tier1_client
        self.tier4_client = tier4_client

    def embed(self, document_id: str, chunks: ChunkingResult) -> EmbeddingBundle:
        tier1_records: list[IndexRecord] = []
        for chunk in chunks.child_chunks:
            embedding = self.tier1_client.embed_text(chunk.text)
            tier1_records.append(
                IndexRecord(
                    record_id=f"t1_{chunk.chunk_id}",
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    vector=embedding.vector,
                    metadata={
                        "kind": "child",
                        "order": chunk.order,
                    },
                    tags=["tier1", "child"],
                    parent_chunk_id=chunk.parent_chunk_id,
                    section_path=chunk.section_path,
                    page_refs=chunk.page_refs,
                    embedding_version=embedding.version,
                    embedding_provider=embedding.provider,
                    embedding_model=embedding.model,
                )
            )

        tier4_records: list[IndexRecord] = []
        for chunk in chunks.parent_chunks:
            embedding = self.tier4_client.embed_text(chunk.text)
            tier4_records.append(
                IndexRecord(
                    record_id=f"t4_{chunk.chunk_id}",
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    vector=embedding.vector,
                    metadata={
                        "kind": "parent",
                        "order": chunk.order,
                    },
                    tags=["tier4", "parent"],
                    section_path=chunk.section_path,
                    page_refs=chunk.page_refs,
                    embedding_version=embedding.version,
                    embedding_provider=embedding.provider,
                    embedding_model=embedding.model,
                )
            )

        return EmbeddingBundle(tier1_records=tier1_records, tier4_records=tier4_records)


class HashingIngestionEmbedder(ProviderIngestionEmbedder):
    def __init__(
        self,
        *,
        tier1_dimension: int = 384,
        tier4_dimension: int = 3072,
        embedding_version: str = "hash-v1",
    ) -> None:
        super().__init__(
            tier1_client=HashingEmbeddingClient(
                model="hash:tier1",
                dimension=tier1_dimension,
                version=embedding_version,
            ),
            tier4_client=HashingEmbeddingClient(
                model="hash:tier4",
                dimension=tier4_dimension,
                version=embedding_version,
            ),
        )


class TextExtractorDelegate(Protocol):
    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult: ...


class MarkdownParserDelegate(Protocol):
    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument: ...


def _read_text_fallback(file_path: Path) -> str:
    data = file_path.read_bytes()
    text = data.decode("utf-8", errors="ignore").strip()
    if text:
        return text
    return f"binary document: {file_path.name}"


def _guess_mime(file_path: Path) -> str:
    guessed = mimetypes.guess_type(str(file_path))[0]
    return guessed or "application/octet-stream"


def _fallback_block_id(*, document_id: str, text: str) -> str:
    digest = sha256(f"{document_id}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"fallback_{digest}"


def _fallback_blocks_from_pages(
    *,
    document_id: str,
    file_name: str,
    extraction: ExtractionResult,
) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    sorted_pages = sorted(extraction.pages, key=lambda page: page.page_number)
    for page in sorted_pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        block_text = f"page={page.page_number}|{page_text}"
        blocks.append(
            ParsedBlock(
                block_id=_fallback_block_id(document_id=document_id, text=block_text),
                block_type="paragraph",
                markdown=page_text,
                section_path=(file_name, f"Page {page.page_number}"),
                page_number=page.page_number,
            )
        )
    return blocks


def _resolve_parser_order(
    *,
    parser_order: tuple[str, ...] | None,
    docling_enabled: bool,
    google_enabled: bool,
) -> tuple[str, ...]:
    candidate = parser_order or ("docling", "google", "fallback")
    allowed: tuple[str, ...] = ("docling", "google", "fallback")
    deduped: list[str] = []
    for parser_name in candidate:
        if parser_name not in allowed:
            continue
        if parser_name == "docling" and not docling_enabled:
            continue
        if parser_name == "google" and not google_enabled:
            continue
        if parser_name not in deduped:
            deduped.append(parser_name)

    if "fallback" not in deduped:
        deduped.append("fallback")
    return tuple(deduped)


def _normalize_parsed_document(
    *,
    parsed: ParsedMarkdownDocument,
    parser_name: str,
) -> ParsedMarkdownDocument:
    markdown = parsed.markdown.strip()
    blocks = [block for block in parsed.blocks if block.markdown.strip()]
    if blocks:
        return ParsedMarkdownDocument(
            document_id=parsed.document_id,
            source_path=parsed.source_path,
            markdown=markdown,
            blocks=blocks,
            parser_name=parser_name,
        )

    fallback_text = markdown or f"document {parsed.source_path.name}"
    block = ParsedBlock(
        block_id=_fallback_block_id(document_id=parsed.document_id, text=fallback_text),
        block_type="paragraph",
        markdown=fallback_text,
        section_path=(parsed.source_path.name,),
        page_number=1,
    )
    return ParsedMarkdownDocument(
        document_id=parsed.document_id,
        source_path=parsed.source_path,
        markdown=markdown or f"# {parsed.source_path.name}\n\n{fallback_text}",
        blocks=[block],
        parser_name=parser_name,
    )
