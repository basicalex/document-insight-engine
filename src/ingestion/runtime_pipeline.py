from __future__ import annotations

import mimetypes
from hashlib import sha256
from pathlib import Path

from src.ingestion.chunking import ChunkingResult
from src.ingestion.extraction import ExtractionResult, PageText, Tier1TextExtractor
from src.ingestion.indexing import IndexRecord
from src.ingestion.orchestration import EmbeddingBundle
from src.ingestion.parsing import (
    ParsedBlock,
    ParsedMarkdownDocument,
    Tier2DoclingParser,
)
from src.ingestion.vectorize import hashing_vector


class BestEffortTextExtractor:
    def __init__(self, delegate: Tier1TextExtractor | None = None) -> None:
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
        docling_parser: Tier2DoclingParser | None = None,
        fallback_extractor: BestEffortTextExtractor | None = None,
    ) -> None:
        self._docling_parser = docling_parser or Tier2DoclingParser()
        self._fallback_extractor = fallback_extractor or BestEffortTextExtractor()

    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
        try:
            return self._docling_parser.parse(
                document_id=document_id, file_path=file_path
            )
        except Exception:
            mime_type = _guess_mime(file_path)
            extraction = self._fallback_extractor.extract(
                file_path=file_path,
                mime_type=mime_type,
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
            )


class HashingIngestionEmbedder:
    def __init__(
        self,
        *,
        tier1_dimension: int = 384,
        tier4_dimension: int = 3072,
        embedding_version: str = "hash-v1",
    ) -> None:
        self.tier1_dimension = tier1_dimension
        self.tier4_dimension = tier4_dimension
        self.embedding_version = embedding_version

    def embed(self, document_id: str, chunks: ChunkingResult) -> EmbeddingBundle:
        tier1_records = [
            IndexRecord(
                record_id=f"t1_{chunk.chunk_id}",
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                vector=hashing_vector(chunk.text, self.tier1_dimension),
                metadata={
                    "kind": "child",
                    "order": chunk.order,
                },
                tags=["tier1", "child"],
                parent_chunk_id=chunk.parent_chunk_id,
                section_path=chunk.section_path,
                page_refs=chunk.page_refs,
                embedding_version=self.embedding_version,
            )
            for chunk in chunks.child_chunks
        ]

        tier4_records = [
            IndexRecord(
                record_id=f"t4_{chunk.chunk_id}",
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                vector=hashing_vector(chunk.text, self.tier4_dimension),
                metadata={
                    "kind": "parent",
                    "order": chunk.order,
                },
                tags=["tier4", "parent"],
                section_path=chunk.section_path,
                page_refs=chunk.page_refs,
                embedding_version=self.embedding_version,
            )
            for chunk in chunks.parent_chunks
        ]

        return EmbeddingBundle(tier1_records=tier1_records, tier4_records=tier4_records)


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
