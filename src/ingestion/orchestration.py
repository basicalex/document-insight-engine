from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from src.ingestion.chunking import ChunkingResult
from src.ingestion.extraction import ExtractionResult
from src.ingestion.indexing import EmbeddingTier, HybridVectorIndexStore, IndexRecord
from src.ingestion.parsing import ParsedMarkdownDocument
from src.models.schemas import IngestionStatus


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class IngestionEvent:
    timestamp: datetime
    stage: str
    level: str
    message: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class IngestionRecord:
    document_id: str
    idempotency_key: str
    status: IngestionStatus = IngestionStatus.UPLOADED
    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    stage_attempts: dict[str, int] = field(default_factory=dict)
    events: list[IngestionEvent] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class EmbeddingBundle:
    tier1_records: list[IndexRecord]
    tier4_records: list[IndexRecord]


class Extractor(Protocol):
    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult: ...


class Parser(Protocol):
    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument: ...


class Chunker(Protocol):
    def chunk_document(self, parsed: ParsedMarkdownDocument) -> ChunkingResult: ...


class Embedder(Protocol):
    def embed(self, document_id: str, chunks: ChunkingResult) -> EmbeddingBundle: ...


class PipelineError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class RetryableStageError(PipelineError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code=code, message=message, retryable=True)


class IngestionOrchestrator:
    def __init__(
        self,
        extractor: Extractor,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        index_store: HybridVectorIndexStore,
        max_retries_per_stage: int = 2,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        self.extractor = extractor
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.index_store = index_store
        self.max_retries_per_stage = max_retries_per_stage
        self.retry_backoff_seconds = retry_backoff_seconds
        self._records_by_key: dict[str, IngestionRecord] = {}
        self._records_by_document: dict[str, IngestionRecord] = {}

    def process(
        self,
        document_id: str,
        file_path: Path,
        mime_type: str,
        idempotency_key: str | None = None,
    ) -> IngestionRecord:
        key = idempotency_key or self._default_idempotency_key(document_id=document_id)

        existing = self._records_by_key.get(key)
        if existing is not None and existing.status in {
            IngestionStatus.PROCESSING,
            IngestionStatus.PARTIAL,
            IngestionStatus.INDEXED,
        }:
            return existing

        record = IngestionRecord(document_id=document_id, idempotency_key=key)
        self._records_by_key[key] = record
        self._records_by_document[document_id] = record

        self._transition(record=record, next_status=IngestionStatus.PROCESSING)

        extraction: ExtractionResult | None = None
        parsed: ParsedMarkdownDocument | None = None
        chunked: ChunkingResult | None = None
        embeddings: EmbeddingBundle | None = None

        stages = [
            (
                "extract",
                lambda: self.extractor.extract(
                    file_path=file_path, mime_type=mime_type
                ),
            ),
            (
                "parse",
                lambda: self.parser.parse(document_id=document_id, file_path=file_path),
            ),
            (
                "chunk",
                lambda: self.chunker.chunk_document(parsed=self._must(parsed, "parse")),
            ),
            (
                "embed",
                lambda: self.embedder.embed(
                    document_id=document_id,
                    chunks=self._must(chunked, "chunk"),
                ),
            ),
            (
                "index",
                lambda: self._index_embeddings(
                    embeddings=self._must(embeddings, "embed")
                ),
            ),
        ]

        for stage_name, operation in stages:
            try:
                output = self._execute_stage(
                    record=record, stage_name=stage_name, operation=operation
                )
            except PipelineError as exc:
                final = (
                    IngestionStatus.PARTIAL
                    if len(record.completed_stages) > 0
                    else IngestionStatus.FAILED
                )
                self._transition(record=record, next_status=final)
                record.error_code = exc.code
                record.error_message = exc.message
                self._event(
                    record=record,
                    stage=stage_name,
                    level="error",
                    message=exc.message,
                    metadata={"code": exc.code},
                )
                return record

            if stage_name == "extract":
                extraction = output
            elif stage_name == "parse":
                parsed = output
            elif stage_name == "chunk":
                chunked = output
            elif stage_name == "embed":
                embeddings = output

            record.completed_stages.append(stage_name)

        self._transition(record=record, next_status=IngestionStatus.INDEXED)
        self._event(
            record=record,
            stage="pipeline",
            level="info",
            message="ingestion pipeline completed",
            metadata={"document_id": document_id},
        )
        return record

    def get_record(self, document_id: str) -> IngestionRecord | None:
        return self._records_by_document.get(document_id)

    def _execute_stage(
        self, record: IngestionRecord, stage_name: str, operation: callable
    ) -> object:
        attempts = self.max_retries_per_stage + 1
        last_error: PipelineError | None = None

        for attempt in range(1, attempts + 1):
            record.current_stage = stage_name
            record.stage_attempts[stage_name] = attempt
            self._event(
                record=record,
                stage=stage_name,
                level="info",
                message="stage started",
                metadata={"attempt": str(attempt)},
            )
            try:
                value = operation()
                self._event(
                    record=record,
                    stage=stage_name,
                    level="info",
                    message="stage completed",
                    metadata={"attempt": str(attempt)},
                )
                return value
            except RetryableStageError as exc:
                last_error = exc
                self._event(
                    record=record,
                    stage=stage_name,
                    level="warning",
                    message=exc.message,
                    metadata={"attempt": str(attempt), "code": exc.code},
                )
                if attempt >= attempts:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds)
            except PipelineError as exc:
                raise exc
            except Exception as exc:
                raise PipelineError(
                    code=f"{stage_name}_error",
                    message=f"{stage_name} stage failed: {exc}",
                ) from exc

        assert last_error is not None
        raise PipelineError(
            code=f"{stage_name}_retries_exhausted",
            message=f"{stage_name} failed after {attempts} attempts: {last_error.message}",
        )

    def _index_embeddings(self, embeddings: EmbeddingBundle) -> None:
        self.index_store.persist_records(EmbeddingTier.TIER1, embeddings.tier1_records)
        self.index_store.persist_records(EmbeddingTier.TIER4, embeddings.tier4_records)

    def _transition(
        self, record: IngestionRecord, next_status: IngestionStatus
    ) -> None:
        allowed = {
            IngestionStatus.UPLOADED: {
                IngestionStatus.PROCESSING,
                IngestionStatus.FAILED,
            },
            IngestionStatus.PROCESSING: {
                IngestionStatus.PARTIAL,
                IngestionStatus.INDEXED,
                IngestionStatus.FAILED,
            },
            IngestionStatus.PARTIAL: set(),
            IngestionStatus.INDEXED: set(),
            IngestionStatus.FAILED: set(),
        }
        if next_status == record.status:
            return
        if next_status not in allowed.get(record.status, set()):
            raise PipelineError(
                code="invalid_transition",
                message=f"cannot transition {record.status.value} -> {next_status.value}",
            )

        record.status = next_status
        record.updated_at = _utcnow()
        self._event(
            record=record,
            stage="lifecycle",
            level="info",
            message="status changed",
            metadata={"status": next_status.value, "document_id": record.document_id},
        )

    def _event(
        self,
        record: IngestionRecord,
        stage: str,
        level: str,
        message: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        event = IngestionEvent(
            timestamp=_utcnow(),
            stage=stage,
            level=level,
            message=message,
            metadata=metadata or {},
        )
        record.events.append(event)
        record.updated_at = event.timestamp

    def _default_idempotency_key(self, document_id: str) -> str:
        digest = sha256(document_id.encode("utf-8")).hexdigest()[:24]
        return f"ingest_{digest}"

    def _must(self, value: object | None, stage: str) -> object:
        if value is None:
            raise PipelineError(
                code="stage_dependency_missing", message=f"missing output from {stage}"
            )
        return value
