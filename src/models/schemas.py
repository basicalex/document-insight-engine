from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Mode(str, Enum):
    FAST = "fast"
    DEEP_LITE = "deep-lite"
    DEEP = "deep"


class IngestionStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PARTIAL = "partial"
    INDEXED = "indexed"
    FAILED = "failed"


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=_utcnow)
    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)
    latency_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    model: str = Field(min_length=1)
    prompt_version: str | None = None
    iterations: int = Field(default=0, ge=0, le=5)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    tool_calls: list[TraceEvent] = Field(default_factory=list)
    termination_reason: str | None = None
    total_latency_ms: int | None = Field(default=None, ge=0)


class Document(BaseModel):
    id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    filepath: str = Field(min_length=1)
    mime_type: str | None = None
    checksum: str | None = None
    status: IngestionStatus = IngestionStatus.UPLOADED
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    mode: Mode = Mode.FAST
    document_id: str | None = None
    session_id: str | None = None


class Citation(BaseModel):
    chunk_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class ChatResponse(BaseModel):
    answer: str = Field(min_length=1)
    mode: Mode
    document_id: str | None = None
    insufficient_evidence: bool = False
    citations: list[Citation] = Field(default_factory=list)
    trace: AgentTrace | None = None


class IngestionProgress(BaseModel):
    stage: str | None = None
    processed_items: int | None = None
    total_items: int | None = None


class IngestResponse(BaseModel):
    document_id: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    status: IngestionStatus = IngestionStatus.UPLOADED
    message: str | None = None
    progress: IngestionProgress | None = None


class UploadBatchResponse(BaseModel):
    documents: list[IngestResponse] = Field(default_factory=list)
    count: int = Field(ge=0)


class StructuredExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(min_length=1)
    extraction_schema: dict[str, Any] = Field(alias="schema")
    prompt: str | None = None


class StructuredFieldProvenance(BaseModel):
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    text: str


class StructuredValidationDiagnostic(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    field: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class StructuredExtractResponse(BaseModel):
    document_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, StructuredFieldProvenance] = Field(default_factory=dict)
    accepted_fields: list[str] = Field(default_factory=list)
    rejected_fields: list[str] = Field(default_factory=list)
    diagnostics: list[StructuredValidationDiagnostic] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = Field(ge=0)
    artifact_path: str = Field(min_length=1)


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    correlation_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
