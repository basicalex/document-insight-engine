from __future__ import annotations

import asyncio
import importlib.util
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config.settings import Settings, settings
from src.engine.cloud_agent import CloudAgentEngine, CloudAgentModelClient
from src.engine.local_llm import LocalQAEngine
from src.ingestion import (
    BestEffortParser,
    BestEffortTextExtractor,
    HashingIngestionEmbedder,
    HybridVectorIndexStore,
    IngestionOrchestrator,
    InMemoryIndexBackend,
    IndexingError,
    ParentChildChunker,
    UploadIntakeError,
    UploadIntakeService,
)
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    IngestResponse,
    IngestionStatus,
    Mode,
    UploadBatchResponse,
)


logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class _FallbackCloudModelClient(CloudAgentModelClient):
    def next_step(
        self,
        *,
        question: str,
        mode: Mode,
        document_id: str,
        iteration: int,
        history: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        return {
            "action": "final",
            "answer": "Deep mode cloud agent is not configured for this environment.",
            "insufficient_evidence": True,
        }


class _DisabledCloudModelClient(CloudAgentModelClient):
    def next_step(
        self,
        *,
        question: str,
        mode: Mode,
        document_id: str,
        iteration: int,
        history: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        del question, mode, document_id, iteration, history, allowed_tools
        return {
            "action": "final",
            "answer": "Deep mode is disabled in this deployment.",
            "insufficient_evidence": True,
        }


@dataclass
class ApiServices:
    cfg: Settings
    intake: UploadIntakeService
    index_store: HybridVectorIndexStore
    local_qa: LocalQAEngine
    cloud_agent: CloudAgentEngine
    orchestrator: IngestionOrchestrator | None = None
    idempotency_records: dict[str, IngestResponse] = field(default_factory=dict)
    ingestion_records: dict[str, IngestResponse] = field(default_factory=dict)
    idempotency_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


app = FastAPI(title=settings.project_name)


def get_app_settings() -> Settings:
    return settings


def _build_default_services(cfg: Settings) -> ApiServices:
    try:
        index_store = HybridVectorIndexStore(cfg=cfg)
        index_store.bootstrap_indices()
    except Exception as exc:  # pragma: no cover - runtime dependency fallback
        logger.warning(
            "Falling back to in-memory index backend for API runtime: %s",
            exc,
        )
        in_memory = InMemoryIndexBackend()
        index_store = HybridVectorIndexStore(cfg=cfg, backend=in_memory)
        index_store.bootstrap_indices()

    extractor = BestEffortTextExtractor()
    parser = BestEffortParser(fallback_extractor=extractor)
    chunker = ParentChildChunker()
    embedder = HashingIngestionEmbedder()

    model_client: CloudAgentModelClient
    if cfg.cloud_agent_provider == "fallback":
        model_client = _FallbackCloudModelClient()
    else:
        model_client = _DisabledCloudModelClient()

    return ApiServices(
        cfg=cfg,
        intake=UploadIntakeService(cfg),
        index_store=index_store,
        local_qa=LocalQAEngine(index_store=index_store, cfg=cfg),
        cloud_agent=CloudAgentEngine(model_client=model_client, cfg=cfg),
        orchestrator=IngestionOrchestrator(
            extractor=extractor,
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            index_store=index_store,
            max_retries_per_stage=2,
            retry_backoff_seconds=0.0,
        ),
    )


def get_services(request: Request) -> ApiServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        services = _build_default_services(get_app_settings())
        request.app.state.services = services
    return services


@app.on_event("startup")
def ensure_runtime_dirs() -> None:
    settings.ensure_runtime_dirs()
    if not hasattr(app.state, "services"):
        app.state.services = _build_default_services(settings)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
    correlation_id = request.headers.get("x-correlation-id") or uuid4().hex
    request.state.correlation_id = correlation_id

    response = await call_next(request)
    response.headers["x-correlation-id"] = correlation_id
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return _build_error_response(
        request=request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _build_error_response(
        request=request,
        status_code=422,
        code="validation_error",
        message="request validation failed",
        details={"errors": exc.errors()},
    )


@app.get("/healthz")
def healthz(services: ApiServices = Depends(get_services)) -> dict[str, Any]:
    backend_name = type(services.index_store.backend).__name__
    return {
        "status": "ok",
        "index_backend": backend_name,
        "orchestrator_configured": services.orchestrator is not None,
        "deep_mode_enabled": services.cfg.deep_mode_enabled,
        "cloud_agent_provider": services.cfg.cloud_agent_provider,
        "dependencies": _runtime_dependency_report(),
    }


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    response: Response,
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    services: ApiServices = Depends(get_services),
) -> IngestResponse:
    normalized_key = _normalize_idempotency_key(idempotency_key)

    ingest_response, replayed = await _ingest_upload(
        upload=file,
        idempotency_key=normalized_key,
        services=services,
    )
    if replayed:
        response.status_code = 200
    return ingest_response


@app.get("/ingest/{document_id}", response_model=IngestResponse)
def get_ingest_status(
    document_id: str,
    services: ApiServices = Depends(get_services),
) -> IngestResponse:
    record = services.ingestion_records.get(document_id)
    if record is None:
        raise ApiError(
            code="document_not_found",
            message=f"document not found: {document_id}",
            status_code=404,
        )
    return record


@app.post(
    "/upload",
    response_model=IngestResponse | UploadBatchResponse,
    status_code=201,
)
async def upload(
    response: Response,
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    services: ApiServices = Depends(get_services),
) -> IngestResponse | UploadBatchResponse:
    normalized_key = _normalize_idempotency_key(idempotency_key)
    uploads = _normalize_upload_inputs(file=file, files=files)
    if not uploads:
        raise ApiError(
            code="no_files_uploaded",
            message="at least one file is required",
            status_code=400,
        )

    responses: list[IngestResponse] = []
    replayed_count = 0
    for index, upload_file in enumerate(uploads, start=1):
        key = _compose_upload_key(
            base_key=normalized_key,
            sequence=index,
            total_count=len(uploads),
        )
        ingest_response, replayed = await _ingest_upload(
            upload=upload_file,
            idempotency_key=key,
            services=services,
        )
        responses.append(ingest_response)
        if replayed:
            replayed_count += 1

    if len(responses) == 1:
        if replayed_count == 1:
            response.status_code = 200
        return responses[0]

    if replayed_count == len(responses):
        response.status_code = 200
    return UploadBatchResponse(documents=responses, count=len(responses))


async def _ingest_upload(
    *,
    upload: UploadFile,
    idempotency_key: str | None,
    services: ApiServices,
) -> tuple[IngestResponse, bool]:
    try:
        if idempotency_key:
            async with services.idempotency_lock:
                replay = services.idempotency_records.get(idempotency_key)
                if replay is not None:
                    await upload.close()
                    return replay, True

                receipt = await _save_upload_with_timeout(
                    intake=services.intake,
                    upload=upload,
                    timeout_seconds=services.cfg.ingest_timeout_seconds,
                )
                ingest_response = await _build_ingest_response_for_receipt(
                    receipt=receipt,
                    services=services,
                    mime_type=_normalize_upload_mime(
                        upload, receipt_file_path=str(receipt.file_path)
                    ),
                )
                services.idempotency_records[idempotency_key] = ingest_response
                return ingest_response, False

        receipt = await _save_upload_with_timeout(
            intake=services.intake,
            upload=upload,
            timeout_seconds=services.cfg.ingest_timeout_seconds,
        )
        ingest_response = await _build_ingest_response_for_receipt(
            receipt=receipt,
            services=services,
            mime_type=_normalize_upload_mime(
                upload, receipt_file_path=str(receipt.file_path)
            ),
        )
        return ingest_response, False
    except ApiError:
        raise
    except UploadIntakeError as exc:
        raise ApiError(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="ingest_failed",
            message="ingest request failed unexpectedly",
            status_code=500,
            details={"error": str(exc)},
        ) from exc


async def _build_ingest_response_for_receipt(
    *,
    receipt: Any,
    services: ApiServices,
    mime_type: str,
) -> IngestResponse:
    if services.orchestrator is None:
        response = _build_ingest_response(
            document_id=receipt.document_id,
            file_path=str(receipt.file_path),
        )
        services.ingestion_records[receipt.document_id] = response
        return response

    record = await _run_orchestration_with_timeout(
        services=services,
        document_id=receipt.document_id,
        file_path=Path(receipt.file_path),
        mime_type=mime_type,
    )
    message = _ingestion_status_message(record.status, record.error_message)
    response = IngestResponse(
        document_id=receipt.document_id,
        file_path=str(receipt.file_path),
        status=record.status,
        message=message,
    )
    services.ingestion_records[receipt.document_id] = response
    return response


@app.post("/ask", response_model=ChatResponse)
async def ask(
    payload: ChatRequest,
    services: ApiServices = Depends(get_services),
) -> ChatResponse:
    try:
        _validate_document_ready(document_id=payload.document_id, services=services)

        if payload.mode == Mode.DEEP:
            if not services.cfg.deep_mode_enabled:
                raise ApiError(
                    code="deep_mode_disabled",
                    message="deep mode is disabled in this deployment",
                    status_code=503,
                    details={"cloud_agent_provider": services.cfg.cloud_agent_provider},
                )
            return await _ask_with_timeout(
                handler=lambda: services.cloud_agent.ask(
                    question=payload.question,
                    mode=payload.mode,
                    document_id=payload.document_id,
                ),
                timeout_seconds=services.cfg.request_timeout_seconds,
            )

        return await _ask_with_timeout(
            handler=lambda: services.local_qa.ask(
                question=payload.question,
                mode=payload.mode,
                document_id=payload.document_id,
            ),
            timeout_seconds=services.cfg.request_timeout_seconds,
        )
    except IndexingError as exc:
        raise ApiError(
            code="retrieval_unavailable",
            message="retrieval backend is unavailable",
            status_code=503,
            details={"error": str(exc)},
        ) from exc
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            code="ask_failed",
            message="ask request failed unexpectedly",
            status_code=500,
            details={"error": str(exc)},
        ) from exc


def _build_ingest_response(document_id: str, file_path: str) -> IngestResponse:
    return IngestResponse(
        document_id=document_id,
        file_path=file_path,
        status=IngestionStatus.UPLOADED,
        message="queued for processing",
    )


def _normalize_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


async def _save_upload_with_timeout(
    *,
    intake: UploadIntakeService,
    upload: UploadFile,
    timeout_seconds: int,
) -> Any:
    try:
        return await asyncio.wait_for(
            intake.save_upload(upload),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(
            code="ingest_timeout",
            message="ingest request timed out",
            status_code=504,
        ) from exc


async def _run_orchestration_with_timeout(
    *,
    services: ApiServices,
    document_id: str,
    file_path: Path,
    mime_type: str,
) -> Any:
    orchestrator = services.orchestrator
    if orchestrator is None:
        raise ApiError(
            code="orchestrator_unavailable",
            message="ingestion orchestrator is not configured",
            status_code=503,
        )

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                orchestrator.process,
                document_id,
                file_path,
                mime_type,
            ),
            timeout=services.cfg.ingest_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(
            code="ingest_pipeline_timeout",
            message="ingestion pipeline timed out",
            status_code=504,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="ingest_pipeline_failed",
            message="ingestion pipeline failed unexpectedly",
            status_code=500,
            details={"error": str(exc)},
        ) from exc


async def _ask_with_timeout(*, handler: Any, timeout_seconds: int) -> ChatResponse:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(handler),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(
            code="ask_timeout",
            message="ask request timed out",
            status_code=504,
        ) from exc


def _build_error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any],
) -> JSONResponse:
    envelope = ErrorEnvelope(
        code=code,
        message=message,
        correlation_id=getattr(request.state, "correlation_id", None),
        details=details,
    )
    response = JSONResponse(status_code=status_code, content=envelope.model_dump())

    correlation_id = getattr(request.state, "correlation_id", None)
    if correlation_id:
        response.headers["x-correlation-id"] = correlation_id
    return response


def _normalize_upload_inputs(
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    if file is not None:
        uploads.append(file)
    if files:
        uploads.extend(files)
    return uploads


def _compose_upload_key(
    *,
    base_key: str | None,
    sequence: int,
    total_count: int,
) -> str | None:
    if not base_key:
        return None
    if total_count == 1:
        return base_key
    return f"{base_key}:{sequence}"


def _normalize_upload_mime(upload: UploadFile, receipt_file_path: str) -> str:
    if upload.content_type:
        return upload.content_type
    guessed = mimetypes.guess_type(receipt_file_path)[0]
    return guessed or "application/octet-stream"


def _ingestion_status_message(
    status: IngestionStatus, error_message: str | None
) -> str:
    if status == IngestionStatus.INDEXED:
        return "indexed and ready for retrieval"
    if status == IngestionStatus.PARTIAL:
        return error_message or "partially indexed; some stages failed"
    if status == IngestionStatus.FAILED:
        return error_message or "ingestion failed"
    if status == IngestionStatus.PROCESSING:
        return "ingestion is processing"
    return "queued for processing"


def _validate_document_ready(*, document_id: str | None, services: ApiServices) -> None:
    if not document_id:
        return
    if services.orchestrator is None:
        return

    ingest = services.ingestion_records.get(document_id)
    if ingest is None:
        return

    if ingest.status == IngestionStatus.INDEXED:
        return

    raise ApiError(
        code="document_not_ready",
        message=f"document status is {ingest.status.value}; ready status is indexed",
        status_code=409,
        details={"document_id": document_id, "status": ingest.status.value},
    )


def _runtime_dependency_report() -> dict[str, bool]:
    modules = {
        "redisvl": "redisvl",
        "pymupdf": "pymupdf",
        "pillow": "PIL",
        "pytesseract": "pytesseract",
        "docling": "docling",
    }
    return {
        label: importlib.util.find_spec(module_name) is not None
        for label, module_name in modules.items()
    }
