from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config.settings import Settings, settings
from src.engine.cloud_agent import CloudAgentEngine, CloudAgentModelClient
from src.engine.local_llm import LocalQAEngine
from src.ingestion import (
    HybridVectorIndexStore,
    InMemoryIndexBackend,
    IndexingError,
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


@dataclass
class ApiServices:
    cfg: Settings
    intake: UploadIntakeService
    index_store: HybridVectorIndexStore
    local_qa: LocalQAEngine
    cloud_agent: CloudAgentEngine
    idempotency_records: dict[str, IngestResponse] = field(default_factory=dict)
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

    return ApiServices(
        cfg=cfg,
        intake=UploadIntakeService(cfg),
        index_store=index_store,
        local_qa=LocalQAEngine(index_store=index_store, cfg=cfg),
        cloud_agent=CloudAgentEngine(model_client=_FallbackCloudModelClient(), cfg=cfg),
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
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    response: Response,
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    services: ApiServices = Depends(get_services),
) -> IngestResponse:
    normalized_key = _normalize_idempotency_key(idempotency_key)

    try:
        if normalized_key:
            async with services.idempotency_lock:
                replay = services.idempotency_records.get(normalized_key)
                if replay is not None:
                    await file.close()
                    response.status_code = 200
                    return replay

                receipt = await _save_upload_with_timeout(
                    intake=services.intake,
                    upload=file,
                    timeout_seconds=services.cfg.ingest_timeout_seconds,
                )
                ingest_response = _build_ingest_response(
                    document_id=receipt.document_id,
                    file_path=str(receipt.file_path),
                )
                services.idempotency_records[normalized_key] = ingest_response
                return ingest_response

        receipt = await _save_upload_with_timeout(
            intake=services.intake,
            upload=file,
            timeout_seconds=services.cfg.ingest_timeout_seconds,
        )
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

    return _build_ingest_response(
        document_id=receipt.document_id,
        file_path=str(receipt.file_path),
    )


@app.post("/ask", response_model=ChatResponse)
async def ask(
    payload: ChatRequest,
    services: ApiServices = Depends(get_services),
) -> ChatResponse:
    try:
        if payload.mode == Mode.DEEP:
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
