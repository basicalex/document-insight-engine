from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from src.config.settings import Settings, settings
from src.api.ingestion_queue import (
    IngestionQueueBackendError,
    IngestionWorkerPool,
    build_worker_pool_with_fallback,
)
from src.api.state_store import (
    ApiStateStore,
    ApiStateStoreError,
    InMemoryApiStateStore,
    RedisApiStateStore,
)
from src.api.telemetry import ObservabilityRegistry, ObservabilitySLOs
from src.engine.cloud_agent import (
    CloudAgentEngine,
    CloudAgentModelClient,
    DeepProviderErrorCode,
)
from src.engine.extractor import Tier4StructuredExtractor
from src.engine.gemini_client import GeminiCloudModelClient
from src.engine.local_agent_client import LocalDeepModelClient
from src.engine.local_llm import (
    GeminiTextGenerationClient,
    LocalQAEngine,
    ProviderQueryEmbedder,
)
from src.ingestion.embeddings import (
    build_ingestion_embedding_clients,
    build_query_embedding_clients,
)
from src.ingestion import (
    BestEffortParser,
    BestEffortTextExtractor,
    HybridVectorIndexStore,
    IngestionOrchestrator,
    InMemoryIndexBackend,
    IndexingError,
    ParentChildChunker,
    ProviderIngestionEmbedder,
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
    StructuredExtractRequest,
    StructuredExtractResponse,
    StructuredFieldProvenance,
    StructuredValidationDiagnostic,
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
    index_readiness: dict[str, Any]
    local_qa: LocalQAEngine
    cloud_agent: CloudAgentEngine
    state_store: ApiStateStore
    structured_extractor: Tier4StructuredExtractor | None = None
    orchestrator: IngestionOrchestrator | None = None
    ingestion_worker_pool: IngestionWorkerPool | None = None
    telemetry: ObservabilityRegistry = field(default_factory=ObservabilityRegistry)


@dataclass(frozen=True)
class _ChatModelRouting:
    backend: str
    use_api_model: bool
    api_key: str | None
    api_model: str


_DEEP_PROVIDER_FAILURE_CODES = {
    DeepProviderErrorCode.NOT_CONFIGURED.value,
    DeepProviderErrorCode.AUTHENTICATION_FAILED.value,
    DeepProviderErrorCode.RATE_LIMITED.value,
    DeepProviderErrorCode.TIMEOUT.value,
    DeepProviderErrorCode.UNAVAILABLE.value,
    DeepProviderErrorCode.MALFORMED_RESPONSE.value,
}


app = FastAPI(title=settings.project_name)


def get_app_settings() -> Settings:
    return settings


def _build_state_store(cfg: Settings) -> ApiStateStore:
    if cfg.api_state_backend == "memory":
        return InMemoryApiStateStore(
            idempotency_ttl_seconds=cfg.api_state_idempotency_ttl_seconds,
            ingestion_ttl_seconds=cfg.api_state_ingestion_ttl_seconds,
            session_ttl_seconds=cfg.api_state_session_ttl_seconds,
            idempotency_claim_ttl_seconds=cfg.api_state_idempotency_claim_ttl_seconds,
        )

    try:
        return RedisApiStateStore(
            redis_url=cfg.redis_url,
            key_prefix=cfg.api_state_key_prefix,
            idempotency_ttl_seconds=cfg.api_state_idempotency_ttl_seconds,
            ingestion_ttl_seconds=cfg.api_state_ingestion_ttl_seconds,
            session_ttl_seconds=cfg.api_state_session_ttl_seconds,
            idempotency_claim_ttl_seconds=cfg.api_state_idempotency_claim_ttl_seconds,
        )
    except ApiStateStoreError as exc:
        if cfg.api_state_backend == "redis":
            raise
        logger.warning(
            "Falling back to in-memory API state store: %s",
            exc,
        )
        return InMemoryApiStateStore(
            idempotency_ttl_seconds=cfg.api_state_idempotency_ttl_seconds,
            ingestion_ttl_seconds=cfg.api_state_ingestion_ttl_seconds,
            session_ttl_seconds=cfg.api_state_session_ttl_seconds,
            idempotency_claim_ttl_seconds=cfg.api_state_idempotency_claim_ttl_seconds,
        )
    raise RuntimeError("unreachable state store configuration")


def _index_fallback_allowed(cfg: Settings) -> bool:
    return cfg.allow_in_memory_index_fallback and cfg.environment in {"local", "dev"}


def _build_ingestion_worker_pool(
    *,
    cfg: Settings,
    state_store: ApiStateStore,
    orchestrator: IngestionOrchestrator,
    telemetry: ObservabilityRegistry,
) -> IngestionWorkerPool:
    try:
        return build_worker_pool_with_fallback(
            requested_backend=cfg.ingestion_queue_backend,
            redis_url=cfg.redis_url,
            key_prefix=cfg.ingestion_queue_key_prefix,
            dead_letter_max_items=cfg.ingestion_queue_dead_letter_max_items,
            state_store=state_store,
            orchestrator=orchestrator,
            worker_concurrency=cfg.ingestion_worker_concurrency,
            max_retries=cfg.ingestion_queue_max_retries,
            retry_backoff_seconds=cfg.ingestion_queue_retry_backoff_seconds,
            poll_timeout_seconds=cfg.ingestion_queue_poll_timeout_seconds,
            ingest_timeout_seconds=cfg.ingest_timeout_seconds,
            telemetry=telemetry,
        )
    except IngestionQueueBackendError:
        raise
    except Exception as exc:
        raise IngestionQueueBackendError(
            f"failed to configure ingestion queue worker pool: {exc}"
        ) from exc


def _build_telemetry(cfg: Settings) -> ObservabilityRegistry:
    return ObservabilityRegistry(
        slos=ObservabilitySLOs(
            http_request_p95_ms=cfg.slo_http_request_p95_ms,
            retrieval_p95_ms=cfg.slo_retrieval_p95_ms,
            generation_p95_ms=cfg.slo_generation_p95_ms,
            insufficient_evidence_rate_max=cfg.slo_insufficient_evidence_rate_max,
            citation_completeness_min=cfg.slo_citation_completeness_min,
            grounding_gap_rate_max=cfg.slo_grounding_gap_rate_max,
        )
    )


def _build_index_store(cfg: Settings) -> tuple[HybridVectorIndexStore, dict[str, Any]]:
    fallback_allowed = _index_fallback_allowed(cfg)
    logger.info(
        "Connecting to Redis at %s (retries=%d)",
        cfg.redis_url,
        cfg.redis_connection_retries,
    )
    try:
        index_store = HybridVectorIndexStore(cfg=cfg)
        index_store.bootstrap_indices()
        logger.info("Redis index backend initialized successfully")
        return index_store, {
            "state": "ready",
            "backend": type(index_store.backend).__name__,
            "reason": "redis_ready",
            "fallback_allowed": fallback_allowed,
            "degraded": False,
        }
    except Exception as exc:
        if not fallback_allowed:
            raise RuntimeError(
                "Redis index backend is unavailable and in-memory fallback is disabled. "
                "Set ALLOW_IN_MEMORY_INDEX_FALLBACK=true only for local/dev troubleshooting."
            ) from exc

        logger.warning(
            "Redis index backend unavailable; using in-memory fallback because "
            "ALLOW_IN_MEMORY_INDEX_FALLBACK is enabled: %s",
            exc,
        )
        in_memory = InMemoryIndexBackend()
        index_store = HybridVectorIndexStore(cfg=cfg, backend=in_memory)
        index_store.bootstrap_indices()
        return index_store, {
            "state": "degraded",
            "backend": type(index_store.backend).__name__,
            "reason": "redis_unavailable_fallback_enabled",
            "fallback_allowed": fallback_allowed,
            "degraded": True,
            "error": str(exc),
        }


def _build_default_services(cfg: Settings) -> ApiServices:
    telemetry = _build_telemetry(cfg)
    index_store, index_readiness = _build_index_store(cfg)

    extractor = BestEffortTextExtractor()
    parser = BestEffortParser(
        fallback_extractor=extractor,
        docling_enabled=cfg.docling_enabled,
        google_enabled=cfg.google_parser_enabled,
        parser_order=_parser_order_for_mode(cfg.parser_routing_mode),
        parsed_dir=cfg.parsed_dir,
    )
    chunker = ParentChildChunker()
    tier1_embed_client, tier4_embed_client = build_ingestion_embedding_clients(cfg)
    query_primary_client, query_fallback_client = build_query_embedding_clients(cfg)
    embedder = ProviderIngestionEmbedder(
        tier1_client=tier1_embed_client,
        tier4_client=tier4_embed_client,
    )

    model_client: CloudAgentModelClient
    if cfg.cloud_agent_provider == "gemini":
        model_client = GeminiCloudModelClient(cfg=cfg)
    elif cfg.cloud_agent_provider == "local":
        model_client = LocalDeepModelClient(cfg=cfg)
    elif cfg.cloud_agent_provider == "fallback":
        model_client = _FallbackCloudModelClient()
    else:
        model_client = _DisabledCloudModelClient()

    provider_diag = _deep_provider_diagnostics(cfg)
    if cfg.deep_mode_enabled and not provider_diag["ready"]:
        logger.warning(
            "Deep mode enabled but provider is not ready: %s",
            provider_diag,
        )
    else:
        logger.info("Deep provider diagnostics: %s", provider_diag)

    _log_runtime_capabilities(cfg)

    state_store = _build_state_store(cfg)
    orchestrator = IngestionOrchestrator(
        extractor=extractor,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        index_store=index_store,
        max_retries_per_stage=2,
        retry_backoff_seconds=0.0,
    )
    ingestion_worker_pool = _build_ingestion_worker_pool(
        cfg=cfg,
        state_store=state_store,
        orchestrator=orchestrator,
        telemetry=telemetry,
    )

    return ApiServices(
        cfg=cfg,
        intake=UploadIntakeService(cfg),
        index_store=index_store,
        index_readiness=index_readiness,
        local_qa=LocalQAEngine(
            index_store=index_store,
            cfg=cfg,
            embedder=ProviderQueryEmbedder(
                primary_client=query_primary_client,
                fallback_client=query_fallback_client,
            ),
            query_vector_dimension=cfg.local_embedding_dimension,
        ),
        cloud_agent=CloudAgentEngine(model_client=model_client, cfg=cfg),
        state_store=state_store,
        structured_extractor=Tier4StructuredExtractor(
            cfg=cfg,
            max_input_tokens=cfg.extraction_max_input_tokens,
            max_output_tokens=cfg.extraction_max_output_tokens,
            strict_schema=cfg.extraction_strict_schema,
        ),
        orchestrator=orchestrator,
        ingestion_worker_pool=ingestion_worker_pool,
        telemetry=telemetry,
    )


def _ensure_ingestion_workers_running(services: ApiServices) -> None:
    if services.orchestrator is None:
        return

    if services.ingestion_worker_pool is None:
        services.ingestion_worker_pool = _build_ingestion_worker_pool(
            cfg=services.cfg,
            state_store=services.state_store,
            orchestrator=services.orchestrator,
            telemetry=services.telemetry,
        )

    if not services.ingestion_worker_pool.is_running:
        services.ingestion_worker_pool.start()


def get_services(request: Request) -> ApiServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        services = _build_default_services(get_app_settings())
        request.app.state.services = services
    _ensure_ingestion_workers_running(services)
    return services


@app.on_event("startup")
def ensure_runtime_dirs() -> None:
    settings.ensure_runtime_dirs()
    if not hasattr(app.state, "services"):
        app.state.services = _build_default_services(settings)
    _ensure_ingestion_workers_running(app.state.services)


@app.on_event("shutdown")
async def close_runtime_services() -> None:
    services = getattr(app.state, "services", None)
    if services is None:
        return
    if services.ingestion_worker_pool is not None:
        try:
            await services.ingestion_worker_pool.stop()
        except Exception:
            logger.warning(
                "Failed to stop ingestion worker pool cleanly", exc_info=True
            )
    try:
        await services.state_store.close()
    except Exception:
        logger.warning("Failed to close API state store cleanly", exc_info=True)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
    started = time.perf_counter()
    correlation_id = request.headers.get("x-correlation-id") or uuid4().hex
    request.state.correlation_id = correlation_id

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_http_telemetry(
            request=request,
            status_code=500,
            latency_ms=latency_ms,
        )
        raise

    latency_ms = int((time.perf_counter() - started) * 1000)
    _record_http_telemetry(
        request=request,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    response.headers["x-correlation-id"] = correlation_id
    return response


def _record_http_telemetry(
    *,
    request: Request,
    status_code: int,
    latency_ms: int,
) -> None:
    services = getattr(request.app.state, "services", None)
    if services is None or not isinstance(services, ApiServices):
        return

    route = request.scope.get("route")
    route_path = getattr(route, "path", request.url.path)
    services.telemetry.record_http_request(
        route=str(route_path),
        method=request.method,
        status_code=status_code,
        latency_ms=latency_ms,
    )


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


def _ingestion_queue_report(services: ApiServices) -> dict[str, Any]:
    if services.orchestrator is None:
        return {
            "enabled": False,
            "reason": "orchestrator_not_configured",
        }

    pool = services.ingestion_worker_pool
    if pool is None:
        return {
            "enabled": False,
            "reason": "worker_pool_unavailable",
        }

    report = pool.diagnostics()
    report["enabled"] = True
    return report


@app.get("/healthz")
def healthz(services: ApiServices = Depends(get_services)) -> dict[str, Any]:
    backend_name = type(services.index_store.backend).__name__
    state_backend_name = type(services.state_store).__name__
    readiness = _runtime_readiness_report(services)
    return {
        "status": "ok",
        "index_backend": backend_name,
        "state_backend": state_backend_name,
        "readiness": readiness,
        "orchestrator_configured": services.orchestrator is not None,
        "deep_mode_enabled": services.cfg.deep_mode_enabled,
        "cloud_agent_provider": services.cfg.cloud_agent_provider,
        "parser_routing_mode": services.cfg.parser_routing_mode,
        "deep_provider": _deep_provider_diagnostics(services.cfg),
        "dependencies": _runtime_dependency_report(),
        "capabilities": _runtime_capabilities_report(services.cfg),
        "ingestion_queue": _ingestion_queue_report(services),
        "observability": services.telemetry.snapshot(),
    }


@app.get("/readyz")
def readyz(
    response: Response,
    services: ApiServices = Depends(get_services),
) -> dict[str, Any]:
    readiness = _runtime_readiness_report(services)
    if readiness["overall"] != "ready":
        response.status_code = 503
    return readiness


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(services: ApiServices = Depends(get_services)) -> str:
    return services.telemetry.render_prometheus()


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
async def get_ingest_status(
    document_id: str,
    services: ApiServices = Depends(get_services),
) -> IngestResponse:
    record = await services.state_store.get_ingestion_record(document_id)
    if record is None:
        raise ApiError(
            code="document_not_found",
            message=f"document not found: {document_id}",
            status_code=404,
        )
    return record


@app.post("/extract", response_model=StructuredExtractResponse)
async def extract_structured_fields(
    payload: StructuredExtractRequest,
    services: ApiServices = Depends(get_services),
) -> StructuredExtractResponse:
    await _validate_document_ready(document_id=payload.document_id, services=services)
    document_text = await _load_document_text_for_extraction(
        document_id=payload.document_id,
        services=services,
    )

    extractor = services.structured_extractor or Tier4StructuredExtractor(
        cfg=services.cfg,
        max_input_tokens=services.cfg.extraction_max_input_tokens,
        max_output_tokens=services.cfg.extraction_max_output_tokens,
        strict_schema=services.cfg.extraction_strict_schema,
    )
    envelope = await asyncio.to_thread(
        extractor.extract_structured,
        document_id=payload.document_id,
        document_text=document_text,
        schema=payload.extraction_schema,
        prompt=payload.prompt,
    )

    if not envelope.ok or envelope.result is None:
        error = envelope.error
        if error is None:
            raise ApiError(
                code="structured_extraction_failed",
                message="structured extraction failed unexpectedly",
                status_code=500,
            )
        status_map = {
            "invalid_schema": 422,
            "token_budget_exceeded": 422,
            "provider_error": 503,
            "provider_disabled": 503,
            "validation_failed": 422,
        }
        raise ApiError(
            code=error.code,
            message=error.message,
            status_code=status_map.get(error.code, 500),
            details={
                "diagnostics": [
                    {
                        "code": item.code,
                        "message": item.message,
                        "field": item.field,
                        "details": item.details,
                    }
                    for item in error.diagnostics
                ],
                "token_usage": error.token_usage,
                "latency_ms": error.latency_ms,
            },
        )

    artifact_path = _persist_extraction_artifact(
        document_id=payload.document_id,
        prompt=payload.prompt,
        schema=payload.extraction_schema,
        response=StructuredExtractResponse(
            document_id=envelope.result.document_id,
            model=envelope.result.model,
            prompt_version=envelope.result.prompt_version,
            data=envelope.result.data,
            provenance={
                key: StructuredFieldProvenance(
                    start_offset=value.start_offset,
                    end_offset=value.end_offset,
                    text=value.text,
                )
                for key, value in envelope.result.provenance.items()
            },
            accepted_fields=envelope.result.accepted_fields,
            rejected_fields=envelope.result.rejected_fields,
            diagnostics=[
                StructuredValidationDiagnostic(
                    code=item.code,
                    message=item.message,
                    field=item.field,
                    details=item.details,
                )
                for item in envelope.result.diagnostics
            ],
            token_usage=envelope.result.token_usage,
            latency_ms=envelope.result.latency_ms,
            artifact_path="pending",
        ),
        cfg=services.cfg,
    )

    return StructuredExtractResponse(
        document_id=envelope.result.document_id,
        model=envelope.result.model,
        prompt_version=envelope.result.prompt_version,
        data=envelope.result.data,
        provenance={
            key: StructuredFieldProvenance(
                start_offset=value.start_offset,
                end_offset=value.end_offset,
                text=value.text,
            )
            for key, value in envelope.result.provenance.items()
        },
        accepted_fields=envelope.result.accepted_fields,
        rejected_fields=envelope.result.rejected_fields,
        diagnostics=[
            StructuredValidationDiagnostic(
                code=item.code,
                message=item.message,
                field=item.field,
                details=item.details,
            )
            for item in envelope.result.diagnostics
        ],
        token_usage=envelope.result.token_usage,
        latency_ms=envelope.result.latency_ms,
        artifact_path=artifact_path,
    )


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
    claim_acquired = False
    try:
        if idempotency_key:
            replay = await services.state_store.get_idempotency_response(
                idempotency_key
            )
            if replay is not None:
                await upload.close()
                return replay, True

            claim_acquired = await services.state_store.claim_idempotency_key(
                idempotency_key
            )
            if not claim_acquired:
                replay = await _wait_for_idempotency_replay(
                    services=services,
                    idempotency_key=idempotency_key,
                )
                await upload.close()
                if replay is not None:
                    return replay, True
                raise ApiError(
                    code="idempotency_conflict",
                    message="request with the same Idempotency-Key is in progress",
                    status_code=409,
                )

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
                idempotency_key=idempotency_key,
            )
            await services.state_store.put_idempotency_response(
                idempotency_key,
                ingest_response,
            )
            claim_acquired = False
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
            idempotency_key=idempotency_key,
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
    finally:
        if idempotency_key and claim_acquired:
            await services.state_store.release_idempotency_key(idempotency_key)


async def _wait_for_idempotency_replay(
    *,
    services: ApiServices,
    idempotency_key: str,
) -> IngestResponse | None:
    deadline = time.monotonic() + min(
        float(services.cfg.request_timeout_seconds),
        float(services.cfg.api_state_idempotency_claim_ttl_seconds),
    )
    while time.monotonic() < deadline:
        replay = await services.state_store.get_idempotency_response(idempotency_key)
        if replay is not None:
            return replay
        await asyncio.sleep(0.05)
    return None


async def _build_ingest_response_for_receipt(
    *,
    receipt: Any,
    services: ApiServices,
    mime_type: str,
    idempotency_key: str | None,
) -> IngestResponse:
    if services.orchestrator is None:
        response = _build_ingest_response(
            document_id=receipt.document_id,
            file_path=str(receipt.file_path),
        )
        await services.state_store.put_ingestion_record(response)
        services.telemetry.record_ingestion_status(response.status)
        return response

    response = _build_ingest_response(
        document_id=receipt.document_id,
        file_path=str(receipt.file_path),
    )
    await services.state_store.put_ingestion_record(response)
    services.telemetry.record_ingestion_status(response.status)

    worker_pool = services.ingestion_worker_pool
    if worker_pool is None:
        record = await _run_orchestration_with_timeout(
            services=services,
            document_id=receipt.document_id,
            file_path=Path(receipt.file_path),
            mime_type=mime_type,
        )
        message = _ingestion_status_message(record.status, record.error_message)
        synchronous_response = IngestResponse(
            document_id=receipt.document_id,
            file_path=str(receipt.file_path),
            status=record.status,
            message=message,
        )
        await services.state_store.put_ingestion_record(synchronous_response)
        services.telemetry.record_ingestion_status(synchronous_response.status)
        return synchronous_response

    await worker_pool.enqueue(
        document_id=receipt.document_id,
        file_path=Path(receipt.file_path),
        mime_type=mime_type,
        idempotency_key=idempotency_key,
    )
    return response


@app.post("/ask", response_model=ChatResponse)
async def ask(
    payload: ChatRequest,
    request: Request,
    response: Response,
    model_backend: str | None = Header(default=None, alias="x-model-backend"),
    api_key: str | None = Header(default=None, alias="x-api-key"),
    api_model: str | None = Header(default=None, alias="x-api-model"),
    services: ApiServices = Depends(get_services),
) -> ChatResponse:
    try:
        await _validate_document_ready(
            document_id=payload.document_id, services=services
        )
        chat_history = await _read_session_history(
            services=services,
            session_id=payload.session_id,
            document_id=payload.document_id,
        )
        routing = _resolve_chat_model_routing(
            cfg=services.cfg,
            model_backend_header=model_backend,
            api_key_header=api_key,
            api_model_header=api_model,
        )

        chat_response: ChatResponse

        if payload.mode == Mode.DEEP:
            if not services.cfg.deep_mode_enabled:
                raise ApiError(
                    code="deep_mode_disabled",
                    message="deep mode is disabled in this deployment",
                    status_code=503,
                    details={"cloud_agent_provider": services.cfg.cloud_agent_provider},
                )
            deep_engine = _resolve_deep_engine_for_request(
                services=services,
                routing=routing,
            )
            chat_response = await _ask_with_timeout(
                handler=lambda: deep_engine.ask(
                    question=payload.question,
                    mode=payload.mode,
                    document_id=payload.document_id,
                ),
                timeout_seconds=services.cfg.request_timeout_seconds,
            )

            if _should_auto_fallback_to_local(
                routing=routing,
                response=chat_response,
                mode=Mode.DEEP,
            ):
                logger.warning(
                    "Deep API model failed (%s); retrying with local backend",
                    _termination_reason(chat_response),
                )
                fallback_engine = _resolve_deep_engine_for_request(
                    services=services,
                    routing=_local_chat_routing(services.cfg),
                )
                chat_response = await _ask_with_timeout(
                    handler=lambda: fallback_engine.ask(
                        question=payload.question,
                        mode=payload.mode,
                        document_id=payload.document_id,
                    ),
                    timeout_seconds=services.cfg.request_timeout_seconds,
                )

            _raise_if_deep_provider_failed(chat_response)
        else:
            fast_engine = _resolve_fast_engine_for_request(
                services=services,
                routing=routing,
            )
            chat_response = await _ask_with_timeout(
                handler=lambda: fast_engine.ask(
                    question=payload.question,
                    mode=payload.mode,
                    document_id=payload.document_id,
                    chat_history=chat_history,
                ),
                timeout_seconds=services.cfg.request_timeout_seconds,
            )

            if _should_auto_fallback_to_local(
                routing=routing,
                response=chat_response,
                mode=Mode.FAST,
            ):
                logger.warning(
                    "Fast API model failed (%s); retrying with local backend",
                    _termination_reason(chat_response),
                )
                fallback_engine = _resolve_fast_engine_for_request(
                    services=services,
                    routing=_local_chat_routing(services.cfg),
                )
                chat_response = await _ask_with_timeout(
                    handler=lambda: fallback_engine.ask(
                        question=payload.question,
                        mode=payload.mode,
                        document_id=payload.document_id,
                        chat_history=chat_history,
                    ),
                    timeout_seconds=services.cfg.request_timeout_seconds,
                )

        await _record_session_turn(
            services=services,
            session_id=payload.session_id,
            document_id=payload.document_id,
            question=payload.question,
            answer=chat_response.answer,
        )
        _record_qa_telemetry(
            services=services,
            request=request,
            chat_response=chat_response,
        )
        if chat_response.trace is not None:
            response.headers["x-trace-id"] = chat_response.trace.trace_id
        return chat_response
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


@app.post("/ask/stream")
async def ask_stream(
    payload: ChatRequest,
    request: Request,
    model_backend: str | None = Header(default=None, alias="x-model-backend"),
    api_key: str | None = Header(default=None, alias="x-api-key"),
    api_model: str | None = Header(default=None, alias="x-api-model"),
    services: ApiServices = Depends(get_services),
) -> StreamingResponse:
    try:
        await _validate_document_ready(
            document_id=payload.document_id, services=services
        )
        chat_history = await _read_session_history(
            services=services,
            session_id=payload.session_id,
            document_id=payload.document_id,
        )
        routing = _resolve_chat_model_routing(
            cfg=services.cfg,
            model_backend_header=model_backend,
            api_key_header=api_key,
            api_model_header=api_model,
        )

        if payload.mode == Mode.DEEP:
            if not services.cfg.deep_mode_enabled:
                raise ApiError(
                    code="deep_mode_disabled",
                    message="deep mode is disabled in this deployment",
                    status_code=503,
                    details={"cloud_agent_provider": services.cfg.cloud_agent_provider},
                )
            deep_engine = _resolve_deep_engine_for_request(
                services=services,
                routing=routing,
            )
            response = await _ask_with_timeout(
                handler=lambda: deep_engine.ask(
                    question=payload.question,
                    mode=payload.mode,
                    document_id=payload.document_id,
                ),
                timeout_seconds=services.cfg.request_timeout_seconds,
            )

            if _should_auto_fallback_to_local(
                routing=routing,
                response=response,
                mode=Mode.DEEP,
            ):
                logger.warning(
                    "Deep API model failed (%s); retrying stream response with local backend",
                    _termination_reason(response),
                )
                fallback_engine = _resolve_deep_engine_for_request(
                    services=services,
                    routing=_local_chat_routing(services.cfg),
                )
                response = await _ask_with_timeout(
                    handler=lambda: fallback_engine.ask(
                        question=payload.question,
                        mode=payload.mode,
                        document_id=payload.document_id,
                    ),
                    timeout_seconds=services.cfg.request_timeout_seconds,
                )

            _raise_if_deep_provider_failed(response)
            events = _single_response_events(response)
        else:
            fast_engine = _resolve_fast_engine_for_request(
                services=services,
                routing=routing,
            )
            if routing.backend == "auto" and routing.use_api_model:
                response = await _ask_with_timeout(
                    handler=lambda: fast_engine.ask(
                        question=payload.question,
                        mode=payload.mode,
                        document_id=payload.document_id,
                        chat_history=chat_history,
                    ),
                    timeout_seconds=services.cfg.request_timeout_seconds,
                )
                if _should_auto_fallback_to_local(
                    routing=routing,
                    response=response,
                    mode=Mode.FAST,
                ):
                    logger.warning(
                        "Fast API model failed (%s); retrying stream response with local backend",
                        _termination_reason(response),
                    )
                    fallback_engine = _resolve_fast_engine_for_request(
                        services=services,
                        routing=_local_chat_routing(services.cfg),
                    )
                    response = await _ask_with_timeout(
                        handler=lambda: fallback_engine.ask(
                            question=payload.question,
                            mode=payload.mode,
                            document_id=payload.document_id,
                            chat_history=chat_history,
                        ),
                        timeout_seconds=services.cfg.request_timeout_seconds,
                    )
                events = _single_response_events(response)
            else:
                events = fast_engine.ask_stream_events(
                    question=payload.question,
                    mode=payload.mode,
                    document_id=payload.document_id,
                    chat_history=chat_history,
                )

        async def stream_body() -> AsyncIterator[bytes]:
            streamed_tokens: list[str] = []
            final_answer: str | None = None
            final_chat_response: ChatResponse | None = None

            async for event in iterate_in_threadpool(events):
                if not isinstance(event, dict):
                    continue

                event_type = str(event.get("type", "")).strip().lower()
                if event_type == "token":
                    delta = str(event.get("delta", ""))
                    if delta:
                        streamed_tokens.append(delta)
                elif event_type == "final":
                    response_payload = event.get("response")
                    if isinstance(response_payload, dict):
                        try:
                            final_chat_response = ChatResponse.model_validate(
                                response_payload
                            )
                        except Exception:
                            final_chat_response = None
                        answer = str(response_payload.get("answer", "")).strip()
                        if answer:
                            final_answer = answer

                yield _serialize_stream_event(event)

            if final_answer is None:
                joined = "".join(streamed_tokens).strip()
                if joined:
                    final_answer = joined

            if final_answer:
                await _record_session_turn(
                    services=services,
                    session_id=payload.session_id,
                    document_id=payload.document_id,
                    question=payload.question,
                    answer=final_answer,
                )

            if final_chat_response is not None:
                _record_qa_telemetry(
                    services=services,
                    request=request,
                    chat_response=final_chat_response,
                )

        return StreamingResponse(
            stream_body(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
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
        return await asyncio.to_thread(
            orchestrator.process,
            document_id,
            file_path,
            mime_type,
        )
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


def _single_response_events(response: ChatResponse) -> Any:
    yield {
        "type": "status",
        "phase": "generation",
        "message": "Generating deep-mode response...",
    }
    for delta in _chunk_stream_text(response.answer, chunk_size=8):
        yield {"type": "token", "delta": delta}
    yield {"type": "final", "response": response.model_dump(mode="json")}


def _chunk_stream_text(text: str, *, chunk_size: int) -> Any:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not text:
        return
    for start in range(0, len(text), chunk_size):
        delta = text[start : start + chunk_size]
        if delta:
            yield delta


def _serialize_stream_event(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _record_qa_telemetry(
    *,
    services: ApiServices,
    request: Request,
    chat_response: ChatResponse,
) -> None:
    services.telemetry.record_chat_response(chat_response)
    trace_id = chat_response.trace.trace_id if chat_response.trace else None
    services.telemetry.record_trace_link(
        correlation_id=getattr(request.state, "correlation_id", None),
        trace_id=trace_id,
    )


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


def _raise_if_deep_provider_failed(response: ChatResponse) -> None:
    termination_reason = response.trace.termination_reason if response.trace else None
    if not termination_reason:
        return

    status_by_code = {
        DeepProviderErrorCode.NOT_CONFIGURED.value: 503,
        DeepProviderErrorCode.AUTHENTICATION_FAILED.value: 503,
        DeepProviderErrorCode.RATE_LIMITED.value: 429,
        DeepProviderErrorCode.TIMEOUT.value: 504,
        DeepProviderErrorCode.UNAVAILABLE.value: 503,
        DeepProviderErrorCode.MALFORMED_RESPONSE.value: 502,
    }
    status_code = status_by_code.get(termination_reason)
    if status_code is None:
        return

    raise ApiError(
        code=termination_reason,
        message="deep mode provider request failed",
        status_code=status_code,
        details={"provider_message": response.answer},
    )


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


async def _validate_document_ready(
    *, document_id: str | None, services: ApiServices
) -> None:
    if not document_id:
        return
    if services.orchestrator is None:
        return

    ingest = await services.state_store.get_ingestion_record(document_id)
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


async def _load_document_text_for_extraction(
    *,
    document_id: str,
    services: ApiServices,
) -> str:
    record = await services.state_store.get_ingestion_record(document_id)
    if record is None:
        raise ApiError(
            code="document_not_found",
            message=f"document not found: {document_id}",
            status_code=404,
        )

    file_path = Path(record.file_path)
    if not file_path.exists():
        raise ApiError(
            code="document_file_missing",
            message="stored document file is missing on disk",
            status_code=404,
            details={"document_id": document_id, "file_path": str(file_path)},
        )

    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    extractor = BestEffortTextExtractor()
    extraction = await asyncio.to_thread(
        extractor.extract,
        file_path=file_path,
        mime_type=mime_type,
    )
    document_text = extraction.text.strip()
    if not document_text:
        raise ApiError(
            code="document_text_unavailable",
            message="document did not produce extractable text",
            status_code=422,
            details={"document_id": document_id},
        )
    return document_text


def _persist_extraction_artifact(
    *,
    document_id: str,
    prompt: str | None,
    schema: dict[str, Any],
    response: StructuredExtractResponse,
    cfg: Settings,
) -> str:
    artifact_dir = cfg.parsed_dir / "extractions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{document_id}_{uuid4().hex}.json"
    response_payload = response.model_dump(mode="json")
    response_payload["artifact_path"] = str(artifact_path)

    payload = {
        "document_id": document_id,
        "prompt": prompt,
        "schema": schema,
        "response": response_payload,
    }
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(artifact_path)


def _runtime_dependency_report() -> dict[str, bool]:
    modules = {
        "redisvl": "redisvl",
        "pymupdf": "pymupdf",
        "pillow": "PIL",
        "pytesseract": "pytesseract",
        "docling": "docling",
        "langextract": "langextract",
    }
    return {
        label: importlib.util.find_spec(module_name) is not None
        for label, module_name in modules.items()
    }


def _runtime_capabilities_report(cfg: Settings) -> dict[str, dict[str, Any]]:
    return {
        "docling_parser": _docling_parser_capability_status(cfg),
        "google_parser": _google_parser_capability_status(cfg),
        "langextract_extractor": _optional_capability_status(
            enabled=cfg.langextract_enabled,
            module_name="langextract",
            package_name="langextract",
        ),
    }


def _optional_dependency_hint(*, package_name: str) -> str:
    extra_by_package = {
        "docling": "ai-docling",
        "langextract": "ai-lite",
    }
    extra = extra_by_package.get(package_name)
    if extra is None:
        return (
            "install optional runtime dependency: "
            f"pip install {package_name} (missing package: {package_name})"
        )
    return (
        "install optional runtime dependency: "
        f"pip install -e .[{extra}] (missing package: {package_name})"
    )


def _optional_capability_status(
    *,
    enabled: bool,
    module_name: str,
    package_name: str,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "ready": False,
            "reason": "disabled_by_config",
            "hint": f"set {module_name.upper()}_ENABLED=true to enable",
        }

    dependency_available = importlib.util.find_spec(module_name) is not None
    if dependency_available:
        return {
            "enabled": True,
            "ready": True,
            "reason": "ready",
            "hint": None,
        }

    return {
        "enabled": True,
        "ready": False,
        "reason": "missing_dependency",
        "hint": _optional_dependency_hint(package_name=package_name),
    }


def _docling_parser_capability_status(cfg: Settings) -> dict[str, Any]:
    if not _parser_step_is_active(cfg, "docling"):
        return {
            "enabled": False,
            "ready": False,
            "reason": "excluded_by_routing_mode",
            "hint": "set PARSER_ROUTING_MODE to include docling",
        }
    return _optional_capability_status(
        enabled=cfg.docling_enabled,
        module_name="docling",
        package_name="docling",
    )


def _log_runtime_capabilities(cfg: Settings) -> None:
    capabilities = _runtime_capabilities_report(cfg)
    for name, report in capabilities.items():
        level = (
            logger.warning if report["enabled"] and not report["ready"] else logger.info
        )
        level(
            "Runtime capability %s => enabled=%s ready=%s reason=%s hint=%s",
            name,
            report["enabled"],
            report["ready"],
            report["reason"],
            report["hint"],
        )


def _google_parser_capability_status(cfg: Settings) -> dict[str, Any]:
    if not _parser_step_is_active(cfg, "google"):
        return {
            "enabled": False,
            "ready": False,
            "reason": "excluded_by_routing_mode",
            "hint": "set PARSER_ROUTING_MODE to include google",
        }

    if not cfg.google_parser_enabled:
        return {
            "enabled": False,
            "ready": False,
            "reason": "disabled_by_config",
            "hint": "set GOOGLE_PARSER_ENABLED=true to enable",
        }

    api_key_present = bool((cfg.cloud_agent_api_key or "").strip())
    if api_key_present:
        return {
            "enabled": True,
            "ready": True,
            "reason": "ready",
            "hint": None,
        }
    return {
        "enabled": True,
        "ready": False,
        "reason": "missing_api_key",
        "hint": "set CLOUD_AGENT_API_KEY (or GOOGLE_API_KEY) for Google parser routing",
    }


def _parser_step_is_active(cfg: Settings, parser_name: str) -> bool:
    return parser_name in _parser_order_for_mode(cfg.parser_routing_mode)


def _parser_order_for_mode(mode: str) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "docling_google_fallback": ("docling", "google", "fallback"),
        "google_docling_fallback": ("google", "docling", "fallback"),
        "docling_fallback": ("docling", "fallback"),
        "google_fallback": ("google", "fallback"),
        "fallback_only": ("fallback",),
    }
    return mapping.get(mode, ("docling", "google", "fallback"))


def _runtime_readiness_report(services: ApiServices) -> dict[str, Any]:
    index = dict(services.index_readiness)
    capabilities = _runtime_capabilities_report(services.cfg)
    deep_provider = _deep_provider_diagnostics(services.cfg)
    ingest_blockers = _index_readiness_issues(index)
    fast_blockers = list(ingest_blockers)
    deep_blockers = list(fast_blockers)
    deep_blockers.extend(_deep_mode_readiness_issues(services.cfg, deep_provider))

    overall = "degraded" if bool(index.get("degraded")) else "ready"
    return {
        "overall": overall,
        "environment": services.cfg.environment,
        "index": index,
        "actions": {
            "ingest": {
                "ready": not ingest_blockers,
                "blocking_issues": ingest_blockers,
            },
            "ask_fast": {
                "ready": not fast_blockers,
                "blocking_issues": fast_blockers,
            },
            "ask_deep": {
                "ready": not deep_blockers,
                "blocking_issues": deep_blockers,
            },
        },
        "optional_capability_issues": _optional_capability_issues(capabilities),
    }


def _index_readiness_issues(index: dict[str, Any]) -> list[dict[str, str]]:
    if not bool(index.get("degraded")):
        return []

    reason = str(index.get("reason") or "index_degraded")
    return [
        {
            "code": "index_backend_not_ready",
            "message": f"index backend is degraded ({reason})",
        }
    ]


def _deep_mode_readiness_issues(
    cfg: Settings,
    deep_provider: dict[str, Any],
) -> list[dict[str, str]]:
    if not cfg.deep_mode_enabled:
        return [
            {
                "code": "deep_mode_disabled",
                "message": "deep mode is disabled by configuration",
            }
        ]

    if bool(deep_provider.get("ready")):
        return []

    reason = str(deep_provider.get("reason") or "provider_unavailable")
    return [
        {
            "code": "deep_provider_not_ready",
            "message": f"deep provider is not ready ({reason})",
        }
    ]


def _optional_capability_issues(
    capabilities: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for name, report in capabilities.items():
        if not bool(report.get("enabled")) or bool(report.get("ready")):
            continue
        reason = str(report.get("reason") or "not_ready")
        hint = str(report.get("hint") or "").strip()
        issue = {
            "capability": name,
            "reason": reason,
            "message": f"{name} not ready ({reason})",
        }
        if hint:
            issue["hint"] = hint
        issues.append(issue)
    return issues


def _deep_provider_diagnostics(cfg: Settings) -> dict[str, Any]:
    if not cfg.deep_mode_enabled:
        return {
            "provider": "disabled",
            "ready": False,
            "reason": "provider_disabled",
        }

    api_key_present = bool((cfg.cloud_agent_api_key or "").strip())
    if api_key_present:
        return {
            "provider": "gemini",
            "ready": True,
            "reason": "api_key_present",
            "model": cfg.cloud_agent_model,
            "timeout_seconds": cfg.cloud_agent_timeout_seconds,
            "retry_attempts": cfg.cloud_agent_retry_attempts,
        }

    return {
        "provider": "local",
        "ready": True,
        "reason": "local_fallback",
        "model": cfg.local_deep_model or cfg.local_llm_model,
        "base_url": cfg.ollama_base_url,
        "timeout_seconds": cfg.cloud_agent_timeout_seconds,
    }


def _normalized_header_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_chat_model_routing(
    *,
    cfg: Settings,
    model_backend_header: str | None,
    api_key_header: str | None,
    api_model_header: str | None,
) -> _ChatModelRouting:
    backend = (_normalized_header_value(model_backend_header) or "auto").lower()
    if backend not in {"auto", "api", "local"}:
        raise ApiError(
            code="invalid_model_backend",
            message="x-model-backend must be one of: auto, api, local",
            status_code=422,
            details={"x-model-backend": backend},
        )

    api_key = _normalized_header_value(api_key_header) or _normalized_header_value(
        cfg.cloud_agent_api_key
    )
    api_model = _normalized_header_value(api_model_header) or cfg.cloud_agent_model

    if backend == "api" and not api_key:
        raise ApiError(
            code="missing_api_key",
            message="x-api-key or CLOUD_AGENT_API_KEY is required for api backend",
            status_code=400,
        )

    use_api_model = backend == "api" or (backend == "auto" and bool(api_key))
    return _ChatModelRouting(
        backend=backend,
        use_api_model=use_api_model,
        api_key=api_key,
        api_model=api_model,
    )


def _cfg_with_api_overrides(
    *,
    cfg: Settings,
    api_key: str,
    api_model: str,
) -> Settings:
    return cfg.model_copy(
        update={
            "cloud_agent_api_key": api_key,
            "cloud_agent_model": api_model,
        }
    )


def _local_chat_routing(cfg: Settings) -> _ChatModelRouting:
    return _ChatModelRouting(
        backend="local",
        use_api_model=False,
        api_key=None,
        api_model=cfg.local_deep_model or cfg.local_llm_model,
    )


def _termination_reason(response: ChatResponse) -> str:
    if response.trace is None:
        return ""
    return str(response.trace.termination_reason or "").strip()


def _should_auto_fallback_to_local(
    *,
    routing: _ChatModelRouting,
    response: ChatResponse,
    mode: Mode,
) -> bool:
    if routing.backend != "auto" or not routing.use_api_model:
        return False

    reason = _termination_reason(response)
    if not reason:
        return False

    if mode == Mode.DEEP:
        return reason in _DEEP_PROVIDER_FAILURE_CODES
    return reason == "generation_error"


def _resolve_fast_engine_for_request(
    *,
    services: ApiServices,
    routing: _ChatModelRouting,
) -> Any:
    if not routing.use_api_model:
        return services.local_qa
    if not isinstance(services.local_qa, LocalQAEngine):
        return services.local_qa
    if not routing.api_key:
        return services.local_qa

    base_engine = services.local_qa
    cfg_override = _cfg_with_api_overrides(
        cfg=services.cfg,
        api_key=routing.api_key,
        api_model=routing.api_model,
    )
    generation_client = GeminiTextGenerationClient(
        base_url=cfg_override.cloud_agent_api_base_url,
        api_key=routing.api_key,
        model=routing.api_model,
    )
    return LocalQAEngine(
        index_store=base_engine.index_store,
        cfg=cfg_override,
        embedder=base_engine.embedder,
        ollama_client=generation_client,
        generation_model=routing.api_model,
        prompt_version=base_engine.prompt_version,
        retrieval_top_k=base_engine.retrieval_top_k,
        min_token_overlap=base_engine.min_token_overlap,
        query_vector_dimension=base_engine.query_vector_dimension,
    )


def _resolve_deep_engine_for_request(
    *,
    services: ApiServices,
    routing: _ChatModelRouting,
) -> Any:
    if not isinstance(services.cloud_agent, CloudAgentEngine):
        return services.cloud_agent

    # Preserve test mocks by checking if the base client is one of our standard providers
    base_client = services.cloud_agent.model_client
    is_standard_client = isinstance(
        base_client, (GeminiCloudModelClient, LocalDeepModelClient)
    )
    if not is_standard_client:
        return services.cloud_agent

    tool_provider = services.cloud_agent.tool_provider
    max_iterations = services.cloud_agent.max_iterations
    prompt_version = services.cloud_agent.prompt_version

    if routing.use_api_model and routing.api_key:
        cfg_override = _cfg_with_api_overrides(
            cfg=services.cfg,
            api_key=routing.api_key,
            api_model=routing.api_model,
        )
        return CloudAgentEngine(
            model_client=GeminiCloudModelClient(cfg=cfg_override),
            tool_provider=tool_provider,
            cfg=cfg_override,
            max_iterations=max_iterations,
            model_name=routing.api_model,
            prompt_version=prompt_version,
        )

    local_model = services.cfg.local_deep_model or services.cfg.local_llm_model
    return CloudAgentEngine(
        model_client=LocalDeepModelClient(cfg=services.cfg, model_name=local_model),
        tool_provider=tool_provider,
        cfg=services.cfg,
        max_iterations=max_iterations,
        model_name=local_model,
        prompt_version=prompt_version,
    )


async def _read_session_history(
    *,
    services: ApiServices,
    session_id: str | None,
    document_id: str | None,
) -> list[dict[str, str]]:
    key = _session_history_key(session_id=session_id, document_id=document_id)
    if key is None:
        return []
    return await services.state_store.get_session_history(key)


async def _record_session_turn(
    *,
    services: ApiServices,
    session_id: str | None,
    document_id: str | None,
    question: str,
    answer: str,
) -> None:
    key = _session_history_key(session_id=session_id, document_id=document_id)
    clean_question = question.strip()
    clean_answer = answer.strip()
    if key is None or not clean_question or not clean_answer:
        return
    await services.state_store.append_session_turn(
        key=key,
        question=clean_question,
        answer=clean_answer,
        max_turns=services.cfg.api_state_session_max_turns,
    )


def _session_history_key(
    *, session_id: str | None, document_id: str | None
) -> str | None:
    if session_id is None:
        return None

    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        return None

    normalized_document_id = document_id.strip() if document_id else "*"
    return f"{normalized_session_id}::{normalized_document_id}"
