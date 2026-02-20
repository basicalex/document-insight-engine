from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import ApiServices, app
from src.api.state_store import InMemoryApiStateBackend, InMemoryApiStateStore
from src.config.settings import Settings
from src.engine.extractor import (
    FieldProvenance,
    StructuredExtractionEnvelope,
    StructuredExtractionError,
    StructuredExtractionResult,
    ValidationDiagnostic,
)
from src.ingestion import (
    HybridVectorIndexStore,
    InMemoryIndexBackend,
    UploadIntakeService,
)
from src.ingestion.orchestration import IngestionRecord
from src.models.schemas import (
    AgentTrace,
    ChatResponse,
    IngestResponse,
    IngestionStatus,
    Mode,
)


class StubLocalQAEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mode, str | None]] = []
        self.chat_history_calls: list[list[dict[str, str]] | None] = []

    def ask(
        self,
        question: str,
        mode: Mode,
        document_id: str | None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        self.chat_history_calls.append(chat_history)
        return ChatResponse(
            answer="local-answer",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=False,
            citations=[],
        )

    def ask_stream_events(
        self,
        *,
        question: str,
        mode: Mode,
        document_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> Any:
        response = self.ask(
            question=question,
            mode=mode,
            document_id=document_id,
            chat_history=chat_history,
        )
        yield {
            "type": "status",
            "phase": "generation",
            "message": "Generating answer...",
        }
        yield {"type": "token", "delta": response.answer}
        yield {"type": "final", "response": response.model_dump(mode="json")}


class SlowLocalQAEngine(StubLocalQAEngine):
    def ask(
        self,
        question: str,
        mode: Mode,
        document_id: str | None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        time.sleep(1.2)
        return super().ask(
            question=question,
            mode=mode,
            document_id=document_id,
            chat_history=chat_history,
        )


class TraceLocalQAEngine(StubLocalQAEngine):
    def ask(
        self,
        question: str,
        mode: Mode,
        document_id: str | None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        self.chat_history_calls.append(chat_history)
        return ChatResponse(
            answer="local-answer",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=False,
            citations=[],
            trace=AgentTrace(model="local-qa-test", termination_reason="completed"),
        )


class StubCloudAgentEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mode, str | None]] = []

    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        return ChatResponse(
            answer="cloud-answer",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=False,
            citations=[],
        )


class VerboseCloudAgentEngine(StubCloudAgentEngine):
    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        return ChatResponse(
            answer="deep stream response",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=False,
            citations=[],
        )


class FailingCloudAgentEngine(StubCloudAgentEngine):
    def __init__(self, termination_reason: str) -> None:
        super().__init__()
        self.termination_reason = termination_reason

    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        return ChatResponse(
            answer="provider failed",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=True,
            citations=[],
            trace=AgentTrace(
                model="gemini-2.5-flash",
                iterations=1,
                termination_reason=self.termination_reason,
            ),
        )


class ParsedArtifactMissingCloudAgentEngine(StubCloudAgentEngine):
    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        return ChatResponse(
            answer="Re-ingest required",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=True,
            citations=[],
            trace=AgentTrace(
                model="gemini-2.5-flash",
                iterations=0,
                termination_reason="parsed_artifact_missing",
            ),
        )


class StubStructuredExtractor:
    def __init__(self, envelope: StructuredExtractionEnvelope) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, str, dict[str, Any], str | None]] = []

    def extract_structured(
        self,
        *,
        document_id: str,
        document_text: str,
        schema: dict[str, Any],
        prompt: str | None = None,
    ) -> StructuredExtractionEnvelope:
        self.calls.append((document_id, document_text, schema, prompt))
        return self.envelope


class StubOrchestrator:
    def __init__(self, status: IngestionStatus = IngestionStatus.INDEXED) -> None:
        self.calls: list[tuple[str, Path, str, str | None]] = []
        self.status = status

    def process(
        self,
        document_id: str,
        file_path: Path,
        mime_type: str,
        idempotency_key: str | None = None,
    ) -> IngestionRecord:
        self.calls.append((document_id, file_path, mime_type, idempotency_key))
        record = IngestionRecord(
            document_id=document_id,
            idempotency_key=idempotency_key or f"ingest-{document_id}",
            status=self.status,
        )
        if self.status == IngestionStatus.INDEXED:
            record.completed_stages = ["extract", "parse", "chunk", "embed", "index"]
        if self.status == IngestionStatus.PARTIAL:
            record.completed_stages = ["extract", "parse"]
            record.error_message = "parse degraded"
        record.updated_at = datetime.now(timezone.utc)
        return record

    def get_record(self, document_id: str) -> IngestionRecord | None:
        return None


class RetryThenSuccessOrchestrator:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = max(0, failures_before_success)
        self.calls: list[tuple[str, Path, str, str | None]] = []

    def process(
        self,
        document_id: str,
        file_path: Path,
        mime_type: str,
        idempotency_key: str | None = None,
    ) -> IngestionRecord:
        self.calls.append((document_id, file_path, mime_type, idempotency_key))
        if len(self.calls) <= self.failures_before_success:
            raise RuntimeError("transient queue failure")
        record = IngestionRecord(
            document_id=document_id,
            idempotency_key=idempotency_key or f"ingest-{document_id}",
            status=IngestionStatus.INDEXED,
            completed_stages=["extract", "parse", "chunk", "embed", "index"],
        )
        record.updated_at = datetime.now(timezone.utc)
        return record

    def get_record(self, document_id: str) -> IngestionRecord | None:
        return None


class AlwaysFailOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, str, str | None]] = []

    def process(
        self,
        document_id: str,
        file_path: Path,
        mime_type: str,
        idempotency_key: str | None = None,
    ) -> IngestionRecord:
        self.calls.append((document_id, file_path, mime_type, idempotency_key))
        raise RuntimeError("poison ingestion payload")

    def get_record(self, document_id: str) -> IngestionRecord | None:
        return None


class FlakyClaimStateStore:
    def __init__(self, delegate: InMemoryApiStateStore) -> None:
        self._delegate = delegate
        self._failed = False

    async def get_idempotency_response(self, key: str) -> IngestResponse | None:
        return await self._delegate.get_idempotency_response(key)

    async def claim_idempotency_key(self, key: str) -> bool:
        if key.startswith("queue:doc:") and not self._failed:
            self._failed = True
            raise RuntimeError("transient claim failure")
        return await self._delegate.claim_idempotency_key(key)

    async def put_idempotency_response(
        self, key: str, response: IngestResponse
    ) -> None:
        await self._delegate.put_idempotency_response(key, response)

    async def release_idempotency_key(self, key: str) -> None:
        await self._delegate.release_idempotency_key(key)

    async def get_ingestion_record(self, document_id: str) -> IngestResponse | None:
        return await self._delegate.get_ingestion_record(document_id)

    async def put_ingestion_record(self, response: IngestResponse) -> None:
        await self._delegate.put_ingestion_record(response)

    async def get_session_history(self, key: str) -> list[dict[str, str]]:
        return await self._delegate.get_session_history(key)

    async def append_session_turn(
        self,
        *,
        key: str,
        question: str,
        answer: str,
        max_turns: int,
    ) -> None:
        await self._delegate.append_session_turn(
            key=key,
            question=question,
            answer=answer,
            max_turns=max_turns,
        )

    async def close(self) -> None:
        await self._delegate.close()


def _make_services(
    *,
    tmp_path: Path,
    local_qa: StubLocalQAEngine | None = None,
    cloud_agent: StubCloudAgentEngine | None = None,
    orchestrator: Any | None = None,
    state_store: InMemoryApiStateStore | None = None,
    structured_extractor: Any | None = None,
    request_timeout_seconds: int = 10,
    deep_mode_enabled: bool = True,
    docling_enabled: bool = True,
    google_parser_enabled: bool = True,
    langextract_enabled: bool = True,
    parser_routing_mode: Literal[
        "docling_google_fallback",
        "google_docling_fallback",
        "docling_fallback",
        "google_fallback",
        "fallback_only",
    ] = "docling_google_fallback",
    ingestion_queue_max_retries: int = 2,
    ingestion_queue_retry_backoff_seconds: float = 0.0,
    ingestion_worker_concurrency: int = 1,
) -> ApiServices:
    cfg = Settings(
        data_dir=tmp_path,
        max_upload_size_mb=1,
        request_timeout_seconds=request_timeout_seconds,
        ingest_timeout_seconds=max(request_timeout_seconds, 1),
        ingestion_queue_backend="memory",
        ingestion_queue_max_retries=ingestion_queue_max_retries,
        ingestion_queue_retry_backoff_seconds=ingestion_queue_retry_backoff_seconds,
        ingestion_queue_poll_timeout_seconds=0.05,
        ingestion_worker_concurrency=ingestion_worker_concurrency,
        deep_mode_enabled=deep_mode_enabled,
        cloud_agent_provider="fallback" if deep_mode_enabled else "disabled",
        cloud_agent_api_key="",
        docling_enabled=docling_enabled,
        google_parser_enabled=google_parser_enabled,
        langextract_enabled=langextract_enabled,
        parser_routing_mode=parser_routing_mode,
    )
    backend = InMemoryIndexBackend()
    index_store = HybridVectorIndexStore(
        cfg=cfg,
        backend=backend,
        tier1_dimension=4,
        tier4_dimension=6,
    )
    index_store.bootstrap_indices()

    local_engine = cast(Any, local_qa or StubLocalQAEngine())
    cloud_engine = cast(Any, cloud_agent or StubCloudAgentEngine())

    return ApiServices(
        cfg=cfg,
        intake=UploadIntakeService(cfg),
        index_store=index_store,
        index_readiness={
            "state": "ready",
            "backend": type(index_store.backend).__name__,
            "reason": "test_backend",
            "fallback_allowed": False,
            "degraded": False,
        },
        local_qa=local_engine,
        cloud_agent=cloud_engine,
        state_store=state_store
        or InMemoryApiStateStore(
            idempotency_ttl_seconds=cfg.api_state_idempotency_ttl_seconds,
            ingestion_ttl_seconds=cfg.api_state_ingestion_ttl_seconds,
            session_ttl_seconds=cfg.api_state_session_ttl_seconds,
            idempotency_claim_ttl_seconds=cfg.api_state_idempotency_claim_ttl_seconds,
        ),
        structured_extractor=structured_extractor,
        orchestrator=orchestrator,
    )


def _shared_state_store(
    *,
    cfg: Settings,
    backend: InMemoryApiStateBackend,
) -> InMemoryApiStateStore:
    return InMemoryApiStateStore(
        idempotency_ttl_seconds=cfg.api_state_idempotency_ttl_seconds,
        ingestion_ttl_seconds=cfg.api_state_ingestion_ttl_seconds,
        session_ttl_seconds=cfg.api_state_session_ttl_seconds,
        idempotency_claim_ttl_seconds=cfg.api_state_idempotency_claim_ttl_seconds,
        backend=backend,
    )


@contextmanager
def _client_with_services(services: ApiServices) -> Iterator[TestClient]:
    app.state.services = services
    try:
        with TestClient(app) as client:
            yield client
    finally:
        if hasattr(app.state, "services"):
            delattr(app.state, "services")


def _success_extraction_envelope(document_id: str) -> StructuredExtractionEnvelope:
    return StructuredExtractionEnvelope(
        ok=True,
        result=StructuredExtractionResult(
            document_id=document_id,
            model="langextract",
            prompt_version="tier4-extraction-v1",
            data={"total_due": 1234.0},
            provenance={
                "total_due": FieldProvenance(
                    start_offset=13,
                    end_offset=20,
                    text="1234.00",
                )
            },
            accepted_fields=["total_due"],
            rejected_fields=[],
            diagnostics=[],
            token_usage={"input_estimate": 20, "output_estimate": 10, "total": 30},
            latency_ms=12,
        ),
        error=None,
    )


def _error_extraction_envelope(
    *,
    code: str,
    message: str,
    diagnostics: list[ValidationDiagnostic] | None = None,
) -> StructuredExtractionEnvelope:
    return StructuredExtractionEnvelope(
        ok=False,
        result=None,
        error=StructuredExtractionError(
            code=code,
            message=message,
            diagnostics=diagnostics or [],
            token_usage={"input_estimate": 10, "output_estimate": 0, "total": 10},
            latency_ms=5,
        ),
    )


def _seed_ingestion_record(
    *,
    services: ApiServices,
    document_id: str,
    file_path: Path,
    status: IngestionStatus = IngestionStatus.INDEXED,
) -> None:
    asyncio.run(
        services.state_store.put_ingestion_record(
            IngestResponse(
                document_id=document_id,
                file_path=str(file_path),
                status=status,
                message="seeded",
            )
        )
    )


def _wait_for_ingest_status(
    client: TestClient,
    *,
    document_id: str,
    expected: IngestionStatus,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        status_response = client.get(f"/ingest/{document_id}")
        assert status_response.status_code == 200
        payload = cast(dict[str, Any], status_response.json())
        latest_payload = payload
        if payload.get("status") == expected.value:
            return payload
        time.sleep(0.02)

    pytest.fail(
        f"document {document_id} did not reach status {expected.value}; "
        f"last payload={latest_payload}"
    )


def test_ingest_returns_contract_and_correlation_header(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\ninvoice",
                    "application/pdf",
                )
            },
            headers={"x-correlation-id": "corr-ingest-1"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "uploaded"
    assert body["message"] == "queued for processing"
    assert body["document_id"]
    assert body["file_path"].endswith("_invoice.pdf")
    assert Path(body["file_path"]).exists()
    assert response.headers["x-correlation-id"] == "corr-ingest-1"


def test_healthz_includes_deep_provider_diagnostics(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path, deep_mode_enabled=False)

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["deep_provider"]["provider"] == "disabled"
    assert body["deep_provider"]["ready"] is False
    assert body["deep_provider"]["reason"] == "provider_disabled"
    assert body["readiness"]["overall"] == "ready"
    assert body["readiness"]["index"]["state"] == "ready"
    assert body["parser_routing_mode"] == "docling_google_fallback"


def test_healthz_reports_local_deep_provider_as_ready(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)
    services.cfg = Settings(
        data_dir=tmp_path,
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="",
        local_deep_model="qwen2.5:7b-instruct",
        ingestion_queue_backend="memory",
    )

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["deep_provider"]["provider"] == "local"
    assert body["deep_provider"]["ready"] is True
    assert body["deep_provider"]["reason"] == "local_fallback"
    assert body["deep_provider"]["model"] == "qwen2.5:7b-instruct"


def test_healthz_prefers_gemini_when_api_key_is_configured(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)
    services.cfg = Settings(
        data_dir=tmp_path,
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="test-key",
        cloud_agent_model="gemini-3-flash",
        ingestion_queue_backend="memory",
    )

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["deep_provider"]["provider"] == "gemini"
    assert body["deep_provider"]["ready"] is True
    assert body["deep_provider"]["reason"] == "api_key_present"
    assert body["deep_provider"]["model"] == "gemini-3-flash"


def test_resolve_chat_model_routing_prefers_api_in_auto_when_key_available() -> None:
    cfg = Settings(
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="env-key",
    )

    routing = api_main._resolve_chat_model_routing(
        cfg=cfg,
        model_backend_header="auto",
        api_key_header=None,
        api_model_header="gemini-3-flash",
    )

    assert routing.use_api_model is True
    assert routing.api_key == "env-key"
    assert routing.api_model == "gemini-3-flash"


def test_resolve_chat_model_routing_falls_back_to_local_without_key() -> None:
    cfg = Settings(
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="",
    )

    routing = api_main._resolve_chat_model_routing(
        cfg=cfg,
        model_backend_header="auto",
        api_key_header=None,
        api_model_header=None,
    )

    assert routing.use_api_model is False
    assert routing.backend == "auto"


def test_resolve_chat_model_routing_requires_key_for_api_backend() -> None:
    cfg = Settings(
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="",
    )

    with pytest.raises(api_main.ApiError) as exc:
        api_main._resolve_chat_model_routing(
            cfg=cfg,
            model_backend_header="api",
            api_key_header=None,
            api_model_header=None,
        )

    assert exc.value.code == "missing_api_key"


def test_should_auto_fallback_to_local_detects_deep_provider_failure() -> None:
    routing = api_main._ChatModelRouting(
        backend="auto",
        use_api_model=True,
        api_key="key-1",
        api_model="gemini-3-flash",
    )
    response = ChatResponse(
        answer="provider failed",
        mode=Mode.DEEP,
        document_id="doc-1",
        insufficient_evidence=True,
        citations=[],
        trace=AgentTrace(
            model="gemini-3-flash",
            iterations=1,
            termination_reason="provider_unavailable",
        ),
    )

    should_fallback = api_main._should_auto_fallback_to_local(
        routing=routing,
        response=response,
        mode=Mode.DEEP,
    )

    assert should_fallback is True


def test_should_auto_fallback_to_local_ignores_non_auto_backend() -> None:
    routing = api_main._ChatModelRouting(
        backend="api",
        use_api_model=True,
        api_key="key-1",
        api_model="gemini-3-flash",
    )
    response = ChatResponse(
        answer="provider failed",
        mode=Mode.DEEP,
        document_id="doc-1",
        insufficient_evidence=True,
        citations=[],
        trace=AgentTrace(
            model="gemini-3-flash",
            iterations=1,
            termination_reason="provider_unavailable",
        ),
    )

    should_fallback = api_main._should_auto_fallback_to_local(
        routing=routing,
        response=response,
        mode=Mode.DEEP,
    )

    assert should_fallback is False


def test_ask_deep_auto_falls_back_to_local_on_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = _make_services(tmp_path=tmp_path)
    services.cfg = Settings(
        data_dir=tmp_path,
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="api-key",
        ingestion_queue_backend="memory",
    )
    failing_engine = FailingCloudAgentEngine("provider_unavailable")
    fallback_engine = StubCloudAgentEngine()
    calls: list[tuple[str, bool]] = []

    def fake_resolve_deep_engine(*, services: Any, routing: Any) -> Any:
        del services
        calls.append((routing.backend, routing.use_api_model))
        return failing_engine if routing.use_api_model else fallback_engine

    monkeypatch.setattr(
        api_main,
        "_resolve_deep_engine_for_request",
        fake_resolve_deep_engine,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Give me a deep answer",
                "mode": "deep",
                "document_id": "doc-1",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "cloud-answer"
    assert calls == [("auto", True), ("local", False)]


def test_ask_stream_deep_auto_falls_back_to_local_on_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = _make_services(tmp_path=tmp_path)
    services.cfg = Settings(
        data_dir=tmp_path,
        deep_mode_enabled=True,
        cloud_agent_provider="local",
        cloud_agent_api_key="api-key",
        ingestion_queue_backend="memory",
    )
    failing_engine = FailingCloudAgentEngine("provider_unavailable")
    fallback_engine = StubCloudAgentEngine()

    def fake_resolve_deep_engine(*, services: Any, routing: Any) -> Any:
        del services
        return failing_engine if routing.use_api_model else fallback_engine

    monkeypatch.setattr(
        api_main,
        "_resolve_deep_engine_for_request",
        fake_resolve_deep_engine,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ask/stream",
            json={
                "question": "Give me a deep answer",
                "mode": "deep",
                "document_id": "doc-1",
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip()]
    final_event = json.loads(lines[-1])
    assert final_event["type"] == "final"
    assert final_event["response"]["answer"] == "cloud-answer"


def test_healthz_reports_optional_capability_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = _make_services(tmp_path=tmp_path)

    def fake_find_spec(name: str) -> object | None:
        if name in {"docling", "langextract"}:
            return None
        return object()

    monkeypatch.setattr(api_main.importlib.util, "find_spec", fake_find_spec)

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["docling_parser"]["ready"] is False
    assert capabilities["docling_parser"]["reason"] == "missing_dependency"
    assert "pip install -e .[ai-docling]" in capabilities["docling_parser"]["hint"]
    assert capabilities["google_parser"]["ready"] is False
    assert capabilities["google_parser"]["reason"] == "missing_api_key"
    assert "GOOGLE_API_KEY" in capabilities["google_parser"]["hint"]
    assert capabilities["langextract_extractor"]["ready"] is False
    assert capabilities["langextract_extractor"]["reason"] == "missing_dependency"


def test_healthz_marks_optional_capability_disabled_by_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = _make_services(
        tmp_path=tmp_path,
        docling_enabled=False,
        langextract_enabled=False,
    )

    monkeypatch.setattr(api_main.importlib.util, "find_spec", lambda _: None)

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["docling_parser"]["enabled"] is False
    assert capabilities["docling_parser"]["reason"] == "disabled_by_config"
    assert capabilities["google_parser"]["enabled"] is True
    assert capabilities["langextract_extractor"]["enabled"] is False
    assert capabilities["langextract_extractor"]["reason"] == "disabled_by_config"


def test_runtime_capability_logging_warns_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    cfg = Settings(data_dir=tmp_path)

    def fake_find_spec(name: str) -> object | None:
        if name == "docling":
            return None
        return object()

    monkeypatch.setattr(api_main.importlib.util, "find_spec", fake_find_spec)

    with caplog.at_level(logging.INFO):
        api_main._log_runtime_capabilities(cfg)

    assert (
        "Runtime capability docling_parser => enabled=True ready=False" in caplog.text
    )
    assert "pip install -e .[ai-docling]" in caplog.text


def test_healthz_marks_google_parser_disabled_by_config(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path, google_parser_enabled=False)

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["google_parser"]["enabled"] is False
    assert capabilities["google_parser"]["reason"] == "disabled_by_config"


def test_healthz_marks_parser_steps_excluded_by_routing_mode(tmp_path: Path) -> None:
    services = _make_services(
        tmp_path=tmp_path,
        parser_routing_mode="fallback_only",
    )

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["docling_parser"]["enabled"] is False
    assert capabilities["docling_parser"]["reason"] == "excluded_by_routing_mode"
    assert capabilities["google_parser"]["enabled"] is False
    assert capabilities["google_parser"]["reason"] == "excluded_by_routing_mode"


def test_metrics_endpoint_exposes_http_and_qa_counters(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        ask_response = client.post(
            "/ask",
            json={"question": "What is due?", "mode": "fast", "document_id": "doc-1"},
        )
        assert ask_response.status_code == 200

        metrics = client.get("/metrics")

    assert metrics.status_code == 200
    body = metrics.text
    assert "die_http_requests_total" in body
    assert "die_qa_requests_total" in body
    assert "die_qa_insufficient_evidence_rate" in body


def test_healthz_includes_observability_trace_linkage(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path, local_qa=TraceLocalQAEngine())

    with _client_with_services(services) as client:
        ask_response = client.post(
            "/ask",
            json={"question": "What is due?", "mode": "fast", "document_id": "doc-1"},
            headers={"x-correlation-id": "corr-obsv-1"},
        )
        assert ask_response.status_code == 200
        assert ask_response.headers.get("x-trace-id")

        health = client.get("/healthz")

    assert health.status_code == 200
    observability = health.json()["observability"]
    assert observability["qa"]["requests_total"] >= 1
    assert observability["trace_links_recent"]
    assert observability["trace_links_recent"][-1]["correlation_id"] == "corr-obsv-1"


def test_parser_order_mode_mapping() -> None:
    assert api_main._parser_order_for_mode("docling_google_fallback") == (
        "docling",
        "google",
        "fallback",
    )
    assert api_main._parser_order_for_mode("google_fallback") == ("google", "fallback")


def test_readyz_returns_503_when_index_backend_is_degraded(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)
    services.index_readiness = {
        "state": "degraded",
        "backend": "InMemoryIndexBackend",
        "reason": "redis_unavailable_fallback_enabled",
        "fallback_allowed": True,
        "degraded": True,
    }

    with _client_with_services(services) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["overall"] == "degraded"
    assert body["index"]["backend"] == "InMemoryIndexBackend"


def test_healthz_readiness_actions_surface_deep_mode_blockers(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path, deep_mode_enabled=False)

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    readiness = response.json()["readiness"]
    assert readiness["actions"]["ask_fast"]["ready"] is True
    assert readiness["actions"]["ask_deep"]["ready"] is False
    deep_issues = readiness["actions"]["ask_deep"]["blocking_issues"]
    assert deep_issues[0]["code"] == "deep_mode_disabled"


@pytest.mark.parametrize(
    ("profile_name", "profile_kwargs"),
    [
        (
            "lite",
            {
                "docling_enabled": False,
                "google_parser_enabled": False,
                "langextract_enabled": False,
                "parser_routing_mode": "fallback_only",
            },
        ),
        (
            "full",
            {
                "docling_enabled": True,
                "google_parser_enabled": False,
                "langextract_enabled": True,
                "parser_routing_mode": "docling_google_fallback",
            },
        ),
    ],
)
def test_profile_smoke_upload_ingest_and_ask_fast_deep(
    tmp_path: Path,
    profile_name: str,
    profile_kwargs: dict[str, Any],
) -> None:
    del profile_name
    services = _make_services(
        tmp_path=tmp_path,
        orchestrator=StubOrchestrator(status=IngestionStatus.INDEXED),
        **profile_kwargs,
    )

    with _client_with_services(services) as client:
        upload_response = client.post(
            "/upload",
            files={"file": ("invoice.pdf", b"%PDF-1.4\ninvoice", "application/pdf")},
        )

        assert upload_response.status_code == 201
        document_id = upload_response.json()["document_id"]

        _wait_for_ingest_status(
            client,
            document_id=document_id,
            expected=IngestionStatus.INDEXED,
        )

        fast_response = client.post(
            "/ask",
            json={
                "question": "What is due?",
                "mode": "fast",
                "document_id": document_id,
            },
        )
        deep_response = client.post(
            "/ask",
            json={
                "question": "What is due?",
                "mode": "deep",
                "document_id": document_id,
            },
        )

    assert fast_response.status_code == 200
    assert deep_response.status_code == 200


def test_build_default_services_fails_fast_in_prod_when_redis_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingHybridVectorIndexStore:
        def __init__(self, cfg: Settings, backend: Any | None = None) -> None:
            self.cfg = cfg
            self.backend = backend
            self._using_fallback = backend is not None

        def bootstrap_indices(self) -> None:
            if self._using_fallback:
                raise AssertionError(
                    "in-memory fallback should not be attempted in prod"
                )
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(
        api_main,
        "HybridVectorIndexStore",
        FailingHybridVectorIndexStore,
    )
    cfg = Settings(
        data_dir=tmp_path,
        environment="prod",
        deep_mode_enabled=False,
        cloud_agent_provider="disabled",
    )

    with pytest.raises(RuntimeError, match="in-memory fallback is disabled"):
        api_main._build_default_services(cfg)


def test_build_default_services_allows_explicit_dev_in_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingRedisThenMemoryIndexStore:
        def __init__(self, cfg: Settings, backend: Any | None = None) -> None:
            self.cfg = cfg
            self.backend = backend if backend is not None else object()
            self._using_fallback = backend is not None

        def bootstrap_indices(self) -> None:
            if self._using_fallback:
                return
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(
        api_main,
        "HybridVectorIndexStore",
        FailingRedisThenMemoryIndexStore,
    )
    cfg = Settings(
        data_dir=tmp_path,
        environment="dev",
        allow_in_memory_index_fallback=True,
        deep_mode_enabled=False,
        cloud_agent_provider="disabled",
    )

    services = api_main._build_default_services(cfg)

    assert services.index_readiness["state"] == "degraded"
    assert services.index_readiness["reason"] == "redis_unavailable_fallback_enabled"
    assert services.index_readiness["fallback_allowed"] is True


def test_ingest_runs_pipeline_when_orchestrator_is_configured(tmp_path: Path) -> None:
    orchestrator = StubOrchestrator(status=IngestionStatus.INDEXED)
    services = _make_services(tmp_path=tmp_path, orchestrator=orchestrator)

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\ninvoice",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "uploaded"
        assert body["message"] == "queued for processing"
        final = _wait_for_ingest_status(
            client,
            document_id=body["document_id"],
            expected=IngestionStatus.INDEXED,
        )

    assert final["message"] == "indexed and ready for retrieval"
    assert len(orchestrator.calls) == 1


def test_ingest_idempotency_key_replays_previous_response(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        first = client.post(
            "/ingest",
            files={"file": ("invoice.pdf", b"%PDF-1.4\nfirst", "application/pdf")},
            headers={"Idempotency-Key": "idem-123"},
        )
        second = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\nfirst-but-retried",
                    "application/pdf",
                )
            },
            headers={"Idempotency-Key": "idem-123"},
        )

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["file_path"] == second.json()["file_path"]


def test_ingest_returns_conflict_when_idempotency_key_is_in_progress(
    tmp_path: Path,
) -> None:
    services = _make_services(tmp_path=tmp_path, request_timeout_seconds=1)
    claimed = asyncio.run(services.state_store.claim_idempotency_key("idem-busy"))
    assert claimed is True

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={"file": ("invoice.pdf", b"%PDF-1.4\nfirst", "application/pdf")},
            headers={"Idempotency-Key": "idem-busy"},
        )

    assert response.status_code == 409
    assert response.json()["code"] == "idempotency_conflict"


def test_get_ingest_status_returns_record_for_known_document(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        created = client.post(
            "/ingest",
            files={"file": ("invoice.pdf", b"%PDF-1.4\nfirst", "application/pdf")},
        )
        assert created.status_code == 201
        document_id = created.json()["document_id"]

        status_response = client.get(f"/ingest/{document_id}")

    assert status_response.status_code == 200
    assert status_response.json()["document_id"] == document_id


def test_ask_returns_document_not_ready_when_status_is_not_indexed(
    tmp_path: Path,
) -> None:
    services = _make_services(
        tmp_path=tmp_path,
        orchestrator=StubOrchestrator(status=IngestionStatus.PARTIAL),
    )
    asyncio.run(
        services.state_store.put_ingestion_record(
            IngestResponse(
                document_id="doc-pending",
                file_path="data/uploads/doc-pending.pdf",
                status=IngestionStatus.PARTIAL,
                message="partially indexed; some stages failed",
            )
        )
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What is due?",
                "mode": "fast",
                "document_id": "doc-pending",
            },
        )

    assert response.status_code == 409
    assert response.json()["code"] == "document_not_ready"


def test_upload_alias_accepts_single_file_and_matches_ingest_shape(
    tmp_path: Path,
) -> None:
    orchestrator = StubOrchestrator(status=IngestionStatus.INDEXED)
    services = _make_services(tmp_path=tmp_path, orchestrator=orchestrator)

    with _client_with_services(services) as client:
        response = client.post(
            "/upload",
            files={"file": ("invoice.pdf", b"%PDF-1.4\ninvoice", "application/pdf")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["document_id"]
        assert body["status"] == "uploaded"
        assert body["message"] == "queued for processing"
        final = _wait_for_ingest_status(
            client,
            document_id=body["document_id"],
            expected=IngestionStatus.INDEXED,
        )

    assert final["status"] == "indexed"


def test_upload_alias_accepts_multiple_files(tmp_path: Path) -> None:
    orchestrator = StubOrchestrator(status=IngestionStatus.INDEXED)
    services = _make_services(tmp_path=tmp_path, orchestrator=orchestrator)

    with _client_with_services(services) as client:
        response = client.post(
            "/upload",
            files=[
                ("files", ("invoice.pdf", b"%PDF-1.4\ninvoice", "application/pdf")),
                ("files", ("contract.pdf", b"%PDF-1.4\ncontract", "application/pdf")),
            ],
        )

        assert response.status_code == 201
        body = response.json()
        assert body["count"] == 2
        assert len(body["documents"]) == 2
        assert all(item["status"] == "uploaded" for item in body["documents"])
        for item in body["documents"]:
            _wait_for_ingest_status(
                client,
                document_id=item["document_id"],
                expected=IngestionStatus.INDEXED,
            )


def test_ingest_queue_retries_transient_orchestration_failure(tmp_path: Path) -> None:
    orchestrator = RetryThenSuccessOrchestrator(failures_before_success=1)
    services = _make_services(
        tmp_path=tmp_path,
        orchestrator=orchestrator,
        ingestion_queue_max_retries=2,
        ingestion_queue_retry_backoff_seconds=0.0,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\ninvoice",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "uploaded"
        final = _wait_for_ingest_status(
            client,
            document_id=body["document_id"],
            expected=IngestionStatus.INDEXED,
        )

    assert final["status"] == "indexed"
    assert len(orchestrator.calls) == 2


def test_ingest_queue_dead_letters_poison_payload_after_retry_exhaustion(
    tmp_path: Path,
) -> None:
    orchestrator = AlwaysFailOrchestrator()
    services = _make_services(
        tmp_path=tmp_path,
        orchestrator=orchestrator,
        ingestion_queue_max_retries=1,
        ingestion_queue_retry_backoff_seconds=0.0,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\ninvoice",
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 201
        body = response.json()
        final = _wait_for_ingest_status(
            client,
            document_id=body["document_id"],
            expected=IngestionStatus.FAILED,
            timeout_seconds=3.0,
        )

        health = client.get("/healthz")
        assert health.status_code == 200
        queue_report = health.json()["ingestion_queue"]

    assert final["status"] == "failed"
    assert len(orchestrator.calls) == 2
    assert queue_report["dead_letter_jobs"] >= 1


def test_ingest_queue_recovers_from_worker_loop_crash_without_losing_job(
    tmp_path: Path,
) -> None:
    orchestrator = StubOrchestrator(status=IngestionStatus.INDEXED)
    base_store = InMemoryApiStateStore(
        idempotency_ttl_seconds=24 * 60 * 60,
        ingestion_ttl_seconds=30 * 24 * 60 * 60,
        session_ttl_seconds=7 * 24 * 60 * 60,
        idempotency_claim_ttl_seconds=5 * 60,
    )
    services = _make_services(
        tmp_path=tmp_path,
        orchestrator=orchestrator,
        state_store=cast(Any, FlakyClaimStateStore(base_store)),
        ingestion_queue_max_retries=2,
        ingestion_queue_retry_backoff_seconds=0.0,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\ninvoice",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 201
        body = response.json()
        final = _wait_for_ingest_status(
            client,
            document_id=body["document_id"],
            expected=IngestionStatus.INDEXED,
        )

    assert final["status"] == "indexed"
    assert len(orchestrator.calls) >= 1


def test_ingest_unsupported_type_returns_normalized_error(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            headers={"x-correlation-id": "corr-ingest-error"},
        )

    assert response.status_code == 415
    body = response.json()
    assert body["code"] == "unsupported_mime_type"
    assert body["message"].startswith("unsupported MIME type")
    assert body["correlation_id"] == "corr-ingest-error"


def test_extract_returns_structured_payload_and_persists_artifact(
    tmp_path: Path,
) -> None:
    extractor = StubStructuredExtractor(_success_extraction_envelope("doc-1"))
    services = _make_services(tmp_path=tmp_path, structured_extractor=extractor)
    file_path = tmp_path / "invoice.pdf"
    file_path.write_bytes(b"Invoice Total: 1234.00 USD")
    _seed_ingestion_record(services=services, document_id="doc-1", file_path=file_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/extract",
            json={
                "document_id": "doc-1",
                "schema": {
                    "type": "object",
                    "properties": {"total_due": {"type": "number"}},
                    "required": ["total_due"],
                },
                "prompt": "Extract invoice fields",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["data"]["total_due"] == 1234.0
    assert Path(body["artifact_path"]).exists()
    assert extractor.calls


def test_extract_returns_422_for_invalid_schema(tmp_path: Path) -> None:
    extractor = StubStructuredExtractor(
        _error_extraction_envelope(
            code="invalid_schema",
            message="schema is invalid",
        )
    )
    services = _make_services(tmp_path=tmp_path, structured_extractor=extractor)
    file_path = tmp_path / "invoice.pdf"
    file_path.write_bytes(b"Invoice Total: 1234.00 USD")
    _seed_ingestion_record(services=services, document_id="doc-2", file_path=file_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/extract",
            json={"document_id": "doc-2", "schema": {"type": "object"}},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_schema"


def test_extract_returns_503_for_provider_failure(tmp_path: Path) -> None:
    extractor = StubStructuredExtractor(
        _error_extraction_envelope(
            code="provider_error",
            message="LangExtract provider unavailable",
        )
    )
    services = _make_services(tmp_path=tmp_path, structured_extractor=extractor)
    file_path = tmp_path / "invoice.pdf"
    file_path.write_bytes(b"Invoice Total: 1234.00 USD")
    _seed_ingestion_record(services=services, document_id="doc-3", file_path=file_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/extract",
            json={
                "document_id": "doc-3",
                "schema": {
                    "type": "object",
                    "properties": {"total_due": {"type": "number"}},
                },
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "provider_error"


def test_extract_returns_validation_diagnostics_for_provenance_mismatch(
    tmp_path: Path,
) -> None:
    extractor = StubStructuredExtractor(
        _error_extraction_envelope(
            code="validation_failed",
            message="structured extraction failed validation",
            diagnostics=[
                ValidationDiagnostic(
                    code="provenance_text_mismatch",
                    message="provenance text does not match offsets",
                    field="total_due",
                )
            ],
        )
    )
    services = _make_services(tmp_path=tmp_path, structured_extractor=extractor)
    file_path = tmp_path / "invoice.pdf"
    file_path.write_bytes(b"Invoice Total: 1234.00 USD")
    _seed_ingestion_record(services=services, document_id="doc-4", file_path=file_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/extract",
            json={
                "document_id": "doc-4",
                "schema": {
                    "type": "object",
                    "properties": {"total_due": {"type": "number"}},
                },
            },
        )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"
    diagnostics = body["details"]["diagnostics"]
    assert diagnostics[0]["code"] == "provenance_text_mismatch"


def test_ask_fast_mode_routes_to_local_engine(tmp_path: Path) -> None:
    local = StubLocalQAEngine()
    cloud = StubCloudAgentEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={"question": "What is due?", "mode": "fast", "document_id": "doc-1"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "local-answer"
    assert len(local.calls) == 1
    assert len(cloud.calls) == 0


def test_ask_deep_mode_routes_to_cloud_engine(tmp_path: Path) -> None:
    local = StubLocalQAEngine()
    cloud = StubCloudAgentEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Find cancellation clause",
                "mode": "deep",
                "document_id": "doc-2",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "cloud-answer"
    assert len(local.calls) == 0
    assert len(cloud.calls) == 1


def test_ask_deep_mode_returns_capability_error_when_disabled(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path, deep_mode_enabled=False)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Find cancellation clause",
                "mode": "deep",
                "document_id": "doc-2",
            },
        )

    assert response.status_code == 503
    assert response.json()["code"] == "deep_mode_disabled"


def test_ask_deep_mode_returns_deterministic_provider_error_envelope(
    tmp_path: Path,
) -> None:
    local = StubLocalQAEngine()
    cloud = FailingCloudAgentEngine("provider_timeout")
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Find cancellation clause",
                "mode": "deep",
                "document_id": "doc-2",
            },
        )

    assert response.status_code == 504
    body = response.json()
    assert body["code"] == "provider_timeout"
    assert body["message"] == "deep mode provider request failed"


def test_ask_deep_mode_allows_parsed_artifact_missing_as_non_500(
    tmp_path: Path,
) -> None:
    local = StubLocalQAEngine()
    cloud = ParsedArtifactMissingCloudAgentEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Find cancellation clause",
                "mode": "deep",
                "document_id": "doc-2",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["trace"]["termination_reason"] == "parsed_artifact_missing"


def test_ask_with_session_id_reuses_previous_turn_context(tmp_path: Path) -> None:
    local = StubLocalQAEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local)

    with _client_with_services(services) as client:
        first = client.post(
            "/ask",
            json={
                "question": "What does this invoice cover?",
                "mode": "fast",
                "document_id": "doc-1",
                "session_id": "session-1",
            },
        )
        second = client.post(
            "/ask",
            json={
                "question": "What is the total due?",
                "mode": "fast",
                "document_id": "doc-1",
                "session_id": "session-1",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert local.chat_history_calls[0] == []
    assert local.chat_history_calls[1] == [
        {
            "question": "What does this invoice cover?",
            "answer": "local-answer",
        }
    ]


def test_ask_stream_with_session_id_reuses_previous_turn_context(
    tmp_path: Path,
) -> None:
    local = StubLocalQAEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local)

    with _client_with_services(services) as client:
        first = client.post(
            "/ask/stream",
            json={
                "question": "Summarize this document",
                "mode": "fast",
                "document_id": "doc-1",
                "session_id": "session-stream-1",
            },
        )
        second = client.post(
            "/ask/stream",
            json={
                "question": "Can you expand on that?",
                "mode": "fast",
                "document_id": "doc-1",
                "session_id": "session-stream-1",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert local.chat_history_calls[0] == []
    assert local.chat_history_calls[1] == [
        {
            "question": "Summarize this document",
            "answer": "local-answer",
        }
    ]


def test_ingest_idempotency_replay_survives_service_restart(tmp_path: Path) -> None:
    shared_backend = InMemoryApiStateBackend()

    services_before = _make_services(tmp_path=tmp_path)
    services_before.state_store = _shared_state_store(
        cfg=services_before.cfg,
        backend=shared_backend,
    )

    services_after = _make_services(tmp_path=tmp_path)
    services_after.state_store = _shared_state_store(
        cfg=services_after.cfg,
        backend=shared_backend,
    )

    with _client_with_services(services_before) as client:
        first = client.post(
            "/ingest",
            files={"file": ("invoice.pdf", b"%PDF-1.4\nfirst", "application/pdf")},
            headers={"Idempotency-Key": "restart-idem-123"},
        )

    with _client_with_services(services_after) as client:
        replay = client.post(
            "/ingest",
            files={
                "file": (
                    "invoice.pdf",
                    b"%PDF-1.4\nchanged-body-should-replay",
                    "application/pdf",
                )
            },
            headers={"Idempotency-Key": "restart-idem-123"},
        )

    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["document_id"] == first.json()["document_id"]
    assert replay.json()["file_path"] == first.json()["file_path"]


def test_session_history_survives_service_restart(tmp_path: Path) -> None:
    shared_backend = InMemoryApiStateBackend()

    local_before = StubLocalQAEngine()
    services_before = _make_services(tmp_path=tmp_path, local_qa=local_before)
    services_before.state_store = _shared_state_store(
        cfg=services_before.cfg,
        backend=shared_backend,
    )

    local_after = StubLocalQAEngine()
    services_after = _make_services(tmp_path=tmp_path, local_qa=local_after)
    services_after.state_store = _shared_state_store(
        cfg=services_after.cfg,
        backend=shared_backend,
    )

    with _client_with_services(services_before) as client:
        first = client.post(
            "/ask",
            json={
                "question": "What does this invoice cover?",
                "mode": "fast",
                "document_id": "doc-1",
                "session_id": "session-restart-1",
            },
        )
    assert first.status_code == 200

    with _client_with_services(services_after) as client:
        second = client.post(
            "/ask",
            json={
                "question": "What is the total due?",
                "mode": "fast",
                "document_id": "doc-1",
                "session_id": "session-restart-1",
            },
        )

    assert second.status_code == 200
    assert local_after.chat_history_calls[0] == [
        {
            "question": "What does this invoice cover?",
            "answer": "local-answer",
        }
    ]


def test_ask_stream_fast_mode_emits_tokens_and_final_event(tmp_path: Path) -> None:
    local = StubLocalQAEngine()
    cloud = StubCloudAgentEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask/stream",
            json={"question": "What is due?", "mode": "fast", "document_id": "doc-1"},
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 3

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    last = json.loads(lines[-1])
    assert first["type"] == "status"
    assert second["type"] == "token"
    assert second["delta"] == "local-answer"
    assert last["type"] == "final"
    assert last["response"]["answer"] == "local-answer"


def test_ask_stream_deep_mode_emits_normalized_events(tmp_path: Path) -> None:
    local = StubLocalQAEngine()
    cloud = VerboseCloudAgentEngine()
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask/stream",
            json={
                "question": "Give me a deep answer",
                "mode": "deep",
                "document_id": "doc-1",
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 3

    first = json.loads(lines[0])
    token_events = [json.loads(line) for line in lines[1:-1]]
    last = json.loads(lines[-1])

    assert first["type"] == "status"
    assert first["phase"] == "generation"
    assert all(event["type"] == "token" for event in token_events)
    assert "".join(event["delta"] for event in token_events) == "deep stream response"
    assert last["type"] == "final"
    assert last["response"]["answer"] == "deep stream response"


def test_ask_stream_deep_mode_surfaces_provider_error_envelope(tmp_path: Path) -> None:
    local = StubLocalQAEngine()
    cloud = FailingCloudAgentEngine("provider_rate_limited")
    services = _make_services(tmp_path=tmp_path, local_qa=local, cloud_agent=cloud)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask/stream",
            json={
                "question": "Give me a deep answer",
                "mode": "deep",
                "document_id": "doc-1",
            },
        )

    assert response.status_code == 429
    body = response.json()
    assert body["code"] == "provider_rate_limited"
    assert body["message"] == "deep mode provider request failed"


def test_ask_validation_error_is_normalized(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        response = client.post("/ask", json={"mode": "fast"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "errors" in body["details"]


def test_ask_rejects_invalid_model_backend_header(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={"question": "What is due?", "mode": "fast", "document_id": "doc-1"},
            headers={"x-model-backend": "unknown"},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_model_backend"


def test_ask_requires_api_key_when_api_backend_requested(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path)

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={"question": "What is due?", "mode": "fast", "document_id": "doc-1"},
            headers={"x-model-backend": "api"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "missing_api_key"


def test_ask_timeout_returns_gateway_timeout(tmp_path: Path) -> None:
    services = _make_services(
        tmp_path=tmp_path,
        local_qa=SlowLocalQAEngine(),
        request_timeout_seconds=1,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={"question": "slow", "mode": "fast", "document_id": "doc-timeout"},
        )

    assert response.status_code == 504
    assert response.json()["code"] == "ask_timeout"
