from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import ApiServices, app
from src.api.state_store import InMemoryApiStateBackend, InMemoryApiStateStore
from src.config.settings import Settings
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


def _make_services(
    *,
    tmp_path: Path,
    local_qa: StubLocalQAEngine | None = None,
    cloud_agent: StubCloudAgentEngine | None = None,
    orchestrator: Any | None = None,
    state_store: InMemoryApiStateStore | None = None,
    request_timeout_seconds: int = 10,
    deep_mode_enabled: bool = True,
    docling_enabled: bool = True,
    google_parser_enabled: bool = True,
    langextract_enabled: bool = True,
    parser_routing_mode: str = "docling_google_fallback",
) -> ApiServices:
    cfg = Settings(
        data_dir=tmp_path,
        max_upload_size_mb=1,
        request_timeout_seconds=request_timeout_seconds,
        ingest_timeout_seconds=max(request_timeout_seconds, 1),
        deep_mode_enabled=deep_mode_enabled,
        cloud_agent_provider="fallback" if deep_mode_enabled else "disabled",
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
    assert "pip install -e .[ai]" in capabilities["docling_parser"]["hint"]
    assert capabilities["google_parser"]["ready"] is False
    assert capabilities["google_parser"]["reason"] == "missing_api_key"
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
    assert "pip install -e .[ai]" in caplog.text


def test_healthz_marks_google_parser_disabled_by_config(tmp_path: Path) -> None:
    services = _make_services(tmp_path=tmp_path, google_parser_enabled=False)

    with _client_with_services(services) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["google_parser"]["enabled"] is False
    assert capabilities["google_parser"]["reason"] == "disabled_by_config"


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
    assert body["status"] == "indexed"
    assert body["message"] == "indexed and ready for retrieval"
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
    assert body["status"] == "indexed"
    assert body["message"] == "indexed and ready for retrieval"


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
    assert all(item["status"] == "indexed" for item in body["documents"])


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
