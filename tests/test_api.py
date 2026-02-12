from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient

from src.api.main import ApiServices, app
from src.config.settings import Settings
from src.ingestion import (
    HybridVectorIndexStore,
    InMemoryIndexBackend,
    UploadIntakeService,
)
from src.models.schemas import ChatResponse, Mode


class StubLocalQAEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mode, str | None]] = []

    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        self.calls.append((question, mode, document_id))
        return ChatResponse(
            answer="local-answer",
            mode=mode,
            document_id=document_id,
            insufficient_evidence=False,
            citations=[],
        )


class SlowLocalQAEngine(StubLocalQAEngine):
    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        time.sleep(1.2)
        return super().ask(question=question, mode=mode, document_id=document_id)


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


def _make_services(
    *,
    tmp_path: Path,
    local_qa: StubLocalQAEngine | None = None,
    cloud_agent: StubCloudAgentEngine | None = None,
    request_timeout_seconds: int = 10,
) -> ApiServices:
    cfg = Settings(
        data_dir=tmp_path,
        max_upload_size_mb=1,
        request_timeout_seconds=request_timeout_seconds,
        ingest_timeout_seconds=max(request_timeout_seconds, 1),
    )
    backend = InMemoryIndexBackend()
    index_store = HybridVectorIndexStore(
        cfg=cfg,
        backend=backend,
        tier1_dimension=4,
        tier4_dimension=6,
    )
    index_store.bootstrap_indices()

    return ApiServices(
        cfg=cfg,
        intake=UploadIntakeService(cfg),
        index_store=index_store,
        local_qa=local_qa or StubLocalQAEngine(),
        cloud_agent=cloud_agent or StubCloudAgentEngine(),
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
