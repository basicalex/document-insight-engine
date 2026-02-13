from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient

from src.api.main import ApiServices, app
from src.api.state_store import InMemoryApiStateStore
from src.config.settings import Settings
from src.engine.cloud_agent import CloudAgentEngine
from src.engine.extractor import Tier4StructuredExtractor
from src.engine.local_llm import LocalQAEngine
from src.ingestion import (
    EmbeddingTier,
    HybridVectorIndexStore,
    InMemoryIndexBackend,
    IndexRecord,
    UploadIntakeService,
)


class DeterministicEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_query(self, text: str, dimension: int) -> list[float]:
        del text
        if dimension != self.dimension:
            raise AssertionError("unexpected embedding dimension")
        return [0.25 for _ in range(dimension)]


@dataclass
class DeterministicOllama:
    answer: str

    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str:
        del model, prompt, timeout_seconds
        return self.answer


class OneShotCloudModel:
    def next_step(
        self,
        *,
        question: str,
        mode: object,
        document_id: str,
        iteration: int,
        history: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        del question, mode, document_id, iteration, history, allowed_tools
        return {
            "action": "final",
            "answer": "Deep mode fallback answer.",
            "insufficient_evidence": False,
        }


class LoopingCloudModel:
    def next_step(
        self,
        *,
        question: str,
        mode: object,
        document_id: str,
        iteration: int,
        history: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        del question, mode, document_id, iteration, history, allowed_tools
        return {"action": "tool_call", "tool_name": "list_sections", "arguments": {}}


@dataclass
class StubLangExtractClient:
    payload: dict[str, Any]

    def extract(
        self,
        *,
        document_text: str,
        schema: dict[str, Any],
        prompt: str | None,
        model_name: str,
    ) -> dict[str, Any]:
        del document_text, schema, prompt, model_name
        return self.payload


def _build_services(
    *,
    tmp_path: Path,
    backend: InMemoryIndexBackend,
    cloud_agent: CloudAgentEngine | None = None,
) -> ApiServices:
    cfg = Settings(
        data_dir=tmp_path,
        max_upload_size_mb=2,
        request_timeout_seconds=5,
        ingest_timeout_seconds=10,
        deep_mode_enabled=True,
        cloud_agent_provider="fallback",
    )
    index_store = HybridVectorIndexStore(
        cfg=cfg,
        backend=backend,
        tier1_dimension=8,
        tier4_dimension=6,
    )
    index_store.bootstrap_indices()

    local_qa = LocalQAEngine(
        index_store=index_store,
        cfg=cfg,
        embedder=DeterministicEmbedder(dimension=8),
        ollama_client=DeterministicOllama(answer="Total due is 1234.00 USD."),
        query_vector_dimension=8,
    )

    cloud = cloud_agent or CloudAgentEngine(
        model_client=OneShotCloudModel(),
        tool_provider=lambda _document_id: {},
    )

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
        local_qa=local_qa,
        cloud_agent=cloud,
        state_store=InMemoryApiStateStore(
            idempotency_ttl_seconds=cfg.api_state_idempotency_ttl_seconds,
            ingestion_ttl_seconds=cfg.api_state_ingestion_ttl_seconds,
            session_ttl_seconds=cfg.api_state_session_ttl_seconds,
            idempotency_claim_ttl_seconds=cfg.api_state_idempotency_claim_ttl_seconds,
        ),
    )


def _persist_grounding_chunk(
    index_store: HybridVectorIndexStore, document_id: str
) -> None:
    index_store.persist_records(
        EmbeddingTier.TIER1,
        [
            IndexRecord(
                record_id=f"{document_id}:chunk-1",
                document_id=document_id,
                chunk_id="chunk-1",
                text="Invoice line item: total due is 1234.00 USD.",
                vector=[0.25 for _ in range(8)],
                page_refs=[1],
                section_path=("invoice", "summary"),
            )
        ],
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


def test_phase_fast_ingest_to_ask_meets_latency_budget(
    tmp_path: Path,
    record_property: Any,
) -> None:
    services = _build_services(tmp_path=tmp_path, backend=InMemoryIndexBackend())

    with _client_with_services(services) as client:
        ingest_response = client.post(
            "/ingest",
            files={"file": ("invoice.pdf", b"%PDF-1.4\ninvoice", "application/pdf")},
        )
        assert ingest_response.status_code == 201
        document_id = ingest_response.json()["document_id"]

        _persist_grounding_chunk(services.index_store, document_id)

        started = time.perf_counter()
        ask_response = client.post(
            "/ask",
            json={
                "question": "What is the total due?",
                "mode": "fast",
                "document_id": document_id,
            },
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["insufficient_evidence"] is False
    assert body["answer"] == "Total due is 1234.00 USD."
    assert body["citations"]
    assert body["trace"]["termination_reason"] == "completed"
    assert body["trace"]["total_latency_ms"] < 2000
    assert elapsed_ms < 2000
    record_property("tier1_request_latency_ms", elapsed_ms)


def test_phase_restart_retains_upload_and_index_state(tmp_path: Path) -> None:
    backend = InMemoryIndexBackend()
    services_before_restart = _build_services(tmp_path=tmp_path, backend=backend)

    with _client_with_services(services_before_restart) as client:
        ingest_response = client.post(
            "/ingest",
            files={"file": ("policy.pdf", b"%PDF-1.4\npolicy", "application/pdf")},
        )
        assert ingest_response.status_code == 201
        ingest_payload = ingest_response.json()
        document_id = ingest_payload["document_id"]
        file_path = Path(ingest_payload["file_path"])

        _persist_grounding_chunk(services_before_restart.index_store, document_id)

        before = client.post(
            "/ask",
            json={
                "question": "What is the total due?",
                "mode": "fast",
                "document_id": document_id,
            },
        )
        assert before.status_code == 200
        assert before.json()["insufficient_evidence"] is False

    services_after_restart = _build_services(tmp_path=tmp_path, backend=backend)

    with _client_with_services(services_after_restart) as client:
        assert file_path.exists()
        after = client.post(
            "/ask",
            json={
                "question": "What is the total due?",
                "mode": "fast",
                "document_id": document_id,
            },
        )

    assert after.status_code == 200
    assert after.json()["insufficient_evidence"] is False


def test_phase_deep_mode_enforces_five_step_loop_guard(tmp_path: Path) -> None:
    cloud_agent = CloudAgentEngine(
        model_client=LoopingCloudModel(),
        tool_provider=lambda _document_id: {
            "list_sections": lambda limit=200: {
                "ok": True,
                "sections": [],
                "total_sections": 0,
                "truncated": False,
                "limit": limit,
            }
        },
        max_iterations=5,
    )
    services = _build_services(
        tmp_path=tmp_path,
        backend=InMemoryIndexBackend(),
        cloud_agent=cloud_agent,
    )

    with _client_with_services(services) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Find contradictions.",
                "mode": "deep",
                "document_id": "doc-loop",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["trace"]["iterations"] == 5
    assert body["trace"]["termination_reason"] == "max_iterations_reached"


def test_phase_tier4_extraction_accuracy_and_grounding(
    record_property: Any,
) -> None:
    document = """Invoice Number: INV-2026-002
Total Due: 1234.00 USD
Due Date: 2026-03-01
"""
    invoice_text = "INV-2026-002"
    total_text = "1234.00"
    due_text = "2026-03-01"

    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total_due": {"type": "number"},
            "due_date": {"type": "string"},
        },
        "required": ["invoice_number", "total_due", "due_date"],
    }

    client = StubLangExtractClient(
        payload={
            "data": {
                "invoice_number": invoice_text,
                "total_due": 1234.0,
                "due_date": due_text,
            },
            "provenance": {
                "invoice_number": {
                    "start_offset": document.index(invoice_text),
                    "end_offset": document.index(invoice_text) + len(invoice_text),
                    "text": invoice_text,
                },
                "total_due": {
                    "start_offset": document.index(total_text),
                    "end_offset": document.index(total_text) + len(total_text),
                    "text": total_text,
                },
                "due_date": {
                    "start_offset": document.index(due_text),
                    "end_offset": document.index(due_text) + len(due_text),
                    "text": due_text,
                },
            },
        }
    )
    extractor = Tier4StructuredExtractor(client=client)

    envelope = extractor.extract_structured(
        document_id="doc-tier4",
        document_text=document,
        schema=schema,
        prompt="Extract invoice fields",
    )

    assert envelope.ok is True
    assert envelope.result is not None

    expected = {
        "invoice_number": invoice_text,
        "total_due": 1234.0,
        "due_date": due_text,
    }
    matched = sum(
        1 for key, value in expected.items() if envelope.result.data.get(key) == value
    )
    accuracy = matched / len(expected)

    assert accuracy >= 0.98
    for provenance in envelope.result.provenance.values():
        assert (
            document[provenance.start_offset : provenance.end_offset] == provenance.text
        )

    record_property("tier4_extraction_accuracy_pct", round(accuracy * 100, 2))
