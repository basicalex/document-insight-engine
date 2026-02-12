from __future__ import annotations

import json

import httpx
import pytest

from frontend.client import ApiError, DocumentInsightApi


def test_ingest_sends_request_and_returns_contract_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/ingest"
        assert request.headers["Idempotency-Key"] == "idem-123"
        assert request.headers["content-type"].startswith("multipart/form-data")
        return httpx.Response(
            201,
            json={
                "document_id": "doc_123",
                "file_path": "data/uploads/doc_123_invoice.pdf",
                "status": "uploaded",
                "message": "queued for processing",
            },
        )

    client = DocumentInsightApi(
        base_url="http://testserver",
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )

    response = client.ingest(
        file_name="invoice.pdf",
        content=b"%PDF-1.4\nexample",
        content_type="application/pdf",
        idempotency_key="idem-123",
    )

    assert response["document_id"] == "doc_123"
    assert response["status"] == "uploaded"


def test_ask_sends_payload_and_returns_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/ask"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {
            "question": "What is the total due?",
            "mode": "fast",
            "document_id": "doc_123",
            "session_id": "session-1",
        }
        return httpx.Response(
            200,
            json={
                "answer": "Total due is 1234.00 USD.",
                "mode": "fast",
                "document_id": "doc_123",
                "insufficient_evidence": False,
                "citations": [
                    {"chunk_id": "chunk-42", "text": "Total Due: 1234.00 USD"}
                ],
                "trace": {"iterations": 1},
            },
        )

    client = DocumentInsightApi(
        base_url="http://testserver",
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )

    response = client.ask(
        question="What is the total due?",
        mode="fast",
        document_id="doc_123",
        session_id="session-1",
    )

    assert response["answer"].startswith("Total due")
    assert response["trace"]["iterations"] == 1


def test_upload_supports_multi_document_batch_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/upload"
        assert request.headers["content-type"].startswith("multipart/form-data")
        return httpx.Response(
            201,
            json={
                "documents": [
                    {
                        "document_id": "doc_1",
                        "file_path": "data/uploads/doc_1_invoice.pdf",
                        "status": "indexed",
                        "message": "indexed and ready for retrieval",
                    },
                    {
                        "document_id": "doc_2",
                        "file_path": "data/uploads/doc_2_contract.pdf",
                        "status": "indexed",
                        "message": "indexed and ready for retrieval",
                    },
                ],
                "count": 2,
            },
        )

    client = DocumentInsightApi(
        base_url="http://testserver",
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )

    response = client.upload(
        files=[
            ("invoice.pdf", b"%PDF-1.4\ninvoice", "application/pdf"),
            ("contract.pdf", b"%PDF-1.4\ncontract", "application/pdf"),
        ]
    )

    assert response["count"] == 2
    assert len(response["documents"]) == 2


def test_get_ingest_status_fetches_document_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/ingest/doc_123"
        return httpx.Response(
            200,
            json={
                "document_id": "doc_123",
                "file_path": "data/uploads/doc_123_invoice.pdf",
                "status": "indexed",
                "message": "indexed and ready for retrieval",
            },
        )

    client = DocumentInsightApi(
        base_url="http://testserver",
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )

    response = client.get_ingest_status(document_id="doc_123")

    assert response["document_id"] == "doc_123"
    assert response["status"] == "indexed"


def test_error_envelope_is_normalized_to_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "validation_error",
                "message": "request validation failed",
                "correlation_id": "corr-123",
                "details": {"errors": [{"loc": ["body", "question"]}]},
            },
        )

    client = DocumentInsightApi(
        base_url="http://testserver",
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )

    with pytest.raises(ApiError) as exc_info:
        client.ask(question="", mode="fast")

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "validation_error"
    assert exc_info.value.correlation_id == "corr-123"


def test_network_error_is_mapped_to_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = DocumentInsightApi(
        base_url="http://testserver",
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )

    with pytest.raises(ApiError) as exc_info:
        client.ask(question="health check", mode="fast")

    assert exc_info.value.status_code == 0
    assert exc_info.value.code == "network_error"
