from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.extractor import Tier4StructuredExtractor


@dataclass
class StubLangExtractClient:
    payload: dict[str, Any]
    calls: int = 0

    def extract(
        self,
        *,
        document_text: str,
        schema: dict[str, Any],
        prompt: str | None,
        model_name: str,
    ) -> dict[str, Any]:
        del document_text, schema, prompt, model_name
        self.calls += 1
        return self.payload


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "total_due": {"type": "number"},
            "due_date": {"type": "string"},
        },
        "required": ["total_due", "due_date"],
    }


def test_tier4_extractor_accepts_valid_grounded_payload() -> None:
    document = "Invoice Total: 1234.00 USD\nDue Date: 2026-03-01"
    total_start = document.index("1234.00")
    due_start = document.index("2026-03-01")

    client = StubLangExtractClient(
        payload={
            "data": {"total_due": 1234.0, "due_date": "2026-03-01"},
            "provenance": {
                "total_due": {
                    "start_offset": total_start,
                    "end_offset": total_start + len("1234.00"),
                    "text": "1234.00",
                },
                "due_date": {
                    "start_offset": due_start,
                    "end_offset": due_start + len("2026-03-01"),
                    "text": "2026-03-01",
                },
            },
        }
    )
    extractor = Tier4StructuredExtractor(client=client)

    envelope = extractor.extract_structured(
        document_id="doc-1",
        document_text=document,
        schema=_schema(),
        prompt="Extract invoice fields",
    )

    assert envelope.ok is True
    assert envelope.result is not None
    assert envelope.result.data["total_due"] == 1234.0
    assert envelope.result.provenance["due_date"].text == "2026-03-01"
    assert envelope.result.accepted_fields == ["due_date", "total_due"]
    assert envelope.result.rejected_fields == []
    assert client.calls == 1


def test_tier4_extractor_rejects_missing_or_ungrounded_required_fields() -> None:
    document = "Invoice Total: 1234.00 USD\nDue Date: 2026-03-01"
    total_start = document.index("1234.00")

    client = StubLangExtractClient(
        payload={
            "data": {
                "total_due": 1234.0,
                "due_date": "2026-03-01",
                "vendor_name": "Acme Corp",
            },
            "provenance": {
                "total_due": {
                    "start_offset": total_start,
                    "end_offset": total_start + len("1234.00"),
                    "text": "1234.00",
                }
            },
        }
    )
    extractor = Tier4StructuredExtractor(client=client)

    envelope = extractor.extract_structured(
        document_id="doc-2",
        document_text=document,
        schema=_schema(),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "validation_failed"
    diagnostic_codes = {item.code for item in envelope.error.diagnostics}
    assert "missing_provenance" in diagnostic_codes
    assert "unsupported_field" in diagnostic_codes
    assert "required_field_missing" in diagnostic_codes
    assert client.calls == 1


def test_tier4_extractor_fails_fast_when_input_token_budget_exceeded() -> None:
    document = "A" * 5000
    client = StubLangExtractClient(payload={"data": {}, "provenance": {}})
    extractor = Tier4StructuredExtractor(
        client=client,
        max_input_tokens=20,
    )

    envelope = extractor.extract_structured(
        document_id="doc-3",
        document_text=document,
        schema=_schema(),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "token_budget_exceeded"
    assert any(
        item.code == "input_budget_exceeded" for item in envelope.error.diagnostics
    )
    assert client.calls == 0
