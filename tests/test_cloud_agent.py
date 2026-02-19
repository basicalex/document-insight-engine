from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.cloud_agent import ALLOWED_TOOL_NAMES, CloudAgentEngine
from src.models.schemas import Mode


@dataclass
class ScriptedModel:
    steps: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "question": question,
                "mode": mode,
                "document_id": document_id,
                "iteration": iteration,
                "history_size": len(history),
                "allowed_tools": allowed_tools,
            }
        )
        index = min(iteration - 1, len(self.steps) - 1)
        return self.steps[index]


def test_cloud_agent_rejects_non_allowlisted_tool_request() -> None:
    model = ScriptedModel(
        steps=[
            {
                "action": "tool_call",
                "tool_name": "delete_file",
                "arguments": {"path": "data/parsed/doc.md"},
            }
        ]
    )
    engine = CloudAgentEngine(
        model_client=model,
        tool_provider=lambda _document_id: {},
    )

    response = engine.ask(
        question="Find contradictions.",
        mode=Mode.DEEP,
        document_id="doc-1",
    )

    assert response.insufficient_evidence is True
    assert response.trace is not None
    assert response.trace.iterations == 1
    assert response.trace.termination_reason == "tool_not_allowed"
    assert response.trace.tool_calls[0].metadata["tool_name"] == "delete_file"
    assert set(model.calls[0]["allowed_tools"]) == set(ALLOWED_TOOL_NAMES)


def test_cloud_agent_stops_after_five_iterations_when_no_final_answer() -> None:
    call_counter = {"list_sections": 0}

    def list_sections(limit: int = 200) -> dict[str, Any]:
        call_counter["list_sections"] += 1
        return {
            "ok": True,
            "sections": [],
            "total_sections": 0,
            "truncated": False,
            "limit": limit,
        }

    model = ScriptedModel(
        steps=[
            {
                "action": "tool_call",
                "tool_name": "list_sections",
                "arguments": {"limit": 5},
            }
        ]
    )
    engine = CloudAgentEngine(
        model_client=model,
        tool_provider=lambda _document_id: {"list_sections": list_sections},
        max_iterations=5,
    )

    response = engine.ask(
        question="Locate all contradiction points.",
        mode=Mode.DEEP,
        document_id="doc-2",
    )

    assert response.insufficient_evidence is True
    assert response.trace is not None
    assert response.trace.iterations == 5
    assert response.trace.termination_reason == "max_iterations_reached"
    assert call_counter["list_sections"] == 5
    assert len(model.calls) == 5


def test_cloud_agent_executes_allowed_tool_then_returns_final_answer() -> None:
    def read_section(section_key: str, max_chars: int | None = None) -> dict[str, Any]:
        del max_chars
        return {
            "ok": True,
            "section": {"key": section_key, "title": "Confidentiality"},
            "content": "The agreement requires 30-day notice.",
            "truncated": False,
        }

    model = ScriptedModel(
        steps=[
            {
                "action": "tool_call",
                "tool_name": "read_section",
                "arguments": {"section_key": "msa/confidentiality"},
            },
            {
                "action": "final",
                "answer": "The notice period is 30 days.",
                "insufficient_evidence": False,
            },
        ]
    )
    engine = CloudAgentEngine(
        model_client=model,
        tool_provider=lambda _document_id: {"read_section": read_section},
    )

    response = engine.ask(
        question="What notice period is required?",
        mode=Mode.DEEP,
        document_id="doc-3",
    )

    assert response.answer == "The notice period is 30 days."
    assert response.insufficient_evidence is False
    assert response.trace is not None
    assert response.trace.iterations == 2
    assert response.trace.termination_reason == "completed"
    assert response.trace.retrieved_chunk_ids == ["msa/confidentiality"]


def test_cloud_agent_rejects_malformed_provider_payload() -> None:
    model = ScriptedModel(steps=[{"action": "unknown"}])
    engine = CloudAgentEngine(
        model_client=model,
        tool_provider=lambda _document_id: {},
    )

    response = engine.ask(
        question="Find obligations.",
        mode=Mode.DEEP,
        document_id="doc-4",
    )

    assert response.insufficient_evidence is True
    assert response.trace is not None
    assert response.trace.iterations == 1
    assert response.trace.termination_reason == "provider_malformed_response"


def test_cloud_agent_handles_missing_parsed_artifact_without_crashing() -> None:
    model = ScriptedModel(
        steps=[
            {
                "action": "final",
                "answer": "unused",
                "insufficient_evidence": False,
            }
        ]
    )

    def broken_tool_provider(_document_id: str) -> dict[str, Any]:
        raise FileNotFoundError("no parsed markdown found for document")

    engine = CloudAgentEngine(
        model_client=model,
        tool_provider=broken_tool_provider,
    )

    response = engine.ask(
        question="Find contradictions.",
        mode=Mode.DEEP,
        document_id="doc-missing",
    )

    assert response.insufficient_evidence is True
    assert "Re-ingest the document" in response.answer
    assert response.trace is not None
    assert response.trace.iterations == 0
    assert response.trace.termination_reason == "parsed_artifact_missing"
    assert response.trace.tool_calls[0].metadata["code"] == "parsed_artifact_missing"
    assert len(model.calls) == 0
