import pytest
from pydantic import ValidationError

from src.models.schemas import AgentTrace, ChatRequest, ChatResponse, Citation, Mode


def test_chat_request_defaults_to_fast_mode() -> None:
    payload = ChatRequest(question="What is the payment term?")

    assert payload.mode is Mode.FAST


def test_chat_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_chat_request_accepts_deep_lite_mode() -> None:
    payload = ChatRequest(question="Summarize findings", mode="deep-lite")

    assert payload.mode is Mode.DEEP_LITE


def test_agent_trace_caps_iterations() -> None:
    with pytest.raises(ValidationError):
        AgentTrace(model="gemini-2.0-flash", iterations=6)


def test_chat_response_round_trip() -> None:
    response = ChatResponse(
        answer="The invoice total is $1,234.00.",
        mode=Mode.DEEP,
        citations=[
            Citation(
                chunk_id="chunk-1",
                text="Total due: $1,234.00",
                page=1,
                start_offset=10,
                end_offset=28,
            )
        ],
    )

    dumped = response.model_dump()

    assert dumped["mode"] == "deep"
    assert dumped["citations"][0]["chunk_id"] == "chunk-1"
