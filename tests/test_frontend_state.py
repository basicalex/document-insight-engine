from __future__ import annotations

from typing import Any

import pytest

from frontend.state import (
    DEFAULT_API_BASE_URL,
    DEFAULT_EXTRACTION_SCHEMA,
    append_assistant_message,
    append_user_message,
    clear_chat,
    initialize_session_state,
    set_mode,
)


def test_initialize_session_state_sets_expected_defaults() -> None:
    state: dict[str, Any] = {}

    initialize_session_state(state)

    assert state["api_base_url"] == DEFAULT_API_BASE_URL
    assert state["chat_mode"] == "fast"
    assert state["active_document_id"] == ""
    assert state["session_id"] == ""
    assert state["messages"] == []
    assert state["ingest_history"] == []
    assert state["runtime_health"] == {}
    assert state["metrics_text"] == ""
    assert state["runtime_bootstrapped"] is False
    assert state["extract_schema_text"] == DEFAULT_EXTRACTION_SCHEMA
    assert state["extract_prompt"]
    assert state["last_extract_result"] is None
    assert state["model_backend"] == "local"
    assert state["api_model"] == "gemini-2.5-flash"
    assert state["api_key"] == ""


def test_initialize_session_state_preserves_existing_values() -> None:
    state: dict[str, Any] = {
        "api_base_url": "http://api.example",
        "chat_mode": "deep",
        "messages": [{"role": "user", "content": "hello"}],
    }

    initialize_session_state(state)

    assert state["api_base_url"] == "http://api.example"
    assert state["chat_mode"] == "deep"
    assert len(state["messages"]) == 1


def test_initialize_session_state_honors_api_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCUMENT_INSIGHT_API_BASE_URL", "http://api:8000")
    state: dict[str, Any] = {}

    initialize_session_state(state)

    assert state["api_base_url"] == "http://api:8000"


def test_set_mode_rejects_unknown_values() -> None:
    state: dict[str, Any] = {}
    initialize_session_state(state)

    with pytest.raises(ValueError, match="unsupported mode"):
        set_mode(state, "turbo")


def test_set_mode_accepts_deep_lite() -> None:
    state: dict[str, Any] = {}
    initialize_session_state(state)

    set_mode(state, "deep-lite")

    assert state["chat_mode"] == "deep-lite"


def test_chat_message_helpers_append_messages_and_clear() -> None:
    state: dict[str, Any] = {}
    initialize_session_state(state)

    append_user_message(state, "What is the renewal date?")
    append_assistant_message(
        state,
        content="The renewal date is May 1.",
        mode="fast",
        insufficient_evidence=False,
        citations=[{"chunk_id": "chunk-1", "text": "Renewal date: May 1"}],
        trace={"iterations": 1},
    )

    assert len(state["messages"]) == 2
    assert state["messages"][0]["role"] == "user"
    assert state["messages"][1]["role"] == "assistant"
    assert state["messages"][1]["trace"]["iterations"] == 1

    clear_chat(state)
    assert state["messages"] == []
