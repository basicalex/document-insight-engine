from __future__ import annotations

from typing import Any, MutableMapping


ALLOWED_MODES = ("fast", "deep")
DEFAULT_API_BASE_URL = "http://localhost:8000"


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    state.setdefault("api_base_url", DEFAULT_API_BASE_URL)
    state.setdefault("chat_mode", "fast")
    state.setdefault("active_document_id", "")
    state.setdefault("session_id", "")
    state.setdefault("messages", [])
    state.setdefault("ingest_history", [])


def set_mode(state: MutableMapping[str, Any], mode: str) -> None:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    state["chat_mode"] = mode


def set_document_id(state: MutableMapping[str, Any], document_id: str) -> None:
    state["active_document_id"] = document_id.strip()


def append_user_message(state: MutableMapping[str, Any], content: str) -> None:
    state["messages"].append({"role": "user", "content": content})


def append_assistant_message(
    state: MutableMapping[str, Any],
    *,
    content: str,
    mode: str,
    insufficient_evidence: bool,
    citations: list[dict[str, Any]] | None,
    trace: dict[str, Any] | None,
) -> None:
    state["messages"].append(
        {
            "role": "assistant",
            "content": content,
            "mode": mode,
            "insufficient_evidence": insufficient_evidence,
            "citations": citations or [],
            "trace": trace,
        }
    )


def clear_chat(state: MutableMapping[str, Any]) -> None:
    state["messages"] = []
