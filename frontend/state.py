from __future__ import annotations

import os
from typing import Any, MutableMapping


ALLOWED_MODES = ("fast", "deep-lite", "deep")
DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_EXTRACTION_SCHEMA = (
    "{\n"
    '  "type": "object",\n'
    '  "properties": {\n'
    '    "invoice_number": {"type": "string"},\n'
    '    "total_due": {"type": "number"},\n'
    '    "due_date": {"type": "string"}\n'
    "  },\n"
    '  "required": ["invoice_number", "total_due", "due_date"]\n'
    "}"
)


def initialize_session_state(state: MutableMapping[Any, Any]) -> None:
    state.setdefault(
        "api_base_url",
        os.getenv("DOCUMENT_INSIGHT_API_BASE_URL", DEFAULT_API_BASE_URL),
    )
    state.setdefault("chat_mode", "fast")
    state.setdefault("active_document_id", "")
    state.setdefault("session_id", "")
    state.setdefault("messages", [])
    state.setdefault("ingest_history", [])
    state.setdefault("runtime_health", {})
    state.setdefault("metrics_text", "")
    state.setdefault("runtime_bootstrapped", False)
    state.setdefault("extract_schema_text", DEFAULT_EXTRACTION_SCHEMA)
    state.setdefault("extract_prompt", "Extract requested fields with provenance")
    state.setdefault("last_extract_result", None)
    state.setdefault("model_backend", "local")
    state.setdefault("api_model", "gemini-2.5-flash")
    state.setdefault("api_key", "")


def set_mode(state: MutableMapping[Any, Any], mode: str) -> None:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    state["chat_mode"] = mode


def set_document_id(state: MutableMapping[Any, Any], document_id: str) -> None:
    state["active_document_id"] = document_id.strip()


def append_user_message(state: MutableMapping[Any, Any], content: str) -> None:
    state["messages"].append({"role": "user", "content": content})


def append_assistant_message(
    state: MutableMapping[Any, Any],
    *,
    content: str,
    mode: str,
    insufficient_evidence: bool,
    citations: list[dict[str, Any]] | None,
    trace: dict[str, Any] | None,
    backend_label: str | None = None,
) -> None:
    state["messages"].append(
        {
            "role": "assistant",
            "content": content,
            "mode": mode,
            "insufficient_evidence": insufficient_evidence,
            "citations": citations or [],
            "trace": trace,
            "backend_label": backend_label,
        }
    )


def clear_chat(state: MutableMapping[Any, Any]) -> None:
    state["messages"] = []
