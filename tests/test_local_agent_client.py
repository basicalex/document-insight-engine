from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.engine.cloud_agent import CloudAgentProviderError, DeepProviderErrorCode
from src.engine.local_agent_client import LocalDeepModelClient
from src.engine.local_llm import OllamaGenerateError
from src.models.schemas import Mode


class _StubOllamaClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str:
        del model, prompt, timeout_seconds
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _make_client(response: str | Exception) -> LocalDeepModelClient:
    client = LocalDeepModelClient(
        cfg=Settings(deep_mode_enabled=True, cloud_agent_provider="local")
    )
    object.__setattr__(client, "_client", _StubOllamaClient(response))
    return client


def test_local_deep_client_returns_final_json_payload() -> None:
    client = _make_client(
        '{"action":"final","answer":"Local deep answer","insufficient_evidence":false}'
    )
    step = client.next_step(
        question="What is due?",
        mode=Mode.DEEP,
        document_id="doc-1",
        iteration=1,
        history=[],
        allowed_tools=["list_sections", "read_section", "keyword_grep"],
    )

    assert step["action"] == "final"
    assert step["answer"] == "Local deep answer"
    assert step["insufficient_evidence"] is False


def test_local_deep_client_falls_back_when_response_not_json() -> None:
    client = _make_client("Plain model response")
    step = client.next_step(
        question="What is due?",
        mode=Mode.DEEP,
        document_id="doc-1",
        iteration=1,
        history=[],
        allowed_tools=["list_sections", "read_section", "keyword_grep"],
    )

    assert step["action"] == "final"
    assert step["answer"] == "Plain model response"


def test_local_deep_client_wraps_ollama_failure_as_provider_error() -> None:
    client = _make_client(OllamaGenerateError("connection refused"))

    with pytest.raises(CloudAgentProviderError) as exc:
        client.next_step(
            question="What is due?",
            mode=Mode.DEEP,
            document_id="doc-1",
            iteration=1,
            history=[],
            allowed_tools=["list_sections", "read_section", "keyword_grep"],
        )

    assert exc.value.code == DeepProviderErrorCode.UNAVAILABLE
