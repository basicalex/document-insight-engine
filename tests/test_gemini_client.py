from __future__ import annotations

from typing import Any

import pytest

from src.config.settings import Settings
from src.engine.cloud_agent import CloudAgentProviderError, DeepProviderErrorCode
from src.engine.gemini_client import GeminiCloudModelClient, _HttpResponse
from src.models.schemas import Mode


def test_gemini_client_rejects_missing_api_key() -> None:
    cfg = Settings(cloud_agent_provider="fallback", cloud_agent_api_key=None)
    client = GeminiCloudModelClient(cfg=cfg)

    with pytest.raises(CloudAgentProviderError) as exc_info:
        client.next_step(
            question="What is due?",
            mode=Mode.DEEP,
            document_id="doc-1",
            iteration=1,
            history=[],
            allowed_tools=["list_sections"],
        )

    assert exc_info.value.code == DeepProviderErrorCode.NOT_CONFIGURED


def test_gemini_client_parses_function_call_response() -> None:
    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
        _timeout_seconds: float,
    ) -> _HttpResponse:
        return _HttpResponse(
            status_code=200,
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "read_section",
                                        "args": {"section_key": "contract/termination"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    cfg = Settings(cloud_agent_provider="gemini", cloud_agent_api_key="test-key")
    client = GeminiCloudModelClient(
        cfg=cfg, transport=transport, sleep_fn=lambda _s: None
    )

    decision = client.next_step(
        question="What is the termination notice period?",
        mode=Mode.DEEP,
        document_id="doc-1",
        iteration=1,
        history=[],
        allowed_tools=["read_section"],
    )

    assert decision["action"] == "tool_call"
    assert decision["tool_name"] == "read_section"
    assert decision["arguments"]["section_key"] == "contract/termination"


def test_gemini_client_retries_rate_limit_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
        _timeout_seconds: float,
    ) -> _HttpResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return _HttpResponse(status_code=429, payload={"error": {"code": 429}})
        return _HttpResponse(
            status_code=200,
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"action":"final","answer":"Done",'
                                        '"insufficient_evidence":false}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    cfg = Settings(cloud_agent_provider="gemini", cloud_agent_api_key="test-key")
    client = GeminiCloudModelClient(
        cfg=cfg, transport=transport, sleep_fn=sleeps.append
    )

    decision = client.next_step(
        question="Summarize",
        mode=Mode.DEEP,
        document_id="doc-1",
        iteration=1,
        history=[],
        allowed_tools=["list_sections"],
    )

    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert decision["action"] == "final"
    assert decision["answer"] == "Done"


def test_gemini_client_raises_malformed_response_error() -> None:
    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
        _timeout_seconds: float,
    ) -> _HttpResponse:
        return _HttpResponse(status_code=200, payload={"candidates": []})

    cfg = Settings(cloud_agent_provider="gemini", cloud_agent_api_key="test-key")
    client = GeminiCloudModelClient(
        cfg=cfg, transport=transport, sleep_fn=lambda _s: None
    )

    with pytest.raises(CloudAgentProviderError) as exc_info:
        client.next_step(
            question="Summarize",
            mode=Mode.DEEP,
            document_id="doc-1",
            iteration=1,
            history=[],
            allowed_tools=["list_sections"],
        )

    assert exc_info.value.code == DeepProviderErrorCode.MALFORMED_RESPONSE


def test_gemini_client_raises_timeout_after_retry_exhaustion() -> None:
    calls = {"count": 0}

    def transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
        _timeout_seconds: float,
    ) -> _HttpResponse:
        calls["count"] += 1
        raise TimeoutError("timed out")

    cfg = Settings(
        cloud_agent_provider="gemini",
        cloud_agent_api_key="test-key",
        cloud_agent_retry_attempts=2,
    )
    client = GeminiCloudModelClient(
        cfg=cfg, transport=transport, sleep_fn=lambda _s: None
    )

    with pytest.raises(CloudAgentProviderError) as exc_info:
        client.next_step(
            question="Summarize",
            mode=Mode.DEEP,
            document_id="doc-1",
            iteration=1,
            history=[],
            allowed_tools=["list_sections"],
        )

    assert calls["count"] == 2
    assert exc_info.value.code == DeepProviderErrorCode.TIMEOUT
