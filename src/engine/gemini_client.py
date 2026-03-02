from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request

from src.config.settings import Settings, settings
from src.engine.cloud_agent import CloudAgentProviderError, DeepProviderErrorCode
from src.models.schemas import Mode


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class _HttpResponse:
    status_code: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class _ProviderHttpError(Exception):
    status_code: int
    payload: dict[str, Any]
    message: str


class GeminiCloudModelClient:
    def __init__(
        self,
        *,
        cfg: Settings = settings,
        transport: (
            Callable[[str, dict[str, Any], dict[str, str], float], _HttpResponse] | None
        ) = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.model = cfg.cloud_agent_model
        self.api_key = (cfg.cloud_agent_api_key or "").strip()
        self.base_url = cfg.cloud_agent_api_base_url.rstrip("/")
        self.timeout_seconds = float(cfg.cloud_agent_timeout_seconds)
        self.retry_attempts = cfg.cloud_agent_retry_attempts
        self.retry_initial_backoff = cfg.cloud_agent_retry_initial_backoff_seconds
        self.retry_backoff_factor = cfg.cloud_agent_retry_backoff_factor
        self.retry_max_backoff = cfg.cloud_agent_retry_max_backoff_seconds
        self._transport = transport or _default_transport
        self._sleep = sleep_fn

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
        if not self.api_key:
            raise CloudAgentProviderError(
                code=DeepProviderErrorCode.NOT_CONFIGURED,
                message="cloud provider credentials are missing",
            )

        body = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "You are a tool-using reasoning agent. Return either a tool call "
                            "or a final answer. If you provide text, return a JSON object with "
                            "keys action, answer, and insufficient_evidence."
                        )
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": _build_turn_prompt(
                                question=question,
                                mode=mode,
                                document_id=document_id,
                                iteration=iteration,
                                history=history,
                                allowed_tools=allowed_tools,
                            )
                        }
                    ],
                }
            ],
            "tools": [{"function_declarations": _tool_declarations(allowed_tools)}],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            "generation_config": {
                "temperature": 0.0,
            },
        }

        payload = self._request_with_retries(body)
        return _parse_provider_payload(payload)

    def _request_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        url = (
            f"{self.base_url}/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: _ProviderHttpError | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self._transport(url, body, headers, self.timeout_seconds)
                if response.status_code < 200 or response.status_code >= 300:
                    raise _ProviderHttpError(
                        status_code=response.status_code,
                        payload=response.payload,
                        message="provider returned non-success status",
                    )
                return response.payload
            except _ProviderHttpError as exc:
                last_error = exc
                auth_error = exc.status_code in {401, 403}
                retryable = exc.status_code in RETRYABLE_STATUS_CODES
                if auth_error:
                    raise CloudAgentProviderError(
                        code=DeepProviderErrorCode.AUTHENTICATION_FAILED,
                        message="provider rejected authentication credentials",
                    ) from exc
                if not retryable or attempt == self.retry_attempts:
                    code = (
                        DeepProviderErrorCode.RATE_LIMITED
                        if exc.status_code == 429
                        else DeepProviderErrorCode.UNAVAILABLE
                    )
                    raise CloudAgentProviderError(
                        code=code,
                        message="provider request failed after retry attempts",
                    ) from exc
                self._sleep(_backoff_seconds(attempt=attempt, client=self))
            except TimeoutError as exc:
                if attempt == self.retry_attempts:
                    raise CloudAgentProviderError(
                        code=DeepProviderErrorCode.TIMEOUT,
                        message="provider request timed out",
                    ) from exc
                self._sleep(_backoff_seconds(attempt=attempt, client=self))
            except error.URLError as exc:
                if attempt == self.retry_attempts:
                    raise CloudAgentProviderError(
                        code=DeepProviderErrorCode.UNAVAILABLE,
                        message="provider connection failed",
                    ) from exc
                self._sleep(_backoff_seconds(attempt=attempt, client=self))

        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.UNAVAILABLE,
            message="provider request failed",
        ) from last_error


def _default_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> _HttpResponse:
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if not isinstance(parsed, dict):
                raise _ProviderHttpError(
                    status_code=int(getattr(response, "status", 200)),
                    payload={},
                    message="provider returned non-object JSON",
                )
            return _HttpResponse(
                status_code=int(getattr(response, "status", 200)),
                payload=parsed,
            )
    except error.HTTPError as exc:
        response_payload = _safe_json_body(exc.read())
        raise _ProviderHttpError(
            status_code=exc.code,
            payload=response_payload,
            message=str(exc),
        ) from exc
    except error.URLError:
        raise


def _safe_json_body(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _parse_provider_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE,
            message="provider response had no candidates",
        )

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE,
            message="provider candidate is invalid",
        )

    content = candidate.get("content")
    if not isinstance(content, dict):
        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE,
            message="provider candidate content is missing",
        )

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE,
            message="provider candidate parts are invalid",
        )

    for part in parts:
        if not isinstance(part, dict):
            continue
        function_call = part.get("functionCall") or part.get("function_call")
        if isinstance(function_call, dict):
            tool_name = function_call.get("name")
            arguments = function_call.get("args", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise CloudAgentProviderError(
                    code=DeepProviderErrorCode.MALFORMED_RESPONSE,
                    message="provider function call is missing tool name",
                )
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise CloudAgentProviderError(
                    code=DeepProviderErrorCode.MALFORMED_RESPONSE,
                    message="provider function call args must be an object",
                )
            return {
                "action": "tool_call",
                "tool_name": tool_name.strip(),
                "arguments": arguments,
            }

    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()
    if not text:
        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE,
            message="provider response did not include callable content",
        )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"action": "final", "answer": text, "insufficient_evidence": False}

    if not isinstance(parsed, dict):
        raise CloudAgentProviderError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE,
            message="provider JSON response must be an object",
        )
    return parsed


def _tool_declarations(allowed_tools: list[str]) -> list[dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {
        "list_sections": {
            "name": "list_sections",
            "description": "List available document sections.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "limit": {"type": "INTEGER"},
                },
            },
        },
        "read_section": {
            "name": "read_section",
            "description": "Read a specific section by key.",
            "parameters": {
                "type": "OBJECT",
                "required": ["section_key"],
                "properties": {
                    "section_key": {"type": "STRING"},
                    "max_chars": {"type": "INTEGER"},
                },
            },
        },
        "keyword_grep": {
            "name": "keyword_grep",
            "description": "Search document sections by keyword.",
            "parameters": {
                "type": "OBJECT",
                "required": ["keyword"],
                "properties": {
                    "keyword": {"type": "STRING"},
                    "section_key": {"type": "STRING"},
                    "max_matches": {"type": "INTEGER"},
                    "context_chars": {"type": "INTEGER"},
                },
            },
        },
        "structured_extract": {
            "name": "structured_extract",
            "description": (
                "Run schema-based extraction with grounding provenance. "
                "Use section_key and max_chars for long documents."
            ),
            "parameters": {
                "type": "OBJECT",
                "required": ["schema"],
                "properties": {
                    "schema": {"type": "OBJECT"},
                    "prompt": {"type": "STRING"},
                    "section_key": {"type": "STRING"},
                    "max_chars": {"type": "INTEGER"},
                },
            },
        },
    }
    return [definitions[name] for name in allowed_tools if name in definitions]


def _build_turn_prompt(
    *,
    question: str,
    mode: Mode,
    document_id: str,
    iteration: int,
    history: list[dict[str, Any]],
    allowed_tools: list[str],
) -> str:
    return (
        "Return either:\n"
        "1) a function call for one of the allowed tools, or\n"
        '2) JSON object: {"action":"final","answer":"...",'
        '"insufficient_evidence":true|false}.\n\n'
        f"question: {question}\n"
        f"mode: {mode.value}\n"
        f"document_id: {document_id}\n"
        f"iteration: {iteration}\n"
        f"allowed_tools: {json.dumps(allowed_tools)}\n"
        f"history: {json.dumps(history, ensure_ascii=False)}"
    )


def _backoff_seconds(*, attempt: int, client: GeminiCloudModelClient) -> float:
    delay = client.retry_initial_backoff * (
        client.retry_backoff_factor ** (attempt - 1)
    )
    return min(delay, client.retry_max_backoff)
