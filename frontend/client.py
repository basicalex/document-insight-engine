from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
import json
from typing import Any, Iterator

import httpx


DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(slots=True)
class ApiError(Exception):
    status_code: int
    code: str
    message: str
    correlation_id: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        parts = [f"{self.code}: {self.message}"]
        if self.correlation_id:
            parts.append(f"correlation_id={self.correlation_id}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        return " | ".join(parts)


class DocumentInsightApi:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            base_url=normalized_base_url,
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    def __enter__(self) -> "DocumentInsightApi":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def ingest(
        self,
        *,
        file_name: str,
        content: bytes,
        content_type: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        try:
            response = self._client.post(
                "/ingest",
                files={"file": (file_name, content, content_type)},
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                status_code=0,
                code="network_error",
                message="unable to reach API service",
                details={"error": str(exc)},
            ) from exc

        return _decode_response(response)

    def upload(
        self,
        *,
        files: list[tuple[str, bytes, str]],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        multipart_files = [
            ("files", (file_name, content, content_type))
            for file_name, content, content_type in files
        ]
        try:
            response = self._client.post(
                "/upload",
                files=multipart_files,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                status_code=0,
                code="network_error",
                message="unable to reach API service",
                details={"error": str(exc)},
            ) from exc

        return _decode_response(response)

    def get_ingest_status(self, *, document_id: str) -> dict[str, Any]:
        try:
            response = self._client.get(f"/ingest/{document_id}")
        except httpx.HTTPError as exc:
            raise ApiError(
                status_code=0,
                code="network_error",
                message="unable to reach API service",
                details={"error": str(exc)},
            ) from exc
        return _decode_response(response)

    def ask(
        self,
        *,
        question: str,
        mode: str,
        document_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, "mode": mode}
        if document_id:
            payload["document_id"] = document_id
        if session_id:
            payload["session_id"] = session_id

        try:
            response = self._client.post("/ask", json=payload)
        except httpx.HTTPError as exc:
            raise ApiError(
                status_code=0,
                code="network_error",
                message="unable to reach API service",
                details={"error": str(exc)},
            ) from exc

        return _decode_response(response)

    def ask_stream_events(
        self,
        *,
        question: str,
        mode: str,
        document_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"question": question, "mode": mode}
        if document_id:
            payload["document_id"] = document_id
        if session_id:
            payload["session_id"] = session_id

        try:
            with self._client.stream("POST", "/ask/stream", json=payload) as response:
                if response.is_error:
                    raise _build_api_error(
                        response=response, payload=_safe_json(response)
                    )

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        yield event
        except httpx.HTTPError as exc:
            raise ApiError(
                status_code=0,
                code="network_error",
                message="unable to reach API service",
                details={"error": str(exc)},
            ) from exc


def _decode_response(response: httpx.Response) -> dict[str, Any]:
    payload = _safe_json(response)
    if response.is_error:
        raise _build_api_error(response=response, payload=payload)
    if isinstance(payload, dict):
        return payload
    return {}


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {}


def _build_api_error(*, response: httpx.Response, payload: Any) -> ApiError:
    if isinstance(payload, dict):
        code = str(payload.get("code") or "http_error")
        message = str(
            payload.get("message") or _default_http_message(response.status_code)
        )
        correlation_id = payload.get("correlation_id")
        details = payload.get("details")
        normalized_details = details if isinstance(details, dict) else {}
    else:
        code = "http_error"
        message = _default_http_message(response.status_code)
        correlation_id = None
        normalized_details = {}

    if not correlation_id:
        correlation_id = response.headers.get("x-correlation-id")

    return ApiError(
        status_code=response.status_code,
        code=code,
        message=message,
        correlation_id=correlation_id,
        details=normalized_details,
    )


def _default_http_message(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase.lower()
    except ValueError:
        return "request failed"
