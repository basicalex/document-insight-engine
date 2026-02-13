from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import error, request

from src.config.settings import Settings, settings
from src.ingestion.extraction import Tier1TextExtractor
from src.ingestion.parsing import (
    LayoutParsingError,
    ParsedBlock,
    ParsedMarkdownDocument,
)


class GoogleParserBackend(Protocol):
    def convert(self, file_path: Path, mime_type: str) -> str: ...


class GoogleParserError(LayoutParsingError):
    pass


@dataclass(frozen=True)
class _HttpResponse:
    status_code: int
    payload: dict[str, Any]


class Tier2GoogleParser:
    def __init__(
        self,
        *,
        cfg: Settings = settings,
        backend: GoogleParserBackend | None = None,
    ) -> None:
        self.cfg = cfg
        self._backend = backend

    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
        if not document_id:
            raise GoogleParserError("document_id is required")

        mime_type = (
            mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        )
        markdown = self._backend_or_default().convert(
            file_path=file_path, mime_type=mime_type
        )
        normalized = markdown.strip()
        if not normalized:
            raise GoogleParserError("Google parser returned empty markdown output")

        block_id = _make_block_id(document_id=document_id, markdown=normalized)
        block = ParsedBlock(
            block_id=block_id,
            block_type="paragraph",
            markdown=normalized,
            section_path=(file_path.name,),
            page_number=1,
        )
        return ParsedMarkdownDocument(
            document_id=document_id,
            source_path=file_path,
            markdown=normalized,
            blocks=[block],
            parser_name="google",
        )

    def _backend_or_default(self) -> GoogleParserBackend:
        if self._backend is None:
            self._backend = _GeminiMarkdownBackend(self.cfg)
        return self._backend


class _GeminiMarkdownBackend:
    def __init__(
        self, cfg: Settings, extractor: Tier1TextExtractor | None = None
    ) -> None:
        self._cfg = cfg
        self._extractor = extractor or Tier1TextExtractor()
        self._api_key = (cfg.cloud_agent_api_key or "").strip()
        self._base_url = cfg.cloud_agent_api_base_url.rstrip("/")
        self._model = cfg.cloud_agent_model

    def convert(self, file_path: Path, mime_type: str) -> str:
        if not self._api_key:
            raise GoogleParserError(
                "Google parser requires cloud_agent_api_key to call Gemini"
            )

        extraction = self._extractor.extract(file_path=file_path, mime_type=mime_type)
        source_text = extraction.text.strip()
        if not source_text:
            raise GoogleParserError("Google parser cannot process empty extracted text")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Convert the following document text into normalized markdown. "
                                "Preserve headings, tables, and bullet points when present. "
                                "Do not include code fences.\n\n"
                                f"Document:\n{source_text}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
            },
        }

        response = _default_transport(
            url=(
                f"{self._base_url}/v1beta/models/{self._model}:generateContent"
                f"?key={self._api_key}"
            ),
            payload=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout_seconds=float(self._cfg.cloud_agent_timeout_seconds),
        )

        if response.status_code < 200 or response.status_code >= 300:
            raise GoogleParserError(
                f"Google parser request failed with status {response.status_code}"
            )

        markdown = _candidate_text(response.payload)
        if not markdown:
            raise GoogleParserError(
                "Google parser response did not include markdown text"
            )
        return markdown


def _default_transport(
    *,
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
                raise GoogleParserError("Google parser returned non-object JSON")
            return _HttpResponse(
                status_code=int(getattr(response, "status", 200)),
                payload=parsed,
            )
    except error.HTTPError as exc:
        raise GoogleParserError(f"Google parser HTTP error: {exc.code}") from exc
    except error.URLError as exc:
        raise GoogleParserError(f"Google parser transport error: {exc.reason}") from exc


def _candidate_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return ""

    content = candidate.get("content")
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    return "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()


def _make_block_id(*, document_id: str, markdown: str) -> str:
    digest = sha256(f"{document_id}|google|{markdown}".encode("utf-8")).hexdigest()[:16]
    return f"gblk_{digest}"
