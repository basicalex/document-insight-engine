from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.google_parser import GoogleParserError, Tier2GoogleParser


class StubGoogleBackend:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown

    def convert(self, file_path: Path, mime_type: str) -> str:
        del file_path, mime_type
        return self.markdown


def test_google_parser_adapter_returns_parsed_document(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"ignored")
    parser = Tier2GoogleParser(backend=StubGoogleBackend("# Heading\n\nBody"))

    parsed = parser.parse(document_id="doc-1", file_path=file_path)

    assert parsed.document_id == "doc-1"
    assert parsed.parser_name == "google"
    assert parsed.blocks


def test_google_parser_adapter_rejects_empty_markdown(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"ignored")
    parser = Tier2GoogleParser(backend=StubGoogleBackend("  \n  "))

    with pytest.raises(GoogleParserError, match="empty markdown"):
        parser.parse(document_id="doc-2", file_path=file_path)
