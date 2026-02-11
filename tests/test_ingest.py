from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pytest
from starlette.datastructures import Headers, UploadFile

from src.config.settings import Settings
from src.ingestion.uploads import UploadIntakeError, UploadIntakeService


def _upload(filename: str, content: bytes, mime_type: str) -> UploadFile:
    file_obj = SpooledTemporaryFile(max_size=max(1, len(content) + 1))
    file_obj.write(content)
    file_obj.seek(0)
    headers = Headers({"content-type": mime_type})
    return UploadFile(file=file_obj, filename=filename, headers=headers)


def test_ingest_accepts_allowed_mime_and_returns_contract(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path, max_upload_size_mb=1)
    service = UploadIntakeService(cfg)

    receipt = asyncio.run(
        service.save_upload(
            _upload("invoice.pdf", b"%PDF-1.4\ninvoice", "application/pdf")
        )
    )

    assert len(receipt.document_id) == 32
    assert receipt.file_path.exists()
    assert receipt.file_path.parent == tmp_path / "uploads"


def test_ingest_rejects_unsupported_mime_type(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path, max_upload_size_mb=1)
    service = UploadIntakeService(cfg)

    with pytest.raises(UploadIntakeError) as exc_info:
        asyncio.run(service.save_upload(_upload("notes.txt", b"hello", "text/plain")))

    assert exc_info.value.status_code == 415
    assert "unsupported MIME type" in exc_info.value.message


def test_ingest_sanitizes_filename_and_blocks_traversal(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path, max_upload_size_mb=1)
    service = UploadIntakeService(cfg)

    receipt = asyncio.run(
        service.save_upload(
            _upload(
                "../../secret/../invoice?:2026.pdf",
                b"%PDF-1.4\ninvoice",
                "application/pdf",
            )
        )
    )

    assert receipt.file_path.parent == tmp_path / "uploads"
    assert ".." not in receipt.file_path.name
    assert "/" not in receipt.file_path.name


def test_ingest_enforces_max_size_with_streaming_chunks(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path, max_upload_size_mb=1)
    service = UploadIntakeService(cfg)
    oversized_payload = b"a" * ((1024 * 1024) + 1)

    with pytest.raises(UploadIntakeError) as exc_info:
        asyncio.run(
            service.save_upload(
                _upload("large.pdf", oversized_payload, "application/pdf")
            )
        )

    assert exc_info.value.status_code == 413
    assert "exceeds 1MB limit" in exc_info.value.message
    assert list((tmp_path / "uploads").glob("*")) == []


def test_ingest_is_idempotent_for_same_file_content(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path, max_upload_size_mb=1)
    service = UploadIntakeService(cfg)
    payload = b"%PDF-1.4\nrepeated"

    first = asyncio.run(
        service.save_upload(_upload("same.pdf", payload, "application/pdf"))
    )
    second = asyncio.run(
        service.save_upload(_upload("same.pdf", payload, "application/pdf"))
    )

    assert first.document_id == second.document_id
    assert first.file_path == second.file_path
