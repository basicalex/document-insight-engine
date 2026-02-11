from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from src.config.settings import ALLOWED_UPLOAD_MIME_TYPES, Settings


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class UploadReceipt:
    document_id: str
    file_path: Path


class UploadIntakeError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class UploadIntakeService:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

    async def save_upload(self, upload: UploadFile) -> UploadReceipt:
        if not upload.filename:
            raise UploadIntakeError("invalid_filename", "filename is required", 400)

        mime_type = (upload.content_type or "").strip().lower()
        if mime_type not in ALLOWED_UPLOAD_MIME_TYPES:
            raise UploadIntakeError(
                "unsupported_mime_type",
                f"unsupported MIME type: {mime_type or 'unknown'}",
                415,
            )

        sanitized_filename = _sanitize_filename(upload.filename)
        if not sanitized_filename:
            raise UploadIntakeError(
                "invalid_filename", "filename could not be sanitized", 400
            )

        self.cfg.ensure_runtime_dirs()
        uploads_dir = self.cfg.uploads_dir.resolve()
        temp_path = uploads_dir / f".{uuid4().hex}.part"

        bytes_written = 0
        max_bytes = self.cfg.max_upload_size_mb * 1024 * 1024
        hasher = sha256()

        try:
            with temp_path.open("wb") as temp_file:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break

                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise UploadIntakeError(
                            "upload_too_large",
                            f"file exceeds {self.cfg.max_upload_size_mb}MB limit",
                            413,
                        )

                    hasher.update(chunk)
                    temp_file.write(chunk)
        except UploadIntakeError:
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if bytes_written == 0:
            temp_path.unlink(missing_ok=True)
            raise UploadIntakeError("empty_upload", "uploaded file is empty", 400)

        checksum = hasher.hexdigest()
        document_id = sha256(
            f"{sanitized_filename}:{checksum}".encode("utf-8")
        ).hexdigest()[:32]
        destination = uploads_dir / f"{document_id}_{sanitized_filename}"
        destination = _guarded_destination(
            destination=destination, uploads_dir=uploads_dir
        )

        if destination.exists():
            temp_path.unlink(missing_ok=True)
            return UploadReceipt(document_id=document_id, file_path=destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path.replace(destination)

        return UploadReceipt(document_id=document_id, file_path=destination)


def _sanitize_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    candidate = _FILENAME_SAFE_RE.sub("_", candidate)
    candidate = candidate.lstrip(".")
    if not candidate:
        return ""
    return candidate[:128]


def _guarded_destination(destination: Path, uploads_dir: Path) -> Path:
    resolved_destination = destination.resolve()
    resolved_uploads = uploads_dir.resolve()
    if not resolved_destination.is_relative_to(resolved_uploads):
        raise UploadIntakeError(
            "invalid_destination", "resolved path escaped uploads dir", 400
        )
    return resolved_destination
