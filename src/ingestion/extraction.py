from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/tiff"}


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    method: str


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    method: str
    pages: list[PageText]


class TextExtractionError(Exception):
    pass


class Tier1TextExtractor:
    def extract(self, file_path: Path, mime_type: str) -> ExtractionResult:
        normalized_mime = (mime_type or "").strip().lower()
        if normalized_mime == "application/pdf":
            return self._extract_pdf(file_path)

        if normalized_mime in _IMAGE_MIME_TYPES:
            page = self._extract_image_ocr(file_path)
            pages = [PageText(page_number=1, text=page, method="ocr")]
            return ExtractionResult(text=page, method="ocr", pages=pages)

        raise TextExtractionError(
            f"Unsupported MIME type for extraction: {normalized_mime}"
        )

    def _extract_pdf(self, file_path: Path) -> ExtractionResult:
        text_pages = self._extract_pdf_text_layer(file_path)
        if any(page.text.strip() for page in text_pages):
            normalized_pages = _normalize_pages(text_pages, method="pymupdf")
            return ExtractionResult(
                text=_join_pages(normalized_pages),
                method="pymupdf",
                pages=normalized_pages,
            )

        ocr_pages = self._extract_pdf_ocr(file_path)
        normalized_pages = _normalize_pages(ocr_pages, method="ocr")
        return ExtractionResult(
            text=_join_pages(normalized_pages),
            method="ocr",
            pages=normalized_pages,
        )

    def _extract_pdf_text_layer(self, file_path: Path) -> list[PageText]:
        try:
            import pymupdf
        except ModuleNotFoundError as exc:
            raise TextExtractionError("PyMuPDF is required for PDF extraction") from exc

        pages: list[PageText] = []
        with pymupdf.open(file_path) as pdf_doc:
            for idx, page in enumerate(pdf_doc, start=1):
                text = page.get_text("text") or ""
                pages.append(PageText(page_number=idx, text=text, method="pymupdf"))
        return pages

    def _extract_pdf_ocr(self, file_path: Path) -> list[PageText]:
        try:
            import pytesseract
            from PIL import Image
            import pymupdf
        except ModuleNotFoundError as exc:
            raise TextExtractionError(
                "OCR dependencies are required for fallback extraction"
            ) from exc

        pages: list[PageText] = []
        with pymupdf.open(file_path) as pdf_doc:
            for idx, page in enumerate(pdf_doc, start=1):
                pixmap = page.get_pixmap()
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                text = pytesseract.image_to_string(image) or ""
                pages.append(PageText(page_number=idx, text=text, method="ocr"))
        return pages

    def _extract_image_ocr(self, file_path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise TextExtractionError(
                "OCR dependencies are required for image extraction"
            ) from exc

        image = Image.open(file_path)
        return pytesseract.image_to_string(image) or ""


def _normalize_pages(pages: list[PageText], method: str) -> list[PageText]:
    normalized = [
        PageText(page_number=page.page_number, text=page.text, method=method)
        for page in pages
    ]
    return sorted(normalized, key=lambda page: page.page_number)


def _join_pages(pages: list[PageText]) -> str:
    return "\n\n".join(page.text for page in pages)
