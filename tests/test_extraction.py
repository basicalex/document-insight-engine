from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.extraction import (
    PageText,
    TextExtractionError,
    Tier1TextExtractor,
)


class StubExtractor(Tier1TextExtractor):
    def __init__(self) -> None:
        self.used_ocr_fallback = False
        self.used_pdf_text = False
        self.used_image_ocr = False
        self._pdf_text_pages: list[PageText] = []
        self._pdf_ocr_pages: list[PageText] = []
        self._image_text: str = ""

    def _extract_pdf_text_layer(self, file_path: Path) -> list[PageText]:
        self.used_pdf_text = True
        return self._pdf_text_pages

    def _extract_pdf_ocr(self, file_path: Path) -> list[PageText]:
        self.used_ocr_fallback = True
        return self._pdf_ocr_pages

    def _extract_image_ocr(self, file_path: Path) -> str:
        self.used_image_ocr = True
        return self._image_text


def test_pdf_text_layer_extraction_keeps_order_and_skips_ocr() -> None:
    extractor = StubExtractor()
    extractor._pdf_text_pages = [
        PageText(page_number=2, text="Page 2", method="pymupdf"),
        PageText(page_number=1, text="Page 1", method="pymupdf"),
    ]

    result = extractor.extract(Path("dummy.pdf"), "application/pdf")

    assert extractor.used_pdf_text is True
    assert extractor.used_ocr_fallback is False
    assert result.method == "pymupdf"
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.text == "Page 1\n\nPage 2"


def test_pdf_ocr_fallback_runs_when_text_layer_is_empty() -> None:
    extractor = StubExtractor()
    extractor._pdf_text_pages = [
        PageText(page_number=1, text="   ", method="pymupdf"),
        PageText(page_number=2, text="", method="pymupdf"),
    ]
    extractor._pdf_ocr_pages = [
        PageText(page_number=1, text="OCR one", method="ocr"),
        PageText(page_number=2, text="OCR two", method="ocr"),
    ]

    result = extractor.extract(Path("scanned.pdf"), "application/pdf")

    assert extractor.used_ocr_fallback is True
    assert result.method == "ocr"
    assert result.text == "OCR one\n\nOCR two"
    assert all(page.method == "ocr" for page in result.pages)


def test_image_ingest_uses_ocr_path() -> None:
    extractor = StubExtractor()
    extractor._image_text = "Image OCR text"

    result = extractor.extract(Path("image.png"), "image/png")

    assert extractor.used_image_ocr is True
    assert result.method == "ocr"
    assert result.text == "Image OCR text"
    assert result.pages[0].page_number == 1


def test_unsupported_mime_type_raises_error() -> None:
    extractor = StubExtractor()

    with pytest.raises(TextExtractionError):
        extractor.extract(Path("file.docx"), "application/msword")
