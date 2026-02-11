from __future__ import annotations

from pathlib import Path

from src.ingestion.parsing import (
    DoclingBackend,
    DoclingBlockCandidate,
    DoclingPayload,
    Tier2DoclingParser,
)


class StubDoclingBackend(DoclingBackend):
    def __init__(self, payload: DoclingPayload) -> None:
        self.payload = payload

    def convert(self, file_path: Path) -> DoclingPayload:
        return self.payload


def test_docling_parser_keeps_markdown_table_atomic() -> None:
    payload = DoclingPayload(
        markdown=(
            "# Invoice\n\n"
            "Summary before table.\n\n"
            "| Item | Amount |\n"
            "| --- | ---: |\n"
            "| Service Fee | 120.00 |\n"
            "| Tax | 12.00 |\n\n"
            "After table paragraph."
        )
    )
    parser = Tier2DoclingParser(backend=StubDoclingBackend(payload))

    parsed = parser.parse(document_id="doc-1", file_path=Path("invoice.pdf"))

    table_blocks = [block for block in parsed.blocks if block.block_type == "table"]
    assert len(table_blocks) == 1
    assert "| Item | Amount |" in table_blocks[0].markdown
    assert "| Tax | 12.00 |" in table_blocks[0].markdown


def test_docling_parser_tracks_section_paths_from_headings() -> None:
    payload = DoclingPayload(
        markdown=(
            "# Policy\n\n"
            "Intro text.\n\n"
            "## Coverage\n"
            "Coverage paragraph.\n\n"
            "### Limits\n"
            "Limits paragraph."
        )
    )
    parser = Tier2DoclingParser(backend=StubDoclingBackend(payload))

    parsed = parser.parse(document_id="doc-2", file_path=Path("policy.pdf"))

    heading_blocks = [block for block in parsed.blocks if block.block_type == "heading"]
    paragraph_blocks = [
        block for block in parsed.blocks if block.block_type == "paragraph"
    ]

    assert heading_blocks[0].section_path == ("Policy",)
    assert heading_blocks[1].section_path == ("Policy", "Coverage")
    assert heading_blocks[2].section_path == ("Policy", "Coverage", "Limits")
    assert paragraph_blocks[-1].section_path == ("Policy", "Coverage", "Limits")


def test_docling_parser_produces_stable_block_ids() -> None:
    payload = DoclingPayload(markdown="# Contract\n\nPayment due in 30 days.")
    parser = Tier2DoclingParser(backend=StubDoclingBackend(payload))

    first = parser.parse(document_id="doc-3", file_path=Path("contract.pdf"))
    second = parser.parse(document_id="doc-3", file_path=Path("contract.pdf"))

    assert [block.block_id for block in first.blocks] == [
        block.block_id for block in second.blocks
    ]


def test_docling_parser_uses_backend_blocks_when_available() -> None:
    payload = DoclingPayload(
        markdown="# Any",
        blocks=[
            DoclingBlockCandidate(
                block_type="table",
                markdown="| A |\n| - |\n| 1 |",
                section_path=("Sheet",),
                page_number=4,
            )
        ],
    )
    parser = Tier2DoclingParser(backend=StubDoclingBackend(payload))

    parsed = parser.parse(document_id="doc-4", file_path=Path("sheet.pdf"))

    assert len(parsed.blocks) == 1
    assert parsed.blocks[0].block_type == "table"
    assert parsed.blocks[0].section_path == ("Sheet",)
    assert parsed.blocks[0].page_number == 4
