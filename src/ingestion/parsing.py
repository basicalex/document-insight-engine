from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ParsedBlock:
    block_id: str
    block_type: str
    markdown: str
    section_path: tuple[str, ...]
    page_number: int | None


@dataclass(frozen=True)
class ParsedMarkdownDocument:
    document_id: str
    source_path: Path
    markdown: str
    blocks: list[ParsedBlock]


@dataclass(frozen=True)
class DoclingBlockCandidate:
    block_type: str
    markdown: str
    section_path: tuple[str, ...] = ()
    page_number: int | None = None


@dataclass(frozen=True)
class DoclingPayload:
    markdown: str
    blocks: list[DoclingBlockCandidate] | None = None


class DoclingBackend(Protocol):
    def convert(self, file_path: Path) -> DoclingPayload: ...


class LayoutParsingError(Exception):
    pass


class Tier2DoclingParser:
    def __init__(self, backend: DoclingBackend | None = None) -> None:
        self._backend = backend

    def parse(self, document_id: str, file_path: Path) -> ParsedMarkdownDocument:
        if not document_id:
            raise LayoutParsingError("document_id is required")

        payload = self._get_backend().convert(file_path)
        markdown = payload.markdown.strip()

        candidates = payload.blocks or _blocks_from_markdown(markdown)
        blocks = [
            ParsedBlock(
                block_id=_make_block_id(
                    document_id=document_id, index=index, candidate=candidate
                ),
                block_type=candidate.block_type,
                markdown=candidate.markdown,
                section_path=candidate.section_path,
                page_number=candidate.page_number,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]

        return ParsedMarkdownDocument(
            document_id=document_id,
            source_path=file_path,
            markdown=markdown,
            blocks=blocks,
        )

    def _get_backend(self) -> DoclingBackend:
        if self._backend is None:
            self._backend = _DoclingRuntimeBackend()
        return self._backend


class _DoclingRuntimeBackend:
    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ModuleNotFoundError as exc:
            raise LayoutParsingError(
                "Docling is required for layout parsing. Install the 'docling' package."
            ) from exc

        self._converter = DocumentConverter()

    def convert(self, file_path: Path) -> DoclingPayload:
        try:
            result = self._converter.convert(str(file_path))
        except Exception as exc:
            raise LayoutParsingError(f"Docling conversion failed: {exc}") from exc

        document = getattr(result, "document", result)
        markdown = _export_markdown(document)

        if not markdown:
            raise LayoutParsingError("Docling returned empty markdown output")

        return DoclingPayload(markdown=markdown)


def _export_markdown(document: object) -> str:
    for attr in ("export_to_markdown", "to_markdown", "export_markdown"):
        method = getattr(document, attr, None)
        if callable(method):
            output = method()
            if isinstance(output, str):
                return output

    if isinstance(document, str):
        return document

    raise LayoutParsingError("Docling output does not support markdown export")


def _blocks_from_markdown(markdown: str) -> list[DoclingBlockCandidate]:
    if not markdown:
        return []

    lines = markdown.splitlines()
    blocks: list[DoclingBlockCandidate] = []
    section_path: list[str] = []
    heading_levels: list[int] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip()

        if not line.strip():
            idx += 1
            continue

        if _is_heading(line):
            level = _heading_level(line)
            title = line[level + 1 :].strip()
            section_path, heading_levels = _apply_heading(
                section_path=section_path,
                heading_levels=heading_levels,
                level=level,
                title=title,
            )
            blocks.append(
                DoclingBlockCandidate(
                    block_type="heading",
                    markdown=line,
                    section_path=tuple(section_path),
                )
            )
            idx += 1
            continue

        if (
            _is_table_row(line)
            and (idx + 1) < len(lines)
            and _is_table_delimiter(lines[idx + 1])
        ):
            table_lines = [line, lines[idx + 1].rstrip()]
            idx += 2
            while idx < len(lines):
                next_line = lines[idx].rstrip()
                if not _is_table_row(next_line):
                    break
                table_lines.append(next_line)
                idx += 1

            blocks.append(
                DoclingBlockCandidate(
                    block_type="table",
                    markdown="\n".join(table_lines),
                    section_path=tuple(section_path),
                )
            )
            continue

        paragraph_lines = [line]
        idx += 1
        while idx < len(lines):
            next_line = lines[idx].rstrip()
            if not next_line.strip() or _is_heading(next_line):
                break
            if (
                _is_table_row(next_line)
                and (idx + 1) < len(lines)
                and _is_table_delimiter(lines[idx + 1])
            ):
                break
            paragraph_lines.append(next_line)
            idx += 1

        blocks.append(
            DoclingBlockCandidate(
                block_type="paragraph",
                markdown="\n".join(paragraph_lines),
                section_path=tuple(section_path),
            )
        )

    return blocks


def _is_heading(line: str) -> bool:
    return line.startswith("#") and len(line) > 1 and line[1] in {" ", "#"}


def _heading_level(line: str) -> int:
    level = 0
    while level < len(line) and line[level] == "#":
        level += 1
    return max(1, min(level, 6))


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2
    )


def _is_table_delimiter(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if not cells:
        return False
    return all(cell and set(cell) <= {":", "-"} for cell in cells)


def _apply_heading(
    section_path: list[str], heading_levels: list[int], level: int, title: str
) -> tuple[list[str], list[int]]:
    while heading_levels and heading_levels[-1] >= level:
        heading_levels.pop()
        section_path.pop()

    heading_levels.append(level)
    section_path.append(title)
    return section_path, heading_levels


def _make_block_id(
    document_id: str, index: int, candidate: DoclingBlockCandidate
) -> str:
    key = "|".join(
        [
            document_id,
            str(index),
            candidate.block_type,
            "/".join(candidate.section_path),
            candidate.markdown,
            str(candidate.page_number or ""),
        ]
    )
    digest = sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"blk_{index:04d}_{digest}"
