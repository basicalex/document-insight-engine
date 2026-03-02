from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import Settings, settings


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_KEY_SAFE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class _Section:
    key: str
    title: str
    level: int
    line_start: int
    line_end: int
    char_start: int
    char_end: int


class MarkdownFSTools:
    def __init__(
        self,
        document_id: str,
        markdown_text: str,
        max_section_chars: int = 3000,
        max_grep_matches: int = 20,
    ) -> None:
        if not document_id:
            raise ValueError("document_id is required")
        if max_section_chars <= 0:
            raise ValueError("max_section_chars must be positive")
        if max_grep_matches <= 0:
            raise ValueError("max_grep_matches must be positive")

        self.document_id = document_id
        self._markdown_text = markdown_text
        self.max_section_chars = max_section_chars
        self.max_grep_matches = max_grep_matches

        self._lines = markdown_text.splitlines()
        self._line_starts = _line_start_offsets(markdown_text)
        self._sections = _parse_sections(markdown_text)
        self._sections_by_key = {section.key: section for section in self._sections}

    @classmethod
    def from_file(
        cls,
        document_id: str,
        markdown_path: Path,
        max_section_chars: int = 3000,
        max_grep_matches: int = 20,
    ) -> "MarkdownFSTools":
        return cls(
            document_id=document_id,
            markdown_text=markdown_path.read_text(encoding="utf-8"),
            max_section_chars=max_section_chars,
            max_grep_matches=max_grep_matches,
        )

    def list_sections(self, limit: int = 200) -> dict[str, Any]:
        safe_limit = max(1, limit)
        selected = self._sections[:safe_limit]
        return {
            "ok": True,
            "document_id": self.document_id,
            "sections": [self._section_payload(section) for section in selected],
            "total_sections": len(self._sections),
            "truncated": len(self._sections) > safe_limit,
        }

    def read_section(
        self, section_key: str, max_chars: int | None = None
    ) -> dict[str, Any]:
        section = self._sections_by_key.get(section_key)
        if section is None:
            return self._error(
                ToolError(
                    code="section_not_found",
                    message=f"section '{section_key}' was not found",
                    details={
                        "section_key": section_key,
                        "available_section_keys": [
                            item.key
                            for item in self._sections[: min(20, len(self._sections))]
                        ],
                    },
                )
            )

        section_text = self._section_text(section)
        char_budget = max_chars if max_chars is not None else self.max_section_chars
        char_budget = max(1, char_budget)
        clipped = section_text[:char_budget]

        return {
            "ok": True,
            "document_id": self.document_id,
            "section": self._section_payload(section),
            "content": clipped,
            "truncated": len(section_text) > len(clipped),
            "returned_char_count": len(clipped),
        }

    def keyword_grep(
        self,
        keyword: str,
        section_key: str | None = None,
        max_matches: int | None = None,
        context_chars: int = 80,
    ) -> dict[str, Any]:
        search_term = (keyword or "").strip()
        if not search_term:
            return self._error(
                ToolError(
                    code="invalid_keyword",
                    message="keyword is required",
                    details={"keyword": keyword},
                )
            )

        search_section = None
        if section_key is not None:
            search_section = self._sections_by_key.get(section_key)
            if search_section is None:
                return self._error(
                    ToolError(
                        code="section_not_found",
                        message=f"section '{section_key}' was not found",
                        details={"section_key": section_key},
                    )
                )

        text_scope = (
            self._section_text(search_section)
            if search_section is not None
            else self._markdown_text
        )
        scope_offset = search_section.char_start if search_section is not None else 0

        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        bounded_matches = max(1, max_matches or self.max_grep_matches)

        results: list[dict[str, Any]] = []
        for match in pattern.finditer(text_scope):
            absolute_start = scope_offset + match.start()
            absolute_end = scope_offset + match.end()
            line_start = self._line_for_offset(absolute_start)
            line_end = self._line_for_offset(max(absolute_start, absolute_end - 1))

            snippet_start = max(0, match.start() - max(0, context_chars))
            snippet_end = min(len(text_scope), match.end() + max(0, context_chars))
            snippet = text_scope[snippet_start:snippet_end]

            result = {
                "match": match.group(0),
                "section_key": search_section.key
                if search_section is not None
                else None,
                "line_start": line_start,
                "line_end": line_end,
                "char_start": absolute_start,
                "char_end": absolute_end,
                "snippet": snippet,
            }
            results.append(result)
            if len(results) >= bounded_matches:
                break

        if not results:
            return self._error(
                ToolError(
                    code="no_matches",
                    message=f"no matches found for '{search_term}'",
                    details={
                        "keyword": search_term,
                        "section_key": section_key,
                    },
                )
            )

        total_matches = len(list(pattern.finditer(text_scope)))
        return {
            "ok": True,
            "document_id": self.document_id,
            "keyword": search_term,
            "section_key": section_key,
            "matches": results,
            "total_matches": total_matches,
            "truncated": total_matches > len(results),
        }

    def _section_payload(self, section: _Section) -> dict[str, Any]:
        return {
            "key": section.key,
            "title": section.title,
            "level": section.level,
            "line_start": section.line_start,
            "line_end": section.line_end,
            "char_start": section.char_start,
            "char_end": section.char_end,
        }

    def _section_text(self, section: _Section | None) -> str:
        if section is None:
            return self._markdown_text
        if not self._lines:
            return ""
        start_idx = max(0, section.line_start - 1)
        end_idx = max(start_idx, section.line_end)
        return "\n".join(self._lines[start_idx:end_idx])

    def _line_for_offset(self, char_offset: int) -> int:
        if not self._line_starts:
            return 1
        candidate = bisect_right(self._line_starts, max(0, char_offset)) - 1
        return max(1, min(candidate + 1, len(self._lines) if self._lines else 1))

    def _error(self, error: ToolError) -> dict[str, Any]:
        return {
            "ok": False,
            "document_id": self.document_id,
            "error": error.to_payload(),
        }


def load_markdown_scope(document_id: str, cfg: Settings = settings) -> tuple[str, str]:
    normalized_document_id = (document_id or "").strip()
    if _is_all_documents_scope(normalized_document_id):
        return "all-documents", _build_all_documents_markdown(cfg=cfg)

    markdown_path = _resolve_markdown_path(
        document_id=normalized_document_id,
        cfg=cfg,
    )
    markdown_text = markdown_path.read_text(encoding="utf-8")
    return normalized_document_id, markdown_text


def get_fs_tools(document_id: str, cfg: Settings = settings) -> dict[str, Any]:
    scoped_document_id, markdown_text = load_markdown_scope(document_id=document_id, cfg=cfg)
    tools = MarkdownFSTools(
        document_id=scoped_document_id,
        markdown_text=markdown_text,
    )

    return {
        "list_sections": tools.list_sections,
        "read_section": tools.read_section,
        "keyword_grep": tools.keyword_grep,
    }


def _is_all_documents_scope(document_id: str) -> bool:
    normalized = (document_id or "").strip().lower()
    return normalized in {"", "*", "all", "__all_documents__", "all-documents"}


def _build_all_documents_markdown(*, cfg: Settings) -> str:
    parsed_files = sorted(path for path in cfg.parsed_dir.glob("*.md") if path.is_file())
    if not parsed_files:
        raise FileNotFoundError(
            f"no parsed markdown artifacts found in {cfg.parsed_dir}"
        )

    document_sections: list[str] = []
    for parsed_file in parsed_files:
        content = parsed_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        doc_title = _parsed_artifact_title(parsed_file=parsed_file, content=content)
        doc_id = parsed_file.stem
        document_sections.append(
            f"## {doc_title} [{doc_id}]\n\n{content}"
        )

    if not document_sections:
        raise FileNotFoundError(
            f"parsed markdown artifacts were empty in {cfg.parsed_dir}"
        )

    return "# All indexed documents\n\n" + "\n\n---\n\n".join(document_sections)


def _parsed_artifact_title(*, parsed_file: Path, content: str) -> str:
    for line in content.splitlines()[:40]:
        heading = _HEADING_RE.match(line)
        if heading:
            return heading.group(2).strip()
    return parsed_file.stem


def _resolve_markdown_path(document_id: str, cfg: Settings) -> Path:
    candidates = [
        cfg.parsed_dir / f"{document_id}.md",
        cfg.parsed_dir / f"{document_id}.markdown",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in sorted(cfg.parsed_dir.glob(f"{document_id}_*.md")):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"no parsed markdown found for document_id '{document_id}' in {cfg.parsed_dir}"
    )


def _parse_sections(markdown_text: str) -> list[_Section]:
    lines = markdown_text.splitlines()
    if not lines:
        return []

    heading_rows: list[tuple[int, int, str, str]] = []
    stack: list[tuple[int, str]] = []
    key_counts: dict[str, int] = {}

    for line_idx, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if not match:
            continue

        level = len(match.group(1))
        title = match.group(2).strip()

        while stack and stack[-1][0] >= level:
            stack.pop()

        base_slug = _normalize_key_part(title)
        path_parts = [entry[1] for entry in stack] + [base_slug]
        joined_key = "/".join(path_parts)
        sequence = key_counts.get(joined_key, 0) + 1
        key_counts[joined_key] = sequence
        if sequence > 1:
            path_parts[-1] = f"{base_slug}-{sequence}"
        key = "/".join(path_parts)

        stack.append((level, path_parts[-1]))
        heading_rows.append((line_idx, level, title, key))

    if not heading_rows:
        return [
            _Section(
                key="document",
                title="Document",
                level=1,
                line_start=1,
                line_end=len(lines),
                char_start=0,
                char_end=len(markdown_text),
            )
        ]

    line_starts = _line_start_offsets(markdown_text)
    sections: list[_Section] = []
    for idx, (line_start, level, title, key) in enumerate(heading_rows):
        next_line = (
            heading_rows[idx + 1][0] if idx + 1 < len(heading_rows) else len(lines) + 1
        )
        line_end = next_line - 1
        char_start = line_starts[line_start - 1]
        char_end = (
            line_starts[line_end] if line_end < len(lines) else len(markdown_text)
        )
        sections.append(
            _Section(
                key=key,
                title=title,
                level=level,
                line_start=line_start,
                line_end=line_end,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return sections


def _line_start_offsets(markdown_text: str) -> list[int]:
    starts = [0]
    cursor = 0
    for line in markdown_text.splitlines(keepends=True):
        cursor += len(line)
        starts.append(cursor)
    return starts


def _normalize_key_part(title: str) -> str:
    lowered = title.lower().strip()
    slug = _KEY_SAFE_RE.sub("-", lowered).strip("-")
    return slug or "section"
