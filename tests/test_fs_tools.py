from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings
from src.tools.fs_tools import MarkdownFSTools, get_fs_tools, load_markdown_scope


_MARKDOWN = """# Master Service Agreement
Welcome text.

## Payment Terms
Invoice due within 30 days.

## Payment Terms
Second payment block.

### Late Fees
A late fee of 2% applies when overdue.
"""


def test_list_sections_returns_normalized_keys_and_spans() -> None:
    tools = MarkdownFSTools(document_id="doc-1", markdown_text=_MARKDOWN)

    response = tools.list_sections()

    assert response["ok"] is True
    keys = [section["key"] for section in response["sections"]]
    assert keys == [
        "master-service-agreement",
        "master-service-agreement/payment-terms",
        "master-service-agreement/payment-terms-2",
        "master-service-agreement/payment-terms-2/late-fees",
    ]
    first = response["sections"][0]
    assert first["line_start"] == 1
    assert first["char_start"] == 0


def test_read_section_is_bounded_and_reports_missing_section() -> None:
    tools = MarkdownFSTools(document_id="doc-2", markdown_text=_MARKDOWN)

    ok_response = tools.read_section(
        "master-service-agreement/payment-terms-2/late-fees",
        max_chars=18,
    )
    err_response = tools.read_section("master-service-agreement/does-not-exist")

    assert ok_response["ok"] is True
    assert ok_response["truncated"] is True
    assert len(ok_response["content"]) == 18
    assert err_response["ok"] is False
    assert err_response["error"]["code"] == "section_not_found"


def test_keyword_grep_returns_line_spans_and_no_match_error() -> None:
    tools = MarkdownFSTools(document_id="doc-3", markdown_text=_MARKDOWN)

    hit = tools.keyword_grep(
        keyword="late fee",
        section_key="master-service-agreement/payment-terms-2/late-fees",
    )
    miss = tools.keyword_grep(keyword="arbitration")

    assert hit["ok"] is True
    assert hit["total_matches"] >= 1
    match = hit["matches"][0]
    assert match["line_start"] >= 1
    assert match["char_end"] > match["char_start"]
    assert "late fee" in match["snippet"].lower()
    assert miss["ok"] is False
    assert miss["error"]["code"] == "no_matches"


def test_get_fs_tools_supports_all_documents_scope(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path)
    cfg.ensure_runtime_dirs()

    (cfg.parsed_dir / "doc-1.md").write_text(
        "# First Document\n\nContent about plants.",
        encoding="utf-8",
    )
    (cfg.parsed_dir / "doc-2.md").write_text(
        "# Second Document\n\nContent about cognition.",
        encoding="utf-8",
    )

    tools = get_fs_tools("__all_documents__", cfg=cfg)

    sections = tools["list_sections"]()
    assert sections["ok"] is True
    assert sections["document_id"] == "all-documents"
    titles = [section["title"] for section in sections["sections"]]
    assert "First Document" in titles
    assert "Second Document" in titles


def test_load_markdown_scope_returns_document_text_for_single_and_all_scope(
    tmp_path: Path,
) -> None:
    cfg = Settings(data_dir=tmp_path)
    cfg.ensure_runtime_dirs()

    (cfg.parsed_dir / "doc-1.md").write_text(
        "# First Document\n\nContent about plants.",
        encoding="utf-8",
    )

    single_document_id, single_text = load_markdown_scope("doc-1", cfg=cfg)
    all_document_id, all_text = load_markdown_scope("__all_documents__", cfg=cfg)

    assert single_document_id == "doc-1"
    assert "Content about plants." in single_text
    assert all_document_id == "all-documents"
    assert "First Document" in all_text
