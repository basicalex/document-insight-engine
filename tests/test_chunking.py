from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import ParentChildChunker
from src.ingestion.parsing import ParsedBlock, ParsedMarkdownDocument


def _parsed_doc(*blocks: ParsedBlock) -> ParsedMarkdownDocument:
    return ParsedMarkdownDocument(
        document_id="doc-1",
        source_path=Path("invoice.md"),
        markdown="\n\n".join(block.markdown for block in blocks),
        blocks=list(blocks),
    )


def test_chunker_enforces_parent_boundaries_and_parent_child_linkage() -> None:
    chunker = ParentChildChunker(
        parent_token_target=8,
        child_token_target=4,
        child_overlap_tokens=1,
    )
    parsed = _parsed_doc(
        ParsedBlock("b1", "paragraph", "one two three four", ("A",), 1),
        ParsedBlock("b2", "paragraph", "five six", ("A",), 1),
        ParsedBlock("b3", "paragraph", "seven eight nine", ("B",), 2),
    )

    result = chunker.chunk_document(parsed)

    assert len(result.parent_chunks) == 2
    assert result.parent_chunks[0].block_ids == ["b1", "b2"]
    assert result.parent_chunks[1].block_ids == ["b3"]

    for child in result.child_chunks:
        assert child.parent_chunk_id in {
            parent.chunk_id for parent in result.parent_chunks
        }
        assert child.block_ids


def test_chunker_preserves_order_and_lineage_metadata() -> None:
    chunker = ParentChildChunker(
        parent_token_target=20, child_token_target=5, child_overlap_tokens=1
    )
    parsed = _parsed_doc(
        ParsedBlock("b1", "heading", "# Header", ("Header",), 1),
        ParsedBlock(
            "b2", "paragraph", "alpha beta gamma delta epsilon zeta", ("Header",), 2
        ),
    )

    result = chunker.chunk_document(parsed)

    assert [parent.order for parent in result.parent_chunks] == [1]
    assert all(child.order >= 1 for child in result.child_chunks)
    assert all(child.section_path == ("Header",) for child in result.child_chunks)
    assert all(child.page_refs == [1, 2] for child in result.child_chunks)


def test_chunker_keeps_table_blocks_atomic_for_children() -> None:
    chunker = ParentChildChunker(
        parent_token_target=100, child_token_target=5, child_overlap_tokens=1
    )
    large_table = "\n".join(
        [
            "| item | amount |",
            "| --- | ---: |",
            "| service fee | 100 |",
            "| tax | 10 |",
            "| subtotal | 110 |",
            "| discount | 5 |",
            "| total | 105 |",
        ]
    )
    parsed = _parsed_doc(ParsedBlock("tb1", "table", large_table, ("Invoice",), 3))

    result = chunker.chunk_document(parsed)

    assert len(result.parent_chunks) == 1
    assert len(result.child_chunks) == 1
    assert result.child_chunks[0].text == large_table
    assert result.child_chunks[0].block_ids == ["tb1"]
