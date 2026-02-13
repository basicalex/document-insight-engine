from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from src.ingestion.parsing import ParsedBlock, ParsedMarkdownDocument


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class ParentChunk:
    chunk_id: str
    document_id: str
    order: int
    text: str
    token_count: int
    block_ids: list[str]
    section_path: tuple[str, ...]
    page_refs: list[int]


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    document_id: str
    parent_chunk_id: str
    order: int
    text: str
    token_count: int
    section_path: tuple[str, ...]
    page_refs: list[int]
    block_ids: list[str]


@dataclass(frozen=True)
class ChunkingResult:
    parent_chunks: list[ParentChunk]
    child_chunks: list[ChildChunk]


class ParentChildChunker:
    def __init__(
        self,
        parent_token_target: int = 1024,
        child_token_target: int = 256,
        child_overlap_tokens: int = 32,
    ) -> None:
        if parent_token_target <= 0 or child_token_target <= 0:
            raise ValueError("chunk token targets must be positive")
        if child_overlap_tokens < 0:
            raise ValueError("child overlap cannot be negative")
        if child_overlap_tokens >= child_token_target:
            raise ValueError("child overlap must be smaller than child size")

        self.parent_token_target = parent_token_target
        self.child_token_target = child_token_target
        self.child_overlap_tokens = child_overlap_tokens

    def chunk_document(self, parsed: ParsedMarkdownDocument) -> ChunkingResult:
        parent_groups = _group_blocks_into_parents(
            blocks=parsed.blocks,
            token_limit=self.parent_token_target,
        )

        parent_chunks = [
            _build_parent_chunk(document_id=parsed.document_id, order=idx, blocks=group)
            for idx, group in enumerate(parent_groups, start=1)
        ]

        child_chunks: list[ChildChunk] = []
        for parent_index, group in enumerate(parent_groups, start=1):
            parent = parent_chunks[parent_index - 1]
            child_chunks.extend(
                _build_child_chunks(
                    document_id=parsed.document_id,
                    parent_chunk=parent,
                    parent_blocks=group,
                    child_token_target=self.child_token_target,
                    child_overlap_tokens=self.child_overlap_tokens,
                )
            )

        return ChunkingResult(parent_chunks=parent_chunks, child_chunks=child_chunks)


def _group_blocks_into_parents(
    blocks: list[ParsedBlock],
    token_limit: int,
) -> list[list[ParsedBlock]]:
    if not blocks:
        return []

    groups: list[list[ParsedBlock]] = []
    current_group: list[ParsedBlock] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = _count_tokens(block.markdown)

        if current_group and current_tokens + block_tokens > token_limit:
            groups.append(current_group)
            current_group = []
            current_tokens = 0

        current_group.append(block)
        current_tokens += block_tokens

    if current_group:
        groups.append(current_group)

    return groups


def _build_parent_chunk(
    document_id: str, order: int, blocks: list[ParsedBlock]
) -> ParentChunk:
    text = _join_block_text(blocks)
    token_count = _count_tokens(text)
    block_ids = [block.block_id for block in blocks]
    section_path = blocks[-1].section_path if blocks else ()
    page_refs = _collect_page_refs(blocks)
    chunk_id = _chunk_id(prefix="par", document_id=document_id, order=order, text=text)

    return ParentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        order=order,
        text=text,
        token_count=token_count,
        block_ids=block_ids,
        section_path=section_path,
        page_refs=page_refs,
    )


def _build_child_chunks(
    document_id: str,
    parent_chunk: ParentChunk,
    parent_blocks: list[ParsedBlock],
    child_token_target: int,
    child_overlap_tokens: int,
) -> list[ChildChunk]:
    if not parent_blocks:
        return []

    if all(block.block_type == "table" for block in parent_blocks):
        return [
            _create_child_chunk(
                document_id=document_id,
                parent_chunk=parent_chunk,
                order=1,
                text=_join_block_text(parent_blocks),
                section_path=parent_blocks[-1].section_path,
                page_refs=_collect_page_refs(parent_blocks),
                block_ids=[block.block_id for block in parent_blocks],
            )
        ]

    full_text = _join_block_text(parent_blocks)
    tokens = _tokenize(full_text)
    block_token_spans = _block_token_spans(parent_blocks)
    if len(tokens) <= child_token_target:
        return [
            _create_child_chunk(
                document_id=document_id,
                parent_chunk=parent_chunk,
                order=1,
                text=full_text,
                section_path=parent_blocks[-1].section_path,
                page_refs=_page_refs_for_window(
                    block_spans=block_token_spans,
                    start=0,
                    end=len(tokens),
                )
                or _collect_page_refs(parent_blocks),
                block_ids=[block.block_id for block in parent_blocks],
            )
        ]

    stride = child_token_target - child_overlap_tokens
    children: list[ChildChunk] = []
    order = 1
    start = 0

    while start < len(tokens):
        end = min(start + child_token_target, len(tokens))
        child_tokens = tokens[start:end]
        child_text = " ".join(child_tokens)
        children.append(
            _create_child_chunk(
                document_id=document_id,
                parent_chunk=parent_chunk,
                order=order,
                text=child_text,
                section_path=parent_blocks[-1].section_path,
                page_refs=_page_refs_for_window(
                    block_spans=block_token_spans,
                    start=start,
                    end=end,
                )
                or _collect_page_refs(parent_blocks),
                block_ids=[block.block_id for block in parent_blocks],
            )
        )

        if end == len(tokens):
            break

        order += 1
        start += stride

    return children


def _create_child_chunk(
    document_id: str,
    parent_chunk: ParentChunk,
    order: int,
    text: str,
    section_path: tuple[str, ...],
    page_refs: list[int],
    block_ids: list[str],
) -> ChildChunk:
    chunk_id = _chunk_id(
        prefix="chd",
        document_id=document_id,
        order=order,
        text=f"{parent_chunk.chunk_id}|{text}",
    )
    return ChildChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_chunk_id=parent_chunk.chunk_id,
        order=order,
        text=text,
        token_count=_count_tokens(text),
        section_path=section_path,
        page_refs=page_refs,
        block_ids=block_ids,
    )


def _collect_page_refs(blocks: list[ParsedBlock]) -> list[int]:
    refs = {block.page_number for block in blocks if block.page_number is not None}
    return sorted(refs)


def _block_token_spans(blocks: list[ParsedBlock]) -> list[tuple[int, int, int | None]]:
    spans: list[tuple[int, int, int | None]] = []
    cursor = 0
    for block in blocks:
        count = _count_tokens(block.markdown)
        start = cursor
        end = cursor + count
        spans.append((start, end, block.page_number))
        cursor = end
    return spans


def _page_refs_for_window(
    *, block_spans: list[tuple[int, int, int | None]], start: int, end: int
) -> list[int]:
    if end <= start:
        return []

    coverage_by_page: dict[int, int] = {}
    for block_start, block_end, page_number in block_spans:
        if page_number is None:
            continue
        overlap = max(0, min(end, block_end) - max(start, block_start))
        if overlap <= 0:
            continue
        coverage_by_page[page_number] = coverage_by_page.get(page_number, 0) + overlap

    if not coverage_by_page:
        return []

    ordered_pages = sorted(
        coverage_by_page.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [page for page, _ in ordered_pages]


def _chunk_id(prefix: str, document_id: str, order: int, text: str) -> str:
    digest = sha256(f"{document_id}|{order}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{order:04d}_{digest}"


def _join_block_text(blocks: list[ParsedBlock]) -> str:
    return "\n\n".join(block.markdown for block in blocks)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _count_tokens(text: str) -> int:
    return len(_tokenize(text))
