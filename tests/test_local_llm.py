from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.settings import Settings
from src.engine.local_llm import LocalQAEngine, RetrievedEvidence, build_rag_prompt
from src.ingestion.indexing import QueryMatch
from src.models.schemas import Mode


class StubStore:
    def __init__(self, matches: list[QueryMatch]) -> None:
        self.matches = matches
        self.last_filters: dict[str, Any] | None = None

    def query(
        self,
        tier: object,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryMatch]:
        self.last_filters = filters
        return self.matches[:top_k]


class StubEmbedder:
    def embed_query(self, text: str, dimension: int) -> list[float]:
        return [0.1 for _ in range(dimension)]


@dataclass
class StubOllama:
    answer: str
    prompt_seen: str = ""

    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str:
        self.prompt_seen = prompt
        return self.answer


def test_build_rag_prompt_includes_question_and_chunk_context() -> None:
    prompt = build_rag_prompt(
        question="What is the total due?",
        mode=Mode.FAST,
        prompt_version="local-rag-v1",
        evidence=[
            RetrievedEvidence(
                chunk_id="chunk-42",
                text="Total Due: 1234 USD",
                page_refs=[1],
                score=0.8,
            )
        ],
    )

    assert "PROMPT_VERSION=local-rag-v1" in prompt
    assert "Question:" in prompt
    assert "chunk-42" in prompt
    assert "INSUFFICIENT_EVIDENCE" in prompt


def test_local_engine_returns_insufficient_evidence_when_no_chunks_found() -> None:
    engine = LocalQAEngine(
        index_store=StubStore(matches=[]),
        cfg=Settings(),
        embedder=StubEmbedder(),
        ollama_client=StubOllama(answer="unused"),
    )

    response = engine.ask(
        question="What is the renewal window?",
        mode=Mode.FAST,
        document_id="doc-empty",
    )

    assert response.insufficient_evidence is True
    assert response.trace is not None
    assert response.trace.prompt_version == "local-rag-v1"
    assert response.trace.retrieved_chunk_ids == []
    assert response.trace.termination_reason == "insufficient_evidence"


def test_local_engine_returns_grounded_response_with_trace_and_citations() -> None:
    store = StubStore(
        matches=[
            QueryMatch(
                record_id="rec-1",
                score=0.95,
                payload={
                    "chunk_id": "chunk-42",
                    "text": "Invoice states: Total due is 1234 USD.",
                    "page_refs": ["1"],
                    "document_id": "doc-123",
                },
            )
        ]
    )
    ollama = StubOllama(answer="Total due is 1234 USD.")
    engine = LocalQAEngine(
        index_store=store,
        cfg=Settings(),
        embedder=StubEmbedder(),
        ollama_client=ollama,
    )

    response = engine.ask(
        question="What is the total due?",
        mode=Mode.FAST,
        document_id="doc-123",
    )

    assert response.insufficient_evidence is False
    assert response.answer == "Total due is 1234 USD."
    assert response.citations[0].chunk_id == "chunk-42"
    assert response.trace is not None
    assert response.trace.retrieved_chunk_ids == ["chunk-42"]
    assert response.trace.prompt_version == "local-rag-v1"
    assert response.trace.termination_reason == "completed"
    assert store.last_filters == {"document_id": "doc-123"}
    assert "chunk-42" in ollama.prompt_seen
