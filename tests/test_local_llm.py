from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterator
from urllib import error as urllib_error

import pytest

from src.config.settings import Settings
from src.engine.local_llm import (
    LocalQAEngine,
    OllamaGenerateError,
    OllamaHTTPClient,
    QueryEmbeddingCandidate,
    RetrievedEvidence,
    _format_gemini_http_error,
    _local_generation_failure_answer,
    build_rag_prompt,
)
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


class CandidateEmbedder:
    def embed_query_candidates(
        self,
        *,
        text: str,
        dimension: int,
    ) -> list[QueryEmbeddingCandidate]:
        del text
        return [
            QueryEmbeddingCandidate(
                vector=[0.1 for _ in range(dimension)],
                embedding_provider="ollama",
                embedding_model="all-minilm",
                embedding_version="ollama:all-minilm:v1",
            ),
            QueryEmbeddingCandidate(
                vector=[0.2 for _ in range(dimension)],
                embedding_provider="hash",
                embedding_model="hash:all-minilm",
                embedding_version="hash-v1",
            ),
        ]


class SelectiveStore:
    def __init__(self) -> None:
        self.filter_calls: list[dict[str, Any] | None] = []

    def query(
        self,
        tier: object,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryMatch]:
        del tier, vector, top_k
        self.filter_calls.append(filters)
        if filters and filters.get("embedding_provider") == "hash":
            return [
                QueryMatch(
                    record_id="rec-hash",
                    score=0.9,
                    payload={
                        "chunk_id": "chunk-hash",
                        "text": "Legacy hash embedding chunk.",
                        "page_refs": ["1"],
                    },
                )
            ]
        return []


@dataclass
class StubOllama:
    answer: str
    prompt_seen: str = ""

    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str:
        self.prompt_seen = prompt
        return self.answer

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
    ) -> Iterator[str]:
        self.prompt_seen = prompt
        yield self.answer


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


def test_build_rag_prompt_includes_recent_chat_history() -> None:
    prompt = build_rag_prompt(
        question="And what is the due date?",
        mode=Mode.FAST,
        prompt_version="local-rag-v1",
        evidence=[
            RetrievedEvidence(
                chunk_id="chunk-10",
                text="Due date is 2026-01-30.",
                page_refs=[1],
                score=0.9,
            )
        ],
        chat_history=[
            {
                "question": "What is this document?",
                "answer": "It is an invoice.",
            }
        ],
    )

    assert "Conversation history" in prompt
    assert "User: What is this document?" in prompt
    assert "Assistant: It is an invoice." in prompt


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
    assert response.citations[0].start_offset == 0
    assert response.citations[0].end_offset is not None
    assert response.trace is not None
    assert response.trace.retrieved_chunk_ids == ["chunk-42"]
    assert response.trace.prompt_version == "local-rag-v1"
    assert response.trace.termination_reason == "completed"
    assert store.last_filters == {"document_id": "doc-123"}
    assert "chunk-42" in ollama.prompt_seen


def test_local_engine_parses_json_encoded_page_refs_for_citations() -> None:
    store = StubStore(
        matches=[
            QueryMatch(
                record_id="rec-1",
                score=0.95,
                payload={
                    "chunk_id": "chunk-42",
                    "text": "Invoice states: Total due is 1234 USD.",
                    "page_refs": '["3", "4"]',
                    "document_id": "doc-123",
                },
            )
        ]
    )
    engine = LocalQAEngine(
        index_store=store,
        cfg=Settings(),
        embedder=StubEmbedder(),
        ollama_client=StubOllama(answer="Total due is 1234 USD."),
    )

    response = engine.ask(
        question="What is the total due?",
        mode=Mode.FAST,
        document_id="doc-123",
    )

    assert response.citations[0].page == 3


def test_local_engine_stream_events_emit_tokens_and_final_payload() -> None:
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
    engine = LocalQAEngine(
        index_store=store,
        cfg=Settings(),
        embedder=StubEmbedder(),
        ollama_client=StubOllama(answer="Total due is 1234 USD."),
    )

    events = list(
        engine.ask_stream_events(
            question="What is the total due?",
            mode=Mode.FAST,
            document_id="doc-123",
        )
    )

    assert len(events) == 4
    assert events[0]["type"] == "status"
    assert events[1]["type"] == "status"
    assert events[2]["type"] == "token"
    assert events[2]["delta"] == "Total due is 1234 USD."
    assert events[3]["type"] == "final"
    assert events[3]["response"]["answer"] == "Total due is 1234 USD."


def test_local_engine_query_candidate_fallback_supports_migration_path() -> None:
    store = SelectiveStore()
    engine = LocalQAEngine(
        index_store=store,
        cfg=Settings(embedding_filter_strict=True),
        embedder=CandidateEmbedder(),
        ollama_client=StubOllama(answer="Found in legacy index."),
    )

    response = engine.ask(
        question="What is the legacy answer?",
        mode=Mode.FAST,
        document_id="doc-legacy",
    )

    assert response.insufficient_evidence is False
    assert response.answer == "Found in legacy index."
    assert store.filter_calls[0] is not None
    assert store.filter_calls[0]["embedding_provider"] == "ollama"
    assert store.filter_calls[1] is not None
    assert store.filter_calls[1]["embedding_provider"] == "hash"


def test_local_generation_failure_message_guides_model_pull_for_missing_model() -> None:
    answer = _local_generation_failure_answer(
        model="llama3.2:1b",
        error=OllamaGenerateError(
            "Ollama request failed with status 404: model 'llama3.2:1b' not found"
        ),
    )

    assert 'ollama pull "llama3.2:1b"' in answer


def test_local_generation_failure_message_guides_backend_switch_on_api_rate_limit() -> None:
    answer = _local_generation_failure_answer(
        model="gemini-2.5-flash",
        error=OllamaGenerateError("Gemini request failed with status 429"),
    )

    assert "rate-limited" in answer.lower()
    assert "local/auto" in answer.lower()


def test_format_gemini_http_error_surfaces_provider_message() -> None:
    payload = (
        b'{"error":{"code":429,"message":"Quota exceeded for metric: '
        b'generate_content_free_tier_requests"}}'
    )
    exc = urllib_error.HTTPError(
        url="https://example.test",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=io.BytesIO(payload),
    )

    text = _format_gemini_http_error(exc)

    assert "Gemini request failed with status 429" in text
    assert "Quota exceeded for metric" in text


class KeywordAwareStore:
    def __init__(self, matches: list[QueryMatch]) -> None:
        self.matches = matches
        self.top_k_calls: list[int] = []

    def query(
        self,
        tier: object,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryMatch]:
        del tier, vector, filters
        self.top_k_calls.append(top_k)
        return self.matches[:top_k]


def test_local_engine_keyword_queries_expand_retrieval_and_promote_keyword_hits() -> None:
    matches = [
        QueryMatch(
            record_id=f"rec-{index}",
            score=0.9,
            payload={
                "chunk_id": f"chunk-{index}",
                "text": f"generic context block {index}",
                "page_refs": ["1"],
                "document_id": "doc-plant",
            },
        )
        for index in range(1, 9)
    ]
    matches.append(
        QueryMatch(
            record_id="rec-alex",
            score=0.6,
            payload={
                "chunk_id": "chunk-alex",
                "text": "Alex combines both approaches for dual emotional processing.",
                "page_refs": ["85"],
                "document_id": "doc-plant",
            },
        )
    )

    store = KeywordAwareStore(matches=matches)
    engine = LocalQAEngine(
        index_store=store,
        cfg=Settings(),
        embedder=StubEmbedder(),
        ollama_client=StubOllama(answer="Alex is an agent variant in the plant study."),
    )

    response = engine.ask(
        question="who is alex",
        mode=Mode.FAST,
        document_id="doc-plant",
    )

    assert store.top_k_calls
    assert store.top_k_calls[0] > engine.retrieval_top_k
    assert response.insufficient_evidence is False
    assert response.citations
    assert response.citations[0].chunk_id == "chunk-alex"


def test_ollama_http_client_generate_wraps_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OllamaHTTPClient("http://localhost:11434")

    def _raise_timeout(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise TimeoutError("timed out")

    monkeypatch.setattr("src.engine.local_llm.urllib_request.urlopen", _raise_timeout)

    with pytest.raises(OllamaGenerateError, match="timed out"):
        client.generate(
            model="nomic-embed-text:v1.5",
            prompt="hello",
            timeout_seconds=1,
        )
