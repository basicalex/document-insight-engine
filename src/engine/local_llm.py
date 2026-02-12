from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.config.settings import Settings, settings
from src.ingestion.indexing import EmbeddingTier, QueryMatch
from src.ingestion.vectorize import hashing_vector
from src.models.schemas import AgentTrace, ChatResponse, Citation, Mode, TraceEvent


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class RetrievalStore(Protocol):
    def query(
        self,
        tier: EmbeddingTier,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryMatch]: ...


class QueryEmbedder(Protocol):
    def embed_query(self, text: str, dimension: int) -> list[float]: ...


class OllamaClient(Protocol):
    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str: ...


class OllamaGenerateError(Exception):
    pass


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    text: str
    page_refs: list[int]
    score: float


class HashingQueryEmbedder:
    def embed_query(self, text: str, dimension: int) -> list[float]:
        return hashing_vector(text=text, dimension=dimension)


class OllamaHTTPClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        endpoint = f"{self.base_url}/api/generate"
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            endpoint,
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise OllamaGenerateError(f"Ollama request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaGenerateError("Ollama returned invalid JSON") from exc

        text = str(decoded.get("response", "")).strip()
        if not text:
            raise OllamaGenerateError("Ollama returned an empty response")
        return text


class LocalQAEngine:
    def __init__(
        self,
        index_store: RetrievalStore,
        cfg: Settings = settings,
        embedder: QueryEmbedder | None = None,
        ollama_client: OllamaClient | None = None,
        prompt_version: str = "local-rag-v1",
        retrieval_top_k: int = 5,
        min_token_overlap: int = 1,
        query_vector_dimension: int = 384,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        if min_token_overlap < 0:
            raise ValueError("min_token_overlap cannot be negative")
        if query_vector_dimension <= 0:
            raise ValueError("query_vector_dimension must be positive")

        self.index_store = index_store
        self.cfg = cfg
        self.prompt_version = prompt_version
        self.retrieval_top_k = retrieval_top_k
        self.min_token_overlap = min_token_overlap
        self.query_vector_dimension = query_vector_dimension
        self.embedder = embedder or HashingQueryEmbedder()
        self.ollama_client = ollama_client or OllamaHTTPClient(cfg.ollama_base_url)

    def ask(
        self,
        question: str,
        mode: Mode,
        document_id: str | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()

        query_vector = self.embedder.embed_query(
            text=question,
            dimension=self.query_vector_dimension,
        )
        filters = {"document_id": document_id} if document_id else None
        raw_matches = self.index_store.query(
            tier=EmbeddingTier.TIER1,
            vector=query_vector,
            top_k=self.retrieval_top_k,
            filters=filters,
        )
        retrieval_latency_ms = _latency_ms(retrieval_started)

        evidence = [_to_evidence(match) for match in raw_matches]
        retrieved_chunk_ids = [item.chunk_id for item in evidence]
        citations = [_to_citation(item) for item in evidence]

        trace_events = [
            TraceEvent(
                stage="retrieval",
                message="retrieved evidence chunks",
                latency_ms=retrieval_latency_ms,
                metadata={
                    "document_id": document_id or "*",
                    "retrieved": str(len(evidence)),
                    "top_k": str(self.retrieval_top_k),
                },
            )
        ]

        if _is_insufficient_evidence(
            question=question,
            evidence=evidence,
            min_token_overlap=self.min_token_overlap,
        ):
            trace = AgentTrace(
                model=self.cfg.local_llm_model,
                prompt_version=self.prompt_version,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tool_calls=trace_events,
                termination_reason="insufficient_evidence",
                total_latency_ms=_latency_ms(started),
            )
            return ChatResponse(
                answer=(
                    "I do not have enough grounded evidence in the indexed document "
                    "to answer that question."
                ),
                mode=mode,
                document_id=document_id,
                insufficient_evidence=True,
                citations=citations,
                trace=trace,
            )

        prompt = build_rag_prompt(
            question=question,
            evidence=evidence,
            mode=mode,
            prompt_version=self.prompt_version,
        )
        generation_started = time.perf_counter()
        try:
            answer = self.ollama_client.generate(
                model=self.cfg.local_llm_model,
                prompt=prompt,
                timeout_seconds=self.cfg.request_timeout_seconds,
            )
        except OllamaGenerateError:
            trace_events.append(
                TraceEvent(
                    stage="generation",
                    message="local generation failed",
                    latency_ms=_latency_ms(generation_started),
                )
            )
            trace = AgentTrace(
                model=self.cfg.local_llm_model,
                prompt_version=self.prompt_version,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tool_calls=trace_events,
                termination_reason="generation_error",
                total_latency_ms=_latency_ms(started),
            )
            return ChatResponse(
                answer=(
                    "I could not complete local generation right now. Please retry or "
                    "switch to deep mode."
                ),
                mode=mode,
                document_id=document_id,
                insufficient_evidence=True,
                citations=citations,
                trace=trace,
            )

        generation_latency_ms = _latency_ms(generation_started)
        trace_events.append(
            TraceEvent(
                stage="generation",
                message="generated local answer",
                latency_ms=generation_latency_ms,
                metadata={"model": self.cfg.local_llm_model},
            )
        )

        clean_answer = answer.strip()
        insufficient = (
            not clean_answer or "INSUFFICIENT_EVIDENCE" in clean_answer.upper()
        )
        if insufficient:
            clean_answer = (
                "I do not have enough grounded evidence in the indexed document "
                "to answer that question."
            )

        trace = AgentTrace(
            model=self.cfg.local_llm_model,
            prompt_version=self.prompt_version,
            retrieved_chunk_ids=retrieved_chunk_ids,
            tool_calls=trace_events,
            termination_reason="insufficient_evidence" if insufficient else "completed",
            total_latency_ms=_latency_ms(started),
        )
        return ChatResponse(
            answer=clean_answer,
            mode=mode,
            document_id=document_id,
            insufficient_evidence=insufficient,
            citations=citations,
            trace=trace,
        )


def build_rag_prompt(
    question: str,
    evidence: list[RetrievedEvidence],
    mode: Mode,
    prompt_version: str,
) -> str:
    blocks = []
    for item in evidence:
        page_fragment = (
            f" pages={','.join(str(page) for page in item.page_refs)}"
            if item.page_refs
            else ""
        )
        blocks.append(f"[{item.chunk_id}{page_fragment}]\n{item.text}")
    joined_blocks = "\n\n".join(blocks) if blocks else "(no context)"

    return (
        f"PROMPT_VERSION={prompt_version}\n"
        f"MODE={mode.value}\n"
        "You are a retrieval-grounded assistant. Use only the provided evidence chunks.\n"
        "If the evidence is missing or insufficient, respond with the token INSUFFICIENT_EVIDENCE.\n"
        "Do not fabricate values, clauses, dates, or totals.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{joined_blocks}\n\n"
        "Answer in plain text and keep it concise."
    )


def generate_local(
    *,
    question: str,
    mode: Mode,
    index_store: RetrievalStore,
    document_id: str | None = None,
    cfg: Settings = settings,
) -> ChatResponse:
    engine = LocalQAEngine(index_store=index_store, cfg=cfg)
    return engine.ask(question=question, mode=mode, document_id=document_id)


def _to_evidence(match: QueryMatch) -> RetrievedEvidence:
    payload = match.payload
    text = str(payload.get("text", "")).strip()
    chunk_id = str(payload.get("chunk_id", "")).strip() or match.record_id
    raw_pages = payload.get("page_refs", [])
    page_refs: list[int] = []
    if isinstance(raw_pages, list):
        for raw_page in raw_pages:
            try:
                page_refs.append(int(raw_page))
            except (TypeError, ValueError):
                continue

    return RetrievedEvidence(
        chunk_id=chunk_id,
        text=text,
        page_refs=page_refs,
        score=float(match.score),
    )


def _to_citation(evidence: RetrievedEvidence) -> Citation:
    preview = evidence.text[:500]
    return Citation(
        chunk_id=evidence.chunk_id,
        page=evidence.page_refs[0] if evidence.page_refs else None,
        text=preview if preview else "(empty chunk)",
    )


def _is_insufficient_evidence(
    question: str,
    evidence: list[RetrievedEvidence],
    min_token_overlap: int,
) -> bool:
    if not evidence:
        return True

    combined = " ".join(item.text for item in evidence if item.text)
    if not combined.strip():
        return True

    question_tokens = {token for token in _tokenize(question) if len(token) >= 3}
    if not question_tokens:
        return False
    evidence_tokens = set(_tokenize(combined))
    overlap = question_tokens.intersection(evidence_tokens)
    return len(overlap) < min_token_overlap


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
