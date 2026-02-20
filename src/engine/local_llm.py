from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, cast
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.config.settings import Settings, settings
from src.ingestion.embeddings import TextEmbeddingClient
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


@dataclass(frozen=True)
class QueryEmbeddingCandidate:
    vector: list[float]
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None


class OllamaClient(Protocol):
    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str: ...

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
    ) -> Iterator[str]: ...


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


class ProviderQueryEmbedder:
    def __init__(
        self,
        *,
        primary_client: TextEmbeddingClient,
        fallback_client: TextEmbeddingClient | None = None,
    ) -> None:
        self.primary_client = primary_client
        self.fallback_client = fallback_client

    def embed_query(self, text: str, dimension: int) -> list[float]:
        first = self.embed_query_candidates(text=text, dimension=dimension)[0]
        return first.vector

    def embed_query_candidates(
        self, *, text: str, dimension: int
    ) -> list[QueryEmbeddingCandidate]:
        primary = self.primary_client.embed_text(text)
        if len(primary.vector) != dimension:
            raise ValueError(
                f"query vector dimension {len(primary.vector)} does not match expected {dimension}"
            )

        candidates = [
            QueryEmbeddingCandidate(
                vector=primary.vector,
                embedding_provider=primary.provider,
                embedding_model=primary.model,
                embedding_version=primary.version,
            )
        ]

        if self.fallback_client is None:
            return candidates

        fallback = self.fallback_client.embed_text(text)
        if len(fallback.vector) != dimension:
            raise ValueError(
                f"fallback query vector dimension {len(fallback.vector)} does not match expected {dimension}"
            )
        if (
            fallback.provider,
            fallback.model,
            fallback.version,
        ) == (
            primary.provider,
            primary.model,
            primary.version,
        ):
            return candidates

        candidates.append(
            QueryEmbeddingCandidate(
                vector=fallback.vector,
                embedding_provider=fallback.provider,
                embedding_model=fallback.model,
                embedding_version=fallback.version,
            )
        )
        return candidates


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

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
    ) -> Iterator[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
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
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        decoded = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaGenerateError(
                            "Ollama returned invalid JSON while streaming"
                        ) from exc

                    if decoded.get("error"):
                        raise OllamaGenerateError(str(decoded["error"]))

                    token = str(decoded.get("response", ""))
                    if token:
                        yield token

                    if bool(decoded.get("done", False)):
                        break
        except urllib_error.URLError as exc:
            raise OllamaGenerateError(f"Ollama request failed: {exc}") from exc


class GeminiTextGenerationClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()

    def generate(self, *, model: str, prompt: str, timeout_seconds: int) -> str:
        resolved_model = model.strip() or self.model
        if not self.api_key:
            raise OllamaGenerateError("Gemini API key is not configured")
        if not resolved_model:
            raise OllamaGenerateError("Gemini model is not configured")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generation_config": {
                "temperature": 0.0,
            },
        }
        endpoint = (
            f"{self.base_url}/v1beta/models/{resolved_model}:generateContent"
            f"?key={self.api_key}"
        )
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            endpoint,
            method="POST",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            raise OllamaGenerateError(
                f"Gemini request failed with status {exc.code}"
            ) from exc
        except urllib_error.URLError as exc:
            raise OllamaGenerateError(f"Gemini request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaGenerateError("Gemini returned invalid JSON") from exc

        text = _extract_gemini_text(decoded)
        if not text:
            raise OllamaGenerateError("Gemini returned an empty response")
        return text

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
    ) -> Iterator[str]:
        token = self.generate(
            model=model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
        )
        if token:
            yield token


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""

    first = candidates[0]
    if not isinstance(first, dict):
        return ""

    content = first.get("content")
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    fragments = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "".join(fragments).strip()


class LocalQAEngine:
    def __init__(
        self,
        index_store: RetrievalStore,
        cfg: Settings = settings,
        embedder: QueryEmbedder | None = None,
        ollama_client: OllamaClient | None = None,
        generation_model: str | None = None,
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
        self.generation_model = generation_model or cfg.local_llm_model
        self._validate_query_configuration()

    def _validate_query_configuration(self) -> None:
        profile_getter = getattr(self.index_store, "profile", None)
        if not callable(profile_getter):
            return

        profile = profile_getter(EmbeddingTier.TIER1)
        expected_dimension = int(
            getattr(profile, "dimension", self.query_vector_dimension)
        )
        if expected_dimension != self.query_vector_dimension:
            raise ValueError(
                "query_vector_dimension must match tier1 index embedding dimension"
            )

    def ask(
        self,
        question: str,
        mode: Mode,
        document_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        raw_matches, retrieval_latency_ms = self._retrieve_matches(
            question=question,
            document_id=document_id,
        )

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
                model=self.generation_model,
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
            chat_history=chat_history,
        )
        generation_started = time.perf_counter()
        try:
            answer = self.ollama_client.generate(
                model=self.generation_model,
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
                model=self.generation_model,
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
                metadata={"model": self.generation_model},
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
            model=self.generation_model,
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

    def ask_stream_events(
        self,
        *,
        question: str,
        mode: Mode,
        document_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        yield {
            "type": "status",
            "phase": "retrieval",
            "message": "Retrieving grounded evidence...",
        }

        started = time.perf_counter()
        raw_matches, retrieval_latency_ms = self._retrieve_matches(
            question=question,
            document_id=document_id,
        )

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
            answer = (
                "I do not have enough grounded evidence in the indexed document "
                "to answer that question."
            )
            response = ChatResponse(
                answer=answer,
                mode=mode,
                document_id=document_id,
                insufficient_evidence=True,
                citations=citations,
                trace=AgentTrace(
                    model=self.generation_model,
                    prompt_version=self.prompt_version,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    tool_calls=trace_events,
                    termination_reason="insufficient_evidence",
                    total_latency_ms=_latency_ms(started),
                ),
            )
            yield {"type": "token", "delta": answer}
            yield {"type": "final", "response": response.model_dump(mode="json")}
            return

        prompt = build_rag_prompt(
            question=question,
            evidence=evidence,
            mode=mode,
            prompt_version=self.prompt_version,
            chat_history=chat_history,
        )

        generation_started = time.perf_counter()
        chunks: list[str] = []
        stream_generate = getattr(self.ollama_client, "generate_stream", None)
        yield {
            "type": "status",
            "phase": "generation",
            "message": "Generating answer...",
        }
        try:
            if callable(stream_generate):
                stream_generate_fn = cast(Callable[..., Iterator[str]], stream_generate)
                for token in stream_generate_fn(
                    model=self.generation_model,
                    prompt=prompt,
                    timeout_seconds=self.cfg.request_timeout_seconds,
                ):
                    if not token:
                        continue
                    chunks.append(token)
                    yield {"type": "token", "delta": token}
            else:
                answer = self.ollama_client.generate(
                    model=self.generation_model,
                    prompt=prompt,
                    timeout_seconds=self.cfg.request_timeout_seconds,
                )
                if answer:
                    chunks.append(answer)
                    yield {"type": "token", "delta": answer}
        except OllamaGenerateError:
            answer = (
                "I could not complete local generation right now. Please retry or "
                "switch to deep mode."
            )
            trace_events.append(
                TraceEvent(
                    stage="generation",
                    message="local generation failed",
                    latency_ms=_latency_ms(generation_started),
                )
            )
            response = ChatResponse(
                answer=answer,
                mode=mode,
                document_id=document_id,
                insufficient_evidence=True,
                citations=citations,
                trace=AgentTrace(
                    model=self.generation_model,
                    prompt_version=self.prompt_version,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    tool_calls=trace_events,
                    termination_reason="generation_error",
                    total_latency_ms=_latency_ms(started),
                ),
            )
            yield {"type": "token", "delta": answer}
            yield {"type": "final", "response": response.model_dump(mode="json")}
            return

        generation_latency_ms = _latency_ms(generation_started)
        trace_events.append(
            TraceEvent(
                stage="generation",
                message="generated local answer",
                latency_ms=generation_latency_ms,
                metadata={"model": self.generation_model},
            )
        )

        clean_answer = "".join(chunks).strip()
        insufficient = (
            not clean_answer or "INSUFFICIENT_EVIDENCE" in clean_answer.upper()
        )
        if insufficient:
            clean_answer = (
                "I do not have enough grounded evidence in the indexed document "
                "to answer that question."
            )

        response = ChatResponse(
            answer=clean_answer,
            mode=mode,
            document_id=document_id,
            insufficient_evidence=insufficient,
            citations=citations,
            trace=AgentTrace(
                model=self.generation_model,
                prompt_version=self.prompt_version,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tool_calls=trace_events,
                termination_reason="insufficient_evidence"
                if insufficient
                else "completed",
                total_latency_ms=_latency_ms(started),
            ),
        )
        yield {"type": "final", "response": response.model_dump(mode="json")}

    def _retrieve_matches(
        self,
        *,
        question: str,
        document_id: str | None,
    ) -> tuple[list[QueryMatch], int]:
        retrieval_started = time.perf_counter()

        for candidate in self._query_candidates(question):
            filters = self._candidate_filters(
                document_id=document_id,
                candidate=candidate,
            )
            matches = self.index_store.query(
                tier=EmbeddingTier.TIER1,
                vector=candidate.vector,
                top_k=self.retrieval_top_k,
                filters=filters,
            )
            if matches:
                return matches, _latency_ms(retrieval_started)

        return [], _latency_ms(retrieval_started)

    def _query_candidates(self, question: str) -> list[QueryEmbeddingCandidate]:
        candidate_fn = getattr(self.embedder, "embed_query_candidates", None)
        if callable(candidate_fn):
            raw_candidates = cast(
                list[object],
                candidate_fn(
                    text=question,
                    dimension=self.query_vector_dimension,
                ),
            )
            candidates = [
                candidate
                for candidate in raw_candidates
                if isinstance(candidate, QueryEmbeddingCandidate)
            ]
            if candidates:
                for candidate in candidates:
                    if len(candidate.vector) != self.query_vector_dimension:
                        raise ValueError(
                            "query embedding candidate dimension does not match expected dimension"
                        )
                return candidates

        vector = self.embedder.embed_query(
            text=question,
            dimension=self.query_vector_dimension,
        )
        return [QueryEmbeddingCandidate(vector=vector)]

    def _candidate_filters(
        self,
        *,
        document_id: str | None,
        candidate: QueryEmbeddingCandidate,
    ) -> dict[str, Any] | None:
        filters: dict[str, Any] = {}
        if document_id:
            filters["document_id"] = document_id

        if (
            self.cfg.embedding_filter_strict
            and candidate.embedding_provider
            and candidate.embedding_model
            and candidate.embedding_version
        ):
            filters.update(
                {
                    "embedding_provider": candidate.embedding_provider,
                    "embedding_model": candidate.embedding_model,
                    "embedding_version": candidate.embedding_version,
                }
            )

        return filters or None


def build_rag_prompt(
    question: str,
    evidence: list[RetrievedEvidence],
    mode: Mode,
    prompt_version: str,
    chat_history: list[dict[str, str]] | None = None,
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
    history = _format_chat_history(chat_history)

    return (
        f"PROMPT_VERSION={prompt_version}\n"
        f"MODE={mode.value}\n"
        "You are a retrieval-grounded assistant. Use only the provided evidence chunks.\n"
        "Conversation history is for follow-up disambiguation only and is not evidence.\n"
        "If the evidence is missing or insufficient, respond with the token INSUFFICIENT_EVIDENCE.\n"
        "Do not fabricate values, clauses, dates, or totals.\n\n"
        f"Conversation history:\n{history}\n\n"
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


def _format_chat_history(chat_history: list[dict[str, str]] | None) -> str:
    if not chat_history:
        return "(none)"

    lines: list[str] = []
    for item in chat_history[-4:]:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question and not answer:
            continue
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")

    return "\n".join(lines) if lines else "(none)"


def _to_evidence(match: QueryMatch) -> RetrievedEvidence:
    payload = match.payload
    text = str(payload.get("text", "")).strip()
    chunk_id = str(payload.get("chunk_id", "")).strip() or match.record_id
    page_refs = _parse_page_refs(payload.get("page_refs", []))

    return RetrievedEvidence(
        chunk_id=chunk_id,
        text=text,
        page_refs=page_refs,
        score=float(match.score),
    )


def _to_citation(evidence: RetrievedEvidence) -> Citation:
    preview = evidence.text[:500]
    has_text = bool(preview)
    return Citation(
        chunk_id=evidence.chunk_id,
        page=evidence.page_refs[0] if evidence.page_refs else None,
        text=preview if preview else "(empty chunk)",
        start_offset=0 if has_text else None,
        end_offset=len(preview) if has_text else None,
    )


def _parse_page_refs(raw_pages: Any) -> list[int]:
    values: list[Any]
    if isinstance(raw_pages, list):
        values = list(raw_pages)
    elif isinstance(raw_pages, str):
        candidate = raw_pages.strip()
        if not candidate:
            return []
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            decoded = None

        if isinstance(decoded, list):
            values = decoded
        elif isinstance(decoded, (int, float, str)):
            values = [decoded]
        elif "," in candidate:
            values = [part.strip() for part in candidate.split(",")]
        else:
            values = [candidate]
    else:
        values = [raw_pages]

    page_refs: list[int] = []
    for raw_page in values:
        try:
            page_refs.append(int(raw_page))
        except (TypeError, ValueError):
            continue
    return sorted(set(page_refs))


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
