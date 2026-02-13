from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.config.settings import Settings
from src.ingestion.embeddings import (
    EmbeddingProviderError,
    FallbackEmbeddingClient,
    HashingEmbeddingClient,
    OllamaEmbeddingClient,
    build_ingestion_embedding_clients,
    build_query_embedding_clients,
)


@dataclass(frozen=True)
class _StubHttpResponse:
    status_code: int
    payload: dict[str, Any]


def test_ollama_embedding_client_parses_embedding_payload() -> None:
    captured_payload: dict[str, Any] = {}

    def _transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> _StubHttpResponse:
        del url, headers, timeout_seconds
        captured_payload.update(payload)
        return _StubHttpResponse(
            status_code=200,
            payload={"embedding": [0.1, 0.2, 0.3, 0.4]},
        )

    client = OllamaEmbeddingClient(
        base_url="http://ollama:11434",
        model="all-minilm",
        dimension=4,
        timeout_seconds=10,
        transport=_transport,
    )

    embedding = client.embed_text("invoice total")

    assert captured_payload["prompt"] == "invoice total"
    assert embedding.provider == "ollama"
    assert embedding.model == "all-minilm"
    assert len(embedding.vector) == 4


def test_fallback_embedding_client_uses_hash_on_primary_error() -> None:
    class AlwaysFailClient:
        provider = "ollama"
        model = "broken"
        version = "ollama:broken:v1"
        dimension = 8

        def embed_text(self, text: str) -> Any:
            del text
            raise EmbeddingProviderError("provider unavailable")

    fallback = HashingEmbeddingClient(model="hash:test", dimension=8)
    client = FallbackEmbeddingClient(primary=AlwaysFailClient(), fallback=fallback)

    embedding = client.embed_text("renewal clause")

    assert embedding.provider == "hash"
    assert len(embedding.vector) == 8


def test_build_query_embedding_clients_returns_provider_and_hash_candidates() -> None:
    cfg = Settings(
        cloud_agent_provider="fallback",
        embedding_rollout_mode="provider_with_hash_fallback",
    )

    primary, fallback = build_query_embedding_clients(cfg)

    assert primary.provider == "ollama"
    assert fallback is not None
    assert fallback.provider == "hash"


def test_build_ingestion_embedding_clients_hash_mode_returns_hash_clients() -> None:
    cfg = Settings(
        cloud_agent_provider="fallback",
        embedding_rollout_mode="hash",
    )

    tier1, tier4 = build_ingestion_embedding_clients(cfg)

    assert tier1.provider == "hash"
    assert tier4.provider == "hash"


def test_ollama_embedding_client_rejects_dimension_mismatch() -> None:
    def _transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> _StubHttpResponse:
        del url, payload, headers, timeout_seconds
        return _StubHttpResponse(status_code=200, payload={"embedding": [0.1, 0.2]})

    client = OllamaEmbeddingClient(
        base_url="http://ollama:11434",
        model="all-minilm",
        dimension=4,
        timeout_seconds=10,
        transport=_transport,
    )

    with pytest.raises(EmbeddingProviderError):
        client.embed_text("invoice total")
