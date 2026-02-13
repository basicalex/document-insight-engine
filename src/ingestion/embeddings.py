from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.config.settings import Settings, settings
from src.ingestion.vectorize import hashing_vector


class EmbeddingProviderError(Exception):
    pass


@dataclass(frozen=True)
class TextEmbedding:
    vector: list[float]
    provider: str
    model: str
    version: str


class TextEmbeddingClient(Protocol):
    provider: str
    model: str
    version: str
    dimension: int

    def embed_text(self, text: str) -> TextEmbedding: ...


@dataclass(frozen=True)
class _HttpResponse:
    status_code: int
    payload: dict[str, Any]


Transport = Callable[[str, dict[str, Any], dict[str, str], float], _HttpResponse]


class HashingEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        version: str = "hash-v1",
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.provider = "hash"
        self.model = model
        self.version = version
        self.dimension = dimension

    def embed_text(self, text: str) -> TextEmbedding:
        vector = hashing_vector(text=text, dimension=self.dimension)
        return TextEmbedding(
            vector=vector,
            provider=self.provider,
            model=self.model,
            version=self.version,
        )


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        timeout_seconds: int,
        version: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.provider = "ollama"
        self.model = model
        self.dimension = dimension
        self.timeout_seconds = float(timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.version = version or f"ollama:{model}:v1"
        self._transport = transport or _default_transport

    def embed_text(self, text: str) -> TextEmbedding:
        payload = {
            "model": self.model,
            "prompt": text,
        }
        endpoint = f"{self.base_url}/api/embeddings"
        response = self._transport(
            endpoint,
            payload,
            {"Content-Type": "application/json", "Accept": "application/json"},
            self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise EmbeddingProviderError(
                f"ollama embedding request failed with status {response.status_code}"
            )

        vector = _extract_embedding_vector(response.payload)
        _validate_vector_dimension(
            vector=vector,
            expected_dimension=self.dimension,
            context=f"ollama model {self.model}",
        )
        return TextEmbedding(
            vector=vector,
            provider=self.provider,
            model=self.model,
            version=self.version,
        )


class GeminiEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dimension: int,
        timeout_seconds: int,
        task_type: str,
        version: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.provider = "gemini"
        self.model = model
        self.dimension = dimension
        self.timeout_seconds = float(timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip()
        self.task_type = task_type
        self.version = version or f"gemini:{model}:v1"
        self._transport = transport or _default_transport

    def embed_text(self, text: str) -> TextEmbedding:
        if not self.api_key:
            raise EmbeddingProviderError("gemini embedding API key is not configured")

        endpoint = (
            f"{self.base_url}/v1beta/models/{self.model}:embedContent"
            f"?key={self.api_key}"
        )
        payload = {
            "content": {
                "parts": [{"text": text}],
            },
            "taskType": self.task_type,
        }
        response = self._transport(
            endpoint,
            payload,
            {"Content-Type": "application/json", "Accept": "application/json"},
            self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise EmbeddingProviderError(
                f"gemini embedding request failed with status {response.status_code}"
            )

        vector = _extract_embedding_vector(response.payload)
        _validate_vector_dimension(
            vector=vector,
            expected_dimension=self.dimension,
            context=f"gemini model {self.model}",
        )
        return TextEmbedding(
            vector=vector,
            provider=self.provider,
            model=self.model,
            version=self.version,
        )


class FallbackEmbeddingClient:
    def __init__(
        self,
        *,
        primary: TextEmbeddingClient,
        fallback: TextEmbeddingClient,
    ) -> None:
        if primary.dimension != fallback.dimension:
            raise ValueError("primary and fallback embedding dimensions must match")
        self.primary = primary
        self.fallback = fallback
        self.provider = primary.provider
        self.model = primary.model
        self.version = primary.version
        self.dimension = primary.dimension

    def embed_text(self, text: str) -> TextEmbedding:
        try:
            return self.primary.embed_text(text)
        except EmbeddingProviderError:
            return self.fallback.embed_text(text)


def build_ingestion_embedding_clients(
    cfg: Settings = settings,
) -> tuple[TextEmbeddingClient, TextEmbeddingClient]:
    tier1_provider = _build_tier1_provider_client(cfg)
    tier4_provider = _build_tier4_provider_client(cfg)
    tier1_hash = _build_tier1_hash_client(cfg)
    tier4_hash = _build_tier4_hash_client(cfg)

    if cfg.embedding_rollout_mode == "hash":
        return tier1_hash, tier4_hash
    if cfg.embedding_rollout_mode == "provider":
        return tier1_provider, tier4_provider

    return (
        FallbackEmbeddingClient(primary=tier1_provider, fallback=tier1_hash),
        FallbackEmbeddingClient(primary=tier4_provider, fallback=tier4_hash),
    )


def build_query_embedding_clients(
    cfg: Settings = settings,
) -> tuple[TextEmbeddingClient, TextEmbeddingClient | None]:
    provider_client = _build_tier1_provider_client(cfg)
    hash_client = _build_tier1_hash_client(cfg)
    if cfg.embedding_rollout_mode == "hash":
        return hash_client, None
    if cfg.embedding_rollout_mode == "provider":
        return provider_client, None
    return provider_client, hash_client


def _build_tier1_hash_client(cfg: Settings) -> HashingEmbeddingClient:
    return HashingEmbeddingClient(
        model=f"hash:{cfg.local_embedding_model}",
        dimension=cfg.local_embedding_dimension,
    )


def _build_tier4_hash_client(cfg: Settings) -> HashingEmbeddingClient:
    return HashingEmbeddingClient(
        model=f"hash:{cfg.cloud_embedding_model}",
        dimension=cfg.cloud_embedding_dimension,
    )


def _build_tier1_provider_client(cfg: Settings) -> TextEmbeddingClient:
    if cfg.local_embedding_provider == "hash":
        return _build_tier1_hash_client(cfg)
    return OllamaEmbeddingClient(
        base_url=cfg.ollama_base_url,
        model=cfg.local_embedding_model,
        dimension=cfg.local_embedding_dimension,
        timeout_seconds=cfg.embedding_timeout_seconds,
    )


def _build_tier4_provider_client(cfg: Settings) -> TextEmbeddingClient:
    if cfg.cloud_embedding_provider == "hash":
        return _build_tier4_hash_client(cfg)
    if cfg.cloud_embedding_provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=cfg.ollama_base_url,
            model=cfg.cloud_embedding_model,
            dimension=cfg.cloud_embedding_dimension,
            timeout_seconds=cfg.embedding_timeout_seconds,
        )
    return GeminiEmbeddingClient(
        base_url=cfg.cloud_agent_api_base_url,
        api_key=cfg.cloud_agent_api_key,
        model=cfg.cloud_embedding_model,
        dimension=cfg.cloud_embedding_dimension,
        timeout_seconds=cfg.embedding_timeout_seconds,
        task_type="RETRIEVAL_DOCUMENT",
    )


def _extract_embedding_vector(payload: dict[str, Any]) -> list[float]:
    direct = payload.get("embedding")
    if isinstance(direct, dict):
        values = direct.get("values")
        if isinstance(values, list):
            return _coerce_float_list(values)
    if isinstance(direct, list):
        return _coerce_float_list(direct)

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, dict) and isinstance(first.get("values"), list):
            return _coerce_float_list(first["values"])
        if isinstance(first, list):
            return _coerce_float_list(first)

    raise EmbeddingProviderError("provider did not return an embedding vector")


def _coerce_float_list(values: list[Any]) -> list[float]:
    vector: list[float] = []
    for value in values:
        try:
            vector.append(float(value))
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                "embedding vector contains non-numeric values"
            ) from exc
    return vector


def _validate_vector_dimension(
    *,
    vector: list[float],
    expected_dimension: int,
    context: str,
) -> None:
    if len(vector) != expected_dimension:
        raise EmbeddingProviderError(
            f"{context} returned dimension {len(vector)}; expected {expected_dimension}"
        )


def _default_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> _HttpResponse:
    request = urllib_request.Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if not isinstance(parsed, dict):
                raise EmbeddingProviderError(
                    "provider returned non-object JSON payload"
                )
            return _HttpResponse(
                status_code=int(getattr(response, "status", 200)),
                payload=parsed,
            )
    except urllib_error.HTTPError as exc:
        return _HttpResponse(status_code=exc.code, payload=_safe_json_body(exc.read()))
    except urllib_error.URLError as exc:
        raise EmbeddingProviderError(f"provider request failed: {exc}") from exc


def _safe_json_body(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed
