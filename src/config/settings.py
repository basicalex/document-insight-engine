from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


ALLOWED_UPLOAD_MIME_TYPES = (
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_name: str = "document-insight-engine"
    environment: Literal["local", "dev", "staging", "prod"] = "local"

    data_dir: Path = Path("data")
    uploads_dir_name: str = "uploads"
    parsed_dir_name: str = "parsed"
    traces_dir_name: str = "traces"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    redis_url: str = "redis://redis:6379/0"
    redis_index_tier1: str = "tier1_idx"
    redis_index_tier4: str = "tier4_idx"
    redis_connection_retries: PositiveInt = 5
    redis_retry_delay_seconds: float = 1.0
    redis_retry_backoff_factor: float = 2.0
    allow_in_memory_index_fallback: bool = False
    api_state_backend: Literal["auto", "redis", "memory"] = "auto"
    api_state_key_prefix: str = "die:state"
    api_state_idempotency_ttl_seconds: PositiveInt = 24 * 60 * 60
    api_state_ingestion_ttl_seconds: PositiveInt = 30 * 24 * 60 * 60
    api_state_session_ttl_seconds: PositiveInt = 7 * 24 * 60 * 60
    api_state_idempotency_claim_ttl_seconds: PositiveInt = 5 * 60
    api_state_session_max_turns: PositiveInt = 8
    ingestion_queue_backend: Literal["auto", "redis", "memory"] = "auto"
    ingestion_queue_key_prefix: str = "die:queue"
    ingestion_worker_concurrency: PositiveInt = 1
    ingestion_queue_max_retries: int = Field(default=2, ge=0, le=10)
    ingestion_queue_retry_backoff_seconds: float = Field(default=0.25, ge=0.0, le=30.0)
    ingestion_queue_poll_timeout_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    ingestion_queue_dead_letter_max_items: PositiveInt = 500

    slo_http_request_p95_ms: PositiveInt = 1500
    slo_retrieval_p95_ms: PositiveInt = 1000
    slo_generation_p95_ms: PositiveInt = 2500
    slo_insufficient_evidence_rate_max: float = Field(default=0.35, ge=0.0, le=1.0)
    slo_citation_completeness_min: float = Field(default=0.90, ge=0.0, le=1.0)
    slo_grounding_gap_rate_max: float = Field(default=0.20, ge=0.0, le=1.0)

    eval_min_grounded_accuracy: float = Field(default=0.90, ge=0.0, le=1.0)
    eval_max_hallucination_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    eval_min_citation_completeness: float = Field(default=0.90, ge=0.0, le=1.0)
    eval_max_p95_latency_ms: PositiveInt = 2500

    ollama_base_url: str = "http://ollama:11434"
    local_llm_model: str = "hadad/LFM2.5-1.2B:Q8_0"
    local_deep_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "local_deep_model",
            "LOCAL_DEEP_MODEL",
            "CLOUD_AGENT_LOCAL_MODEL",
        ),
    )
    embedding_rollout_mode: Literal[
        "hash", "provider", "provider_with_hash_fallback"
    ] = "provider_with_hash_fallback"
    embedding_filter_strict: bool = True
    embedding_timeout_seconds: PositiveInt = 30
    local_embedding_provider: Literal["hash", "ollama"] = "ollama"
    local_embedding_model: str = "all-minilm"
    local_embedding_dimension: PositiveInt = 384
    cloud_embedding_provider: Literal["hash", "ollama", "gemini"] = "ollama"
    cloud_embedding_model: str = "all-minilm"
    cloud_embedding_dimension: PositiveInt = 384

    docling_enabled: bool = True
    google_parser_enabled: bool = True
    parser_routing_mode: Literal[
        "docling_google_fallback",
        "google_docling_fallback",
        "docling_fallback",
        "google_fallback",
        "fallback_only",
    ] = "docling_google_fallback"
    langextract_enabled: bool = True
    extraction_max_input_tokens: PositiveInt = 12000
    extraction_max_output_tokens: PositiveInt = 4000
    extraction_strict_schema: bool = True

    deep_mode_enabled: bool = False
    cloud_agent_provider: Literal["disabled", "fallback", "gemini", "local"] = (
        "disabled"
    )
    cloud_agent_model: str = "gemini-3-flash"
    cloud_agent_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "cloud_agent_api_key",
            "CLOUD_AGENT_API_KEY",
            "GOOGLE_API_KEY",
        ),
    )
    cloud_agent_api_base_url: str = "https://generativelanguage.googleapis.com"
    cloud_agent_timeout_seconds: PositiveInt = 30
    cloud_agent_retry_attempts: int = Field(default=3, ge=1, le=10)
    cloud_agent_retry_initial_backoff_seconds: float = Field(default=0.5, gt=0)
    cloud_agent_retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=5.0)
    cloud_agent_retry_max_backoff_seconds: float = Field(default=8.0, gt=0)

    max_upload_size_mb: PositiveInt = 50
    request_timeout_seconds: PositiveInt = 60
    ingest_timeout_seconds: PositiveInt = 120

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, value: int) -> int:
        if value < 1 or value > 65535:
            raise ValueError("api_port must be between 1 and 65535")
        return value

    @field_validator("uploads_dir_name", "parsed_dir_name", "traces_dir_name")
    @classmethod
    def validate_relative_dir_name(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("directory names must be relative and traversal-safe")
        return value

    @field_validator("api_state_key_prefix", "ingestion_queue_key_prefix")
    @classmethod
    def validate_key_prefix(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("key prefix cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_timeouts(self) -> "Settings":
        if self.request_timeout_seconds > self.ingest_timeout_seconds:
            raise ValueError(
                "request_timeout_seconds cannot exceed ingest_timeout_seconds"
            )
        if (
            self.cloud_agent_retry_initial_backoff_seconds
            > self.cloud_agent_retry_max_backoff_seconds
        ):
            raise ValueError(
                "cloud_agent_retry_initial_backoff_seconds cannot exceed cloud_agent_retry_max_backoff_seconds"
            )
        if self.cloud_agent_provider == "gemini" and not (
            self.cloud_agent_api_key and self.cloud_agent_api_key.strip()
        ):
            raise ValueError(
                "cloud_agent_api_key is required when cloud_agent_provider is gemini"
            )
        if (
            self.embedding_rollout_mode == "provider"
            and self.cloud_embedding_provider == "gemini"
            and not (self.cloud_agent_api_key and self.cloud_agent_api_key.strip())
        ):
            raise ValueError(
                "cloud_agent_api_key is required for provider embedding rollout with gemini"
            )
        if self.allow_in_memory_index_fallback and self.environment not in (
            "local",
            "dev",
        ):
            raise ValueError(
                "allow_in_memory_index_fallback is only permitted in local or dev environments"
            )
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / self.uploads_dir_name

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / self.parsed_dir_name

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / self.traces_dir_name

    def ensure_runtime_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.parsed_dir, self.traces_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


try:
    settings = get_settings()
except ValidationError as exc:
    raise RuntimeError(f"Invalid runtime settings: {exc}") from exc
