from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
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

    ollama_base_url: str = "http://ollama:11434"
    local_llm_model: str = "llama3.2"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    cloud_embedding_model: str = "gemini-embedding-001"

    deep_mode_enabled: bool = False
    cloud_agent_provider: Literal["disabled", "fallback"] = "disabled"

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

    @model_validator(mode="after")
    def validate_timeouts(self) -> "Settings":
        if self.request_timeout_seconds > self.ingest_timeout_seconds:
            raise ValueError(
                "request_timeout_seconds cannot exceed ingest_timeout_seconds"
            )
        if self.deep_mode_enabled and self.cloud_agent_provider == "disabled":
            raise ValueError(
                "cloud_agent_provider cannot be disabled when deep_mode_enabled is true"
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
