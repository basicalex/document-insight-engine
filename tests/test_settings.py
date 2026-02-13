from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import Settings


def test_settings_builds_runtime_paths(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path)

    assert cfg.uploads_dir == tmp_path / "uploads"
    assert cfg.parsed_dir == tmp_path / "parsed"
    assert cfg.traces_dir == tmp_path / "traces"


def test_settings_rejects_invalid_api_port() -> None:
    with pytest.raises(ValidationError):
        Settings(api_port=70000)


def test_settings_rejects_timeout_inversion() -> None:
    with pytest.raises(ValidationError):
        Settings(request_timeout_seconds=121, ingest_timeout_seconds=120)


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9001")

    cfg = Settings()

    assert cfg.api_port == 9001


def test_settings_ensure_runtime_dirs(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path)
    cfg.ensure_runtime_dirs()

    assert (tmp_path / "uploads").exists()
    assert (tmp_path / "parsed").exists()
    assert (tmp_path / "traces").exists()


def test_settings_rejects_enabled_deep_mode_with_disabled_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(deep_mode_enabled=True, cloud_agent_provider="disabled")


def test_settings_accepts_enabled_deep_mode_with_fallback_provider() -> None:
    cfg = Settings(deep_mode_enabled=True, cloud_agent_provider="fallback")
    assert cfg.deep_mode_enabled is True


def test_settings_accepts_enabled_deep_mode_with_local_provider() -> None:
    cfg = Settings(deep_mode_enabled=True, cloud_agent_provider="local")
    assert cfg.deep_mode_enabled is True
    assert cfg.cloud_agent_provider == "local"


def test_settings_rejects_gemini_provider_without_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings(cloud_agent_provider="gemini")


def test_settings_accepts_gemini_provider_with_api_key() -> None:
    cfg = Settings(cloud_agent_provider="gemini", cloud_agent_api_key="test-key")
    assert cfg.cloud_agent_provider == "gemini"


def test_settings_rejects_retry_initial_backoff_above_max() -> None:
    with pytest.raises(ValidationError):
        Settings(
            cloud_agent_retry_initial_backoff_seconds=2.0,
            cloud_agent_retry_max_backoff_seconds=1.0,
        )


def test_settings_rejects_provider_embedding_rollout_without_gemini_key() -> None:
    with pytest.raises(ValidationError):
        Settings(
            embedding_rollout_mode="provider",
            cloud_embedding_provider="gemini",
            cloud_agent_provider="fallback",
            cloud_agent_api_key=None,
        )


def test_settings_allows_hash_fallback_embedding_rollout_without_gemini_key() -> None:
    cfg = Settings(
        embedding_rollout_mode="provider_with_hash_fallback",
        cloud_embedding_provider="gemini",
        cloud_agent_provider="fallback",
        cloud_agent_api_key=None,
    )
    assert cfg.embedding_rollout_mode == "provider_with_hash_fallback"


def test_settings_accepts_memory_api_state_backend() -> None:
    cfg = Settings(api_state_backend="memory")
    assert cfg.api_state_backend == "memory"


def test_settings_rejects_empty_api_state_key_prefix() -> None:
    with pytest.raises(ValidationError):
        Settings(api_state_key_prefix="   ")


def test_settings_accepts_memory_ingestion_queue_backend() -> None:
    cfg = Settings(ingestion_queue_backend="memory")
    assert cfg.ingestion_queue_backend == "memory"


def test_settings_rejects_empty_ingestion_queue_key_prefix() -> None:
    with pytest.raises(ValidationError):
        Settings(ingestion_queue_key_prefix="   ")


def test_settings_allows_in_memory_index_fallback_in_dev() -> None:
    cfg = Settings(environment="dev", allow_in_memory_index_fallback=True)
    assert cfg.allow_in_memory_index_fallback is True


def test_settings_rejects_in_memory_index_fallback_in_prod() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="prod", allow_in_memory_index_fallback=True)


def test_settings_supports_optional_capability_toggles() -> None:
    cfg = Settings(docling_enabled=False, langextract_enabled=False)
    assert cfg.docling_enabled is False
    assert cfg.langextract_enabled is False


def test_settings_accepts_parser_routing_modes() -> None:
    cfg = Settings(parser_routing_mode="google_docling_fallback")
    assert cfg.parser_routing_mode == "google_docling_fallback"


def test_settings_rejects_invalid_slo_rate_bounds() -> None:
    with pytest.raises(ValidationError):
        Settings(slo_insufficient_evidence_rate_max=1.5)


def test_settings_accepts_eval_threshold_overrides() -> None:
    cfg = Settings(
        eval_min_grounded_accuracy=0.95,
        eval_max_hallucination_rate=0.05,
        eval_min_citation_completeness=0.92,
        eval_max_p95_latency_ms=1800,
    )
    assert cfg.eval_min_grounded_accuracy == 0.95
    assert cfg.eval_max_hallucination_rate == 0.05
    assert cfg.eval_min_citation_completeness == 0.92
    assert cfg.eval_max_p95_latency_ms == 1800
