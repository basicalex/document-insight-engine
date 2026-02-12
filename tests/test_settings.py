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
