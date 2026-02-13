from pathlib import Path


def test_compose_includes_core_services() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "redis:" in compose
    assert "ollama:" in compose
    assert "api:" in compose
    assert "ui:" in compose
    assert "./data:/app/data" in compose
    assert "./models:/root/.ollama" in compose
    assert "redis_data:/data" in compose
    assert "8501:8501" in compose
    assert "mem_limit" in compose
    assert 'DOCLING_ENABLED: "true"' in compose
    assert 'GOOGLE_PARSER_ENABLED: "true"' in compose
    assert 'LANGEXTRACT_ENABLED: "true"' in compose
    assert 'DEEP_MODE_ENABLED: "true"' in compose
    assert "CLOUD_AGENT_PROVIDER: local" in compose


def test_dev_compose_enables_reload_and_bind_mounts() -> None:
    compose = Path("docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "ui:" in compose
    assert "--reload" in compose
    assert "./src:/app/src" in compose
    assert "./frontend:/app/frontend" in compose
    assert "--server.fileWatcherType=poll" in compose
    assert 'DEEP_MODE_ENABLED: "true"' in compose
    assert "CLOUD_AGENT_PROVIDER: local" in compose


def test_dockerfile_installs_optional_ai_dependencies() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "libgl1" in dockerfile
    assert "libglib2.0-0" in dockerfile
    assert 'pip install --no-cache-dir ".[ui,ai]"' in dockerfile
