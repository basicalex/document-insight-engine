from pathlib import Path


def test_compose_includes_core_services() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "redis:" in compose
    assert "ollama:" in compose
    assert "api:" in compose
    assert "./data:/app/data" in compose
    assert "./models:/root/.ollama" in compose
    assert "redis_data:/data" in compose
    assert "mem_limit" in compose
