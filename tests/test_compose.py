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
    assert "${DIE_PROFILE_ENV_FILE:-.env.profile.lite}" in compose
    assert "REDIS_URL_DOCKER" in compose
    assert "OLLAMA_BASE_URL_DOCKER" in compose
    assert "DOCUMENT_INSIGHT_API_BASE_URL_DOCKER" in compose


def test_dev_compose_enables_reload_and_bind_mounts() -> None:
    compose = Path("docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "ui:" in compose
    assert "--reload" in compose
    assert "./src:/app/src" in compose
    assert "./frontend:/app/frontend" in compose
    assert "--server.fileWatcherType=poll" in compose


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        values[key.strip()] = value.strip()
    return values


def test_profile_env_files_define_runtime_contract() -> None:
    lite = _parse_env_file(Path(".env.profile.lite"))
    full = _parse_env_file(Path(".env.profile.full"))

    shared_keys = {
        "DIE_PROFILE",
        "INSTALL_LANGEXTRACT",
        "INSTALL_DOCLING",
        "PYTHON_EXTRAS",
        "DEEP_MODE_ENABLED",
        "INGESTION_WORKER_CONCURRENCY",
        "CLOUD_AGENT_PROVIDER",
        "LOCAL_EMBEDDING_PROVIDER",
        "CLOUD_EMBEDDING_PROVIDER",
        "EMBEDDING_ROLLOUT_MODE",
        "DOCLING_ENABLED",
        "GOOGLE_PARSER_ENABLED",
        "PARSER_ROUTING_MODE",
        "LANGEXTRACT_ENABLED",
        "REDIS_URL_LOCAL",
        "OLLAMA_BASE_URL_LOCAL",
        "REDIS_URL_DOCKER",
        "OLLAMA_BASE_URL_DOCKER",
        "DOCUMENT_INSIGHT_API_BASE_URL_DOCKER",
    }

    assert shared_keys.issubset(set(lite))
    assert shared_keys.issubset(set(full))

    assert lite["DIE_PROFILE"] == "lite"
    assert full["DIE_PROFILE"] == "full"
    assert lite["PYTHON_EXTRAS"] == "dev,ui"
    assert full["PYTHON_EXTRAS"] == "dev,ui"
    assert lite["INSTALL_LANGEXTRACT"] == "false"
    assert full["INSTALL_LANGEXTRACT"] == "false"
    assert lite["INSTALL_DOCLING"] == "false"
    assert full["INSTALL_DOCLING"] == "false"
    assert lite["DOCLING_ENABLED"] == "false"
    assert full["DOCLING_ENABLED"] == "false"
    assert lite["LANGEXTRACT_ENABLED"] == "false"
    assert full["LANGEXTRACT_ENABLED"] == "false"


def test_profile_startup_scripts_exist() -> None:
    assert Path("scripts/dev-lite-up.sh").exists()
    assert Path("scripts/dev-full-up.sh").exists()
    assert Path("scripts/docker-lite-up.sh").exists()
    assert Path("scripts/docker-full-up.sh").exists()


def test_dockerfile_installs_optional_ai_dependencies() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "libgl1" in dockerfile
    assert "libglib2.0-0" in dockerfile
    assert "requirements-ui-runtime.txt" in dockerfile
    assert "ARG INSTALL_LANGEXTRACT=false" in dockerfile
    assert "ARG INSTALL_DOCLING=false" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
