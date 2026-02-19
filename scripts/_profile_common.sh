#!/usr/bin/env bash

set -euo pipefail

die_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

DIE_REPO_ROOT="$(die_repo_root)"

die_profile_file() {
  local profile="$1"
  printf "%s/.env.profile.%s" "${DIE_REPO_ROOT}" "${profile}"
}

die_source_file_if_present() {
  local file_path="$1"
  if [ -f "${file_path}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${file_path}"
    set +a
  fi
}

die_load_profile() {
  local profile="$1"
  local profile_file
  profile_file="$(die_profile_file "${profile}")"

  if [ ! -f "${profile_file}" ]; then
    printf "[error] missing profile file: %s\n" "${profile_file}" >&2
    exit 1
  fi

  die_source_file_if_present "${profile_file}"
  die_source_file_if_present "${DIE_REPO_ROOT}/.env"
  export DIE_PROFILE="${profile}"
}

die_python_extras_for_profile() {
  local profile="$1"
  if [ "${profile}" = "full" ]; then
    printf "dev,ui,ai"
  else
    printf "dev,ui,ai-lite"
  fi
}

die_setup_venv() {
  cd "${DIE_REPO_ROOT}"
  if [ ! -f ".venv/bin/activate" ]; then
    printf "[setup] creating .venv\n"
    if command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv .venv
    else
      python3 -m venv .venv
    fi
  fi
  # shellcheck disable=SC1091
  . .venv/bin/activate
}

die_require_profile_dependencies() {
  local profile="$1"
  local extras
  local modules

  extras="${PYTHON_EXTRAS:-$(die_python_extras_for_profile "${profile}")}"
  modules="redisvl uvicorn streamlit"
  if [ "${profile}" = "full" ]; then
    modules="${modules} langextract docling"
  else
    modules="${modules} langextract"
  fi

  if ! python - "$modules" <<'PY'
from __future__ import annotations

import importlib.util
import sys

module_names = sys.argv[1].split()
missing = [name for name in module_names if importlib.util.find_spec(name) is None]
if missing:
    print("missing modules:", ", ".join(missing))
    raise SystemExit(1)
PY
  then
    printf "[setup] installing profile dependencies (.[%s])\n" "${extras}"
    python -m pip install -e ".[${extras}]"
  fi
}

die_stop_docker_api_ui_if_running() {
  if ! command -v docker >/dev/null 2>&1; then
    return
  fi

  local running_services
  local service
  local should_stop="false"

  running_services="$(cd "${DIE_REPO_ROOT}" && docker compose ps --status running --services 2>/dev/null || true)"
  for service in ${running_services}; do
    if [ "${service}" = "api" ] || [ "${service}" = "ui" ]; then
      should_stop="true"
      break
    fi
  done

  if [ "${should_stop}" = "true" ]; then
    printf "[guard] stopping docker api/ui to avoid host port conflicts\n"
    (cd "${DIE_REPO_ROOT}" && docker compose stop api ui >/dev/null)
  fi
}

die_port_is_open() {
  local port="$1"
  python - "${port}" <<'PY'
from __future__ import annotations

import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.15)
    is_open = sock.connect_ex(("127.0.0.1", port)) == 0
raise SystemExit(0 if is_open else 1)
PY
}

die_require_port_available() {
  local port="$1"
  local label="$2"
  if die_port_is_open "${port}"; then
    printf "[error] %s port %s is already in use\n" "${label}" "${port}" >&2
    exit 1
  fi
}

die_export_local_runtime_endpoints() {
  export REDIS_URL="${REDIS_URL_LOCAL:-redis://localhost:6379/0}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL_LOCAL:-http://localhost:11434}"
  export DOCUMENT_INSIGHT_API_BASE_URL="${DOCUMENT_INSIGHT_API_BASE_URL_LOCAL:-http://localhost:8000}"
  export DATA_DIR="${DATA_DIR:-${DIE_REPO_ROOT}/data}"
}
