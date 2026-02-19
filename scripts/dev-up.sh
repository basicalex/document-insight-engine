#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/_profile_common.sh"

PROFILE="${1:-${DIE_PROFILE:-lite}}"

die_load_profile "${PROFILE}"
die_setup_venv
die_require_profile_dependencies "${PROFILE}"
die_stop_docker_api_ui_if_running
die_require_port_available 8000 "API"
die_require_port_available 8501 "UI"

cd "${DIE_REPO_ROOT}"
docker compose up -d redis ollama
die_export_local_runtime_endpoints

mkdir -p .aoc/logs
API_LOG=".aoc/logs/dev-api-${PROFILE}.log"

printf "[profile] %s\n" "${PROFILE}"
printf "[infra] redis+ollama are up via docker\n"
printf "[api] logs -> %s\n" "${API_LOG}"

python -m uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir src \
  --reload-dir frontend \
  > "${API_LOG}" 2>&1 &
API_PID=$!

cleanup() {
  if kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python -m streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --browser.gatherUsageStats=false
