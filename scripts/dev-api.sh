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
die_export_local_runtime_endpoints

cd "${DIE_REPO_ROOT}"
printf "[profile] %s\n" "${PROFILE}"
printf "[api] uvicorn --reload on :8000\n"

python -m uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir src \
  --reload-dir frontend
