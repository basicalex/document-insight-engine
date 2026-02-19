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
die_require_port_available 8501 "UI"
die_export_local_runtime_endpoints

mkdir -p "${HOME}/.streamlit"
if [ ! -f "${HOME}/.streamlit/config.toml" ]; then
  printf "[browser]\ngatherUsageStats = false\n" > "${HOME}/.streamlit/config.toml"
fi

cd "${DIE_REPO_ROOT}"
printf "[profile] %s\n" "${PROFILE}"
printf "[ui] streamlit on :8501\n"

python -m streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --browser.gatherUsageStats=false
