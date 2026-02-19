#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/_profile_common.sh"

PROFILE="${1:-${DIE_PROFILE:-lite}}"

die_load_profile "${PROFILE}"

export DIE_PROFILE_ENV_FILE=".env.profile.${PROFILE}"
export INSTALL_DOCLING="${INSTALL_DOCLING:-false}"

cd "${DIE_REPO_ROOT}"
printf "[profile] %s\n" "${PROFILE}"
printf "[compose] using %s\n" "${DIE_PROFILE_ENV_FILE}"

docker compose up -d --build
