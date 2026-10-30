#!/usr/bin/env bash
# Launch the bloodBender dashboard on port 8765 via the inference Nix shell.
# Usage:  ./bloodTwin/dashboard/serve.sh
set -euo pipefail

# Resolve the repo root from this script's location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PORT="${DASHBOARD_PORT:-8765}"
HOST="${DASHBOARD_HOST:-127.0.0.1}"

cd "${REPO_ROOT}"
exec nix develop "${REPO_ROOT}#inference" --command bash -c \
  "cd '${REPO_ROOT}' && PYTHONPATH='${REPO_ROOT}' DASHBOARD_HOST='${HOST}' DASHBOARD_PORT='${PORT}' python bloodTwin/dashboard/server.py"
