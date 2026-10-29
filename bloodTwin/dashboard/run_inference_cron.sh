#!/usr/bin/env bash
# Cron-invoked inference runner for the dashboard.
# Runs run_inference.py inside the inference Nix shell and atomically writes
# stdout -> bloodBath/logs/inference_latest.json, stderr -> inference_latest.err.
#
# Atomic write: we render to a temp file and mv it into place so the dashboard
# never reads a half-written JSON. On inference failure the previous good JSON
# is left untouched.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PUMP="${PUMP:-1539514}"

LOG_DIR="${REPO_ROOT}/bloodBath/logs"
OUT="${LOG_DIR}/inference_latest.json"
ERR="${LOG_DIR}/inference_latest.err"
TMP="${LOG_DIR}/.inference_latest.json.tmp"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

nix develop "${REPO_ROOT}#inference" --command bash -c \
  "cd '${REPO_ROOT}' && PYTHONPATH='${REPO_ROOT}' python bloodTwin/pipelines/run_inference.py --pump '${PUMP}'" \
  >"${TMP}" 2>"${ERR}"
rc=$?

if [ "${rc}" -eq 0 ] && [ -s "${TMP}" ]; then
  mv -f "${TMP}" "${OUT}"
else
  rm -f "${TMP}"
  echo "$(date -Iseconds) inference exited ${rc}; kept previous ${OUT}" >>"${ERR}"
fi
exit "${rc}"
