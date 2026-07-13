#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_name> [key=value ...]"
  echo "Example: $0 _tbkv n_items=25"
  exit 1
fi

CONFIG_NAME="$1"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/evaluate/vitdet_vid"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}-${STAMP}.log"

echo "Writing live log to: ${LOG_FILE}"
cd "${REPO_ROOT}"

PYTHONUNBUFFERED=1 python scripts/evaluate/tbkv_vitdet_vid.py "${CONFIG_NAME}" "$@" 2>&1 | tee "${LOG_FILE}"
