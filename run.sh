#!/bin/bash
set -euo pipefail

# Wrapper for launchd/manual restarts. Keep paths relative to this checkout so
# the app does not accidentally relaunch an old Desktop copy.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VOICE_TRANSCRIBE_PYTHON:-${REPO_DIR}/.venv/bin/python3}"
SCRIPT="${REPO_DIR}/transcribe.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" "${SCRIPT}"
