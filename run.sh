#!/bin/bash
set -euo pipefail

# Wrapper for launchd/manual restarts. Keep paths relative to this checkout so
# the app does not accidentally relaunch an old Desktop copy.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VOICE_TRANSCRIBE_PYTHON:-/Applications/Qwen Dictate.app/Contents/MacOS/Python}"
SCRIPT="${REPO_DIR}/transcribe.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/.venv/lib/python3.13/site-packages:${REPO_DIR}:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" "${SCRIPT}"
