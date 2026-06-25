#!/usr/bin/env bash
# Launch Voice Transcribe using Homebrew's Python.app.
#
# Python.app is important on macOS because Accessibility, Input Monitoring, and
# Microphone permissions are granted to the app bundle that owns the process.

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${APP_DIR}/.venv/bin/python"

if [[ -f "${APP_DIR}/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "${APP_DIR}/.env.local"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing virtual environment. Run ./install.sh first." >&2
  exit 1
fi

PY_VERSION="$("${VENV_PYTHON}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

PYTHON_APP="${VOICE_TRANSCRIBE_PYTHON_APP:-}"
if [[ -z "${PYTHON_APP}" ]]; then
  PYTHON_APP="/opt/homebrew/opt/python@${PY_VERSION}/Frameworks/Python.framework/Versions/${PY_VERSION}/Resources/Python.app/Contents/MacOS/Python"
fi

if [[ ! -x "${PYTHON_APP}" ]]; then
  echo "Could not find Python.app at: ${PYTHON_APP}" >&2
  echo "Set VOICE_TRANSCRIBE_PYTHON_APP to the Python.app executable that has macOS permissions." >&2
  exit 1
fi

VENV_SITE="$("${VENV_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

# Prevent duplicate menu bar instances for this checkout.
pkill -f "${APP_DIR}/transcribe.py" 2>/dev/null || true
sleep 0.5

export PYTHONPATH="${VENV_SITE}:${APP_DIR}:${PYTHONPATH:-}"
exec "${PYTHON_APP}" "${APP_DIR}/transcribe.py"
