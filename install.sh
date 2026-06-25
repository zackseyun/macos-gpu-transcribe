#!/usr/bin/env bash
# Voice Transcribe Qwen install helper for Apple Silicon macOS.

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${APP_DIR}"

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$1"; }
bold() { printf '\033[1m%s\033[0m\n' "$1"; }
fail() { printf '\033[0;31m%s\033[0m\n' "$1" >&2; exit 1; }
step() { printf '\n'; bold "==> $1"; }

bold "Voice Transcribe Qwen installer"

step "Checking system"
[[ "$(uname)" == "Darwin" ]] || fail "This app requires macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "This app requires Apple Silicon (arm64)."
command -v brew >/dev/null 2>&1 || fail "Homebrew is required. Install it from https://brew.sh and rerun ./install.sh."
green "macOS Apple Silicon detected"

step "Finding Python"
PYTHON_BIN=""
for version in 3.13 3.14; do
  candidate="/opt/homebrew/bin/python${version}"
  if [[ -x "${candidate}" ]]; then
    PYTHON_BIN="${candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  yellow "Python 3.13/3.14 was not found; installing python@3.13 with Homebrew."
  brew install python@3.13
  PYTHON_BIN="/opt/homebrew/bin/python3.13"
fi

PY_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PYTHON_APP="/opt/homebrew/opt/python@${PY_VERSION}/Frameworks/Python.framework/Versions/${PY_VERSION}/Resources/Python.app/Contents/MacOS/Python"
[[ -x "${PYTHON_APP}" ]] || fail "Could not find Python.app at ${PYTHON_APP}"
green "Using $("${PYTHON_BIN}" --version) at ${PYTHON_BIN}"

step "Creating virtual environment"
if [[ ! -d "${APP_DIR}/.venv" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${APP_DIR}/.venv"
  green "Created .venv"
else
  green ".venv already exists"
fi

step "Installing Python packages"
"${APP_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${APP_DIR}/.venv/bin/python" -m pip install -r "${APP_DIR}/requirements.txt"
green "Dependencies installed"

step "Writing local environment"
cat > "${APP_DIR}/.env.local" <<EOF
# Local machine settings for Voice Transcribe Qwen.
# This file is ignored by git.
export VOICE_TRANSCRIBE_PYTHON_APP="${PYTHON_APP}"
EOF
green "Wrote .env.local"

step "Creating app bundle"
if [[ -x "${APP_DIR}/package_app.sh" ]]; then
  "${APP_DIR}/package_app.sh"
else
  yellow "package_app.sh missing; skipping app bundle."
fi

step "macOS permissions"
cat <<EOF
Grant these manually before first use:

1. System Settings -> Keyboard
   Set "Press Globe key to" to "Do Nothing".

2. System Settings -> Privacy & Security -> Accessibility
   Add this Python.app:
   ${PYTHON_APP}

3. System Settings -> Privacy & Security -> Input Monitoring
   Add the same Python.app.

4. Microphone
   macOS should prompt on first recording. Allow access.

Optional:
- Screen Recording is only needed if you turn on Screen Assist.
EOF

step "Done"
green "Run with: ./run.sh"
green "Or open: dist/Voice Transcribe Qwen.app"
