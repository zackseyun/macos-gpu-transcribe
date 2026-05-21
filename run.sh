#!/bin/bash
# Launcher — uses Python.app for macOS permissions, with the repo venv on PYTHONPATH.
PYTHON_APP="/Applications/Qwen Dictate.app/Contents/MacOS/Python"
REPO_DIR="/Users/zackseyun/Documents/Codex/2026-05-20/do-you-have-acccess-to-github/macos-gpu-transcribe"
VENV_SITE="${REPO_DIR}/.venv/lib/python3.13/site-packages"
SCRIPT="${REPO_DIR}/transcribe.py"

# Kill any existing voice-transcribe processes from this checkout.
pkill -f "${SCRIPT}" 2>/dev/null
sleep 0.5

export PYTHONPATH="${VENV_SITE}:${REPO_DIR}:${PYTHONPATH}"
exec "${PYTHON_APP}" "${SCRIPT}"
