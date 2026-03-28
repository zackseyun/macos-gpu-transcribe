#!/bin/bash
# Wrapper for launchd — uses Python.app (which has Accessibility permission)
# but with venv site-packages available
PYTHON_APP="/opt/homebrew/Cellar/python@3.14/3.14.1/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python"
VENV_SITE="/Users/zackseyun/voice-transcribe/.venv/lib/python3.14/site-packages"
SCRIPT="/Users/zackseyun/voice-transcribe/transcribe.py"

export PYTHONPATH="${VENV_SITE}:/Users/zackseyun/voice-transcribe:${PYTHONPATH}"
exec "$PYTHON_APP" "$SCRIPT"
