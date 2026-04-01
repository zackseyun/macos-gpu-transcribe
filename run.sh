#!/bin/bash
# Wrapper for launchd — uses Python.app (which has Accessibility permission)
# but with venv site-packages available
PYTHON_APP="/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python"
VENV_SITE="/Users/harrymapodile/Desktop/voice-transcribe/.venv/lib/python3.13/site-packages"
SCRIPT="/Users/harrymapodile/Desktop/voice-transcribe/transcribe.py"

# Kill any existing voice-transcribe processes (prevents orphan buildup)
pkill -f "voice-transcribe/transcribe.py" 2>/dev/null
sleep 0.5

export PYTHONPATH="${VENV_SITE}:/Users/harrymapodile/Desktop/voice-transcribe:${PYTHONPATH}"
exec "$PYTHON_APP" "$SCRIPT"
