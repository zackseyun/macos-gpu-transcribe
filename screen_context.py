"""Screenshot capture helpers for screen-context prefetching."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


SCREENSHOT_MAX_DIMENSION = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_MAX_DIMENSION", "1800"))
SCREENSHOT_TYPE = os.getenv("VOICE_TRANSCRIBE_SCREENSHOT_TYPE", "jpg")


def is_feature_enabled(default: bool = False) -> bool:
    raw = os.getenv("VOICE_TRANSCRIBE_SCREEN_CONTEXT")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def capture_screen_snapshot() -> str:
    """Capture a compressed screenshot and return the local file path."""
    screencapture_bin = shutil.which("screencapture") or "/usr/sbin/screencapture"
    if not os.path.exists(screencapture_bin):
        raise RuntimeError("macOS screencapture tool not found")

    fd, path = tempfile.mkstemp(suffix=f".{SCREENSHOT_TYPE}")
    os.close(fd)

    try:
        result = subprocess.run(
            [screencapture_bin, "-x", "-t", SCREENSHOT_TYPE, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(stderr or "screencapture failed")

        sips_bin = shutil.which("sips") or "/usr/bin/sips"
        if os.path.exists(sips_bin):
            subprocess.run(
                [sips_bin, "-Z", str(SCREENSHOT_MAX_DIMENSION), path],
                capture_output=True,
                timeout=10,
            )

        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
