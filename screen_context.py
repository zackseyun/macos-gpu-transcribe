"""Fast frontmost-window screenshot capture + local text extraction."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from AppKit import NSWorkspace
from Foundation import NSURL
import Quartz
import Vision


SCREENSHOT_MAX_DIMENSION = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_MAX_DIMENSION", "1800"))
SCREENSHOT_TYPE = os.getenv("VOICE_TRANSCRIBE_SCREENSHOT_TYPE", "jpg")
OCR_RECOGNITION_LEVEL = os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_LEVEL", "fast").strip().lower()
OCR_MAX_LINES = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_MAX_LINES", "20"))
OCR_MAX_TERMS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_MAX_TERMS", "18"))
OCR_MAX_CONTEXT_CHARS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_CONTEXT_MAX_CHARS", "320"))

URL_RE = re.compile(r"(?:https?://|www\.)\S+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/\S*)?")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{1,}")
GENERIC_UI_TERMS = {
    "file",
    "edit",
    "view",
    "window",
    "help",
    "shell",
    "session",
    "scripts",
    "profiles",
    "format",
    "conversation",
    "messages",
}


@dataclass
class ScreenSnapshot:
    path: str
    app_name: str = ""
    window_title: str = ""
    window_id: int | None = None


@dataclass
class ScreenTextContext:
    glossary: str = ""
    terms: list[str] | None = None
    lines: list[str] | None = None
    app_name: str = ""
    window_title: str = ""
    recognition_time_ms: int = 0
    error: str | None = None


def is_feature_enabled(default: bool = False) -> bool:
    raw = os.getenv("VOICE_TRANSCRIBE_SCREEN_CONTEXT")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def capture_frontmost_window_snapshot() -> ScreenSnapshot:
    info = _get_frontmost_window_info()
    path = _capture_screen_snapshot(window_id=info.get("window_id"))
    return ScreenSnapshot(
        path=path,
        app_name=info.get("app_name", ""),
        window_title=info.get("window_title", ""),
        window_id=info.get("window_id"),
    )


def extract_screen_text_context(
    screenshot_path: str,
    *,
    app_name: str = "",
    window_title: str = "",
) -> ScreenTextContext:
    started = time.time()
    try:
        lines = _recognize_text_lines(screenshot_path)
        terms = _extract_salient_terms(lines, app_name=app_name, window_title=window_title)
        glossary = _build_glossary(
            terms=terms,
            lines=lines,
            app_name=app_name,
            window_title=window_title,
        )
        return ScreenTextContext(
            glossary=glossary,
            terms=terms,
            lines=lines,
            app_name=app_name,
            window_title=window_title,
            recognition_time_ms=int((time.time() - started) * 1000),
        )
    except Exception as exc:
        return ScreenTextContext(
            app_name=app_name,
            window_title=window_title,
            error=str(exc),
            recognition_time_ms=int((time.time() - started) * 1000),
        )


def _get_frontmost_window_info():
    workspace = NSWorkspace.sharedWorkspace()
    app = workspace.frontmostApplication()
    app_name = str(app.localizedName() or "") if app else ""
    pid = int(app.processIdentifier()) if app else None

    selected = None
    if pid is not None:
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
        ) or []
        for window in windows:
            if int(window.get("kCGWindowOwnerPID", -1)) != pid:
                continue
            if int(window.get("kCGWindowLayer", 1)) != 0:
                continue
            bounds = window.get("kCGWindowBounds", {}) or {}
            if bounds.get("Width", 0) < 300 or bounds.get("Height", 0) < 160:
                continue
            selected = window
            break

    return {
        "app_name": app_name,
        "window_title": str((selected or {}).get("kCGWindowName") or ""),
        "window_id": int((selected or {}).get("kCGWindowNumber")) if selected else None,
    }


def _capture_screen_snapshot(*, window_id: int | None = None) -> str:
    screencapture_bin = shutil.which("screencapture") or "/usr/sbin/screencapture"
    if not os.path.exists(screencapture_bin):
        raise RuntimeError("macOS screencapture tool not found")

    fd, path = tempfile.mkstemp(suffix=f".{SCREENSHOT_TYPE}")
    os.close(fd)

    try:
        cmd = [screencapture_bin, "-x", "-t", SCREENSHOT_TYPE]
        if window_id:
            cmd.extend(["-l", str(window_id)])
        cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
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


def _recognize_text_lines(screenshot_path: str) -> list[str]:
    url = NSURL.fileURLWithPath_(screenshot_path)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    level = (
        Vision.VNRequestTextRecognitionLevelAccurate
        if OCR_RECOGNITION_LEVEL == "accurate"
        else Vision.VNRequestTextRecognitionLevelFast
    )
    request.setRecognitionLevel_(level)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")

    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        text = _normalize_line(text)
        if not text or not _looks_reasonably_clean(text):
            continue
        lines.append(text)
        if len(lines) >= OCR_MAX_LINES:
            break

    deduped = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return deduped


def _normalize_line(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def _looks_reasonably_clean(text: str) -> bool:
    if len(text) < 2:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    weird = sum(not (ch.isalnum() or ch in " .,:;/_-()[]{}@#%+&*'\"") for ch in text)
    if alpha + digits < 2:
        return False
    if weird > max(4, len(text) // 6):
        return False
    return True


def _extract_salient_terms(lines: list[str], *, app_name: str, window_title: str) -> list[str]:
    ordered = []
    seen = set()

    def add(term: str):
        cleaned = term.strip(" ,;:()[]{}<>\"'")
        if len(cleaned) < 2:
            return
        key = cleaned.lower()
        if key in seen:
            return
        if key in GENERIC_UI_TERMS:
            return
        seen.add(key)
        ordered.append(cleaned)

    if app_name:
        add(app_name)
    if window_title and window_title.lower() not in GENERIC_UI_TERMS:
        add(window_title)

    for line in lines:
        for url in URL_RE.findall(line):
            add(url)

        if _looks_useful_phrase(line):
            add(line)

        for token in TOKEN_RE.findall(line):
            if _looks_useful_token(token):
                add(token)

        if len(ordered) >= OCR_MAX_TERMS:
            break

    return ordered[:OCR_MAX_TERMS]


def _looks_useful_phrase(text: str) -> bool:
    lower = text.lower()
    if lower in GENERIC_UI_TERMS:
        return False
    if len(text) > 64:
        return False
    words = text.split()
    if len(words) < 2:
        return False
    if any(ch in text for ch in "/._-:@"):
        return True
    capitals = sum(word[:1].isupper() for word in words if word)
    return capitals >= 2


def _looks_useful_token(token: str) -> bool:
    lower = token.lower()
    if lower in GENERIC_UI_TERMS or len(token) < 3:
        return False
    if any(ch in token for ch in "/._-:@"):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if token[:1].isupper() and any(ch.isupper() for ch in token[1:]):
        return True
    return False


def _build_glossary(*, terms: list[str], lines: list[str], app_name: str, window_title: str) -> str:
    parts = []
    if app_name:
        parts.append(f"App: {app_name}")
    if window_title:
        parts.append(f"Window: {window_title}")
    if lines:
        parts.append("Visible text: " + " • ".join(lines[:OCR_MAX_LINES]))
    elif terms:
        parts.append("Visible terms: " + ", ".join(terms[:OCR_MAX_TERMS]))

    glossary = " | ".join(part for part in parts if part).strip()
    if len(glossary) > OCR_MAX_CONTEXT_CHARS:
        glossary = glossary[: OCR_MAX_CONTEXT_CHARS - 1].rstrip() + "…"
    return glossary
