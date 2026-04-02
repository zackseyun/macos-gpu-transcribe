"""Local screenshot capture + OCR context extraction for low-latency ASR biasing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from Foundation import NSURL
import Vision


SCREENSHOT_MAX_DIMENSION = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_MAX_DIMENSION", "1800"))
SCREENSHOT_TYPE = os.getenv("VOICE_TRANSCRIBE_SCREENSHOT_TYPE", "jpg")
OCR_RECOGNITION_LEVEL = os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_LEVEL", "fast").strip().lower()
OCR_MAX_LINES = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_MAX_LINES", "12"))
OCR_MAX_TERMS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_MAX_TERMS", "32"))
OCR_MAX_CONTEXT_CHARS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_OCR_MAX_CONTEXT_CHARS", "320"))


URL_RE = re.compile(r"(?:https?://|www\.)\S+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/\S*)?")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{1,}")


@dataclass
class ScreenTextContext:
    glossary: str = ""
    lines: list[str] | None = None
    terms: list[str] | None = None
    recognition_time_ms: int = 0
    error: str | None = None


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


def extract_screen_text_context(screenshot_path: str) -> ScreenTextContext:
    """Run local OCR on a screenshot and build a compact glossary string."""
    started = time.time()
    try:
        lines = _recognize_text_lines(screenshot_path)
        terms = _extract_salient_terms(lines)
        glossary = _build_glossary(terms=terms, lines=lines)
        elapsed_ms = int((time.time() - started) * 1000)
        return ScreenTextContext(
            glossary=glossary,
            lines=lines,
            terms=terms,
            recognition_time_ms=elapsed_ms,
        )
    except Exception as exc:
        return ScreenTextContext(error=str(exc), recognition_time_ms=int((time.time() - started) * 1000))


def _recognize_text_lines(screenshot_path: str) -> list[str]:
    if not os.path.exists(screenshot_path):
        raise RuntimeError("screenshot file missing")

    url = NSURL.fileURLWithPath_(screenshot_path)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    recognition_level = (
        Vision.VNRequestTextRecognitionLevelAccurate
        if OCR_RECOGNITION_LEVEL == "accurate"
        else Vision.VNRequestTextRecognitionLevelFast
    )
    request.setRecognitionLevel_(recognition_level)
    request.setUsesLanguageCorrection_(False)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")

    results = request.results() or []
    lines: list[str] = []
    for observation in results:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        if not text:
            continue
        lines.append(_normalize_line(text))
        if len(lines) >= OCR_MAX_LINES:
            break

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return deduped


def _normalize_line(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_salient_terms(lines: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(term: str):
        cleaned = term.strip(" ,;:()[]{}<>\"'")
        if len(cleaned) < 2:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(cleaned)

    for line in lines:
        for match in URL_RE.findall(line):
            add(match)

        for token in TOKEN_RE.findall(line):
            if _looks_salient_token(token):
                add(token)

        if len(line) <= 80:
            add(line)

    return ordered[:OCR_MAX_TERMS]


def _looks_salient_token(token: str) -> bool:
    if len(token) < 3:
        return False
    if any(ch in token for ch in "./:_-"):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if token[:1].isupper() and any(ch.isupper() for ch in token[1:]):
        return True
    if token.isupper() and len(token) <= 12:
        return True
    return False


def _build_glossary(*, terms: list[str], lines: list[str]) -> str:
    if not terms and not lines:
        return ""

    parts: list[str] = []
    if terms:
        parts.append("Visible terms: " + ", ".join(terms[:OCR_MAX_TERMS]))
    elif lines:
        parts.append("Visible text: " + " | ".join(lines[:6]))

    glossary = " ".join(parts).strip()
    if len(glossary) > OCR_MAX_CONTEXT_CHARS:
        glossary = glossary[: OCR_MAX_CONTEXT_CHARS - 1].rstrip() + "…"
    return glossary
