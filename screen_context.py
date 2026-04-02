"""Optional screen-aware transcript correction via OpenRouter + Gemini 2.5 Flash.

This module keeps the local ASR path as the primary transcript source, then
optionally uses a screenshot captured at key-release time to conservatively fix
obvious misheard terms (app labels, names, URLs, filenames, code symbols, etc.).

No API keys are stored in the repo. The OpenRouter key is fetched lazily from:
  - OPENROUTER_API_KEY env var, or
  - AWS Secrets Manager secret `cartha/moltbot/openrouter-api-key`
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("VOICE_TRANSCRIBE_SCREEN_MODEL", "google/gemini-2.5-flash")
OPENROUTER_SECRET_ID = os.getenv(
    "VOICE_TRANSCRIBE_OPENROUTER_SECRET_ID",
    "cartha/moltbot/openrouter-api-key",
)
AWS_REGION = os.getenv("VOICE_TRANSCRIBE_AWS_REGION", "us-west-2")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_SCREEN_TIMEOUT_SECONDS", "20"))
SCREENSHOT_MAX_DIMENSION = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_MAX_DIMENSION", "1800"))
SCREENSHOT_TYPE = os.getenv("VOICE_TRANSCRIBE_SCREENSHOT_TYPE", "jpg")
KEY_FETCH_COOLDOWN_SECONDS = 60.0

SYSTEM_PROMPT = """You are a conservative speech-to-text correction assistant.
You receive:
1) a raw speech transcript produced by a local ASR model
2) a screenshot from the user's screen captured at the time they stopped speaking

Your job is to use the screenshot only to correct obvious transcription mistakes.
Focus on visually grounded terms such as:
- app names, product names, proper nouns
- visible UI labels and headings
- filenames, URLs, emails, code symbols, command names
- numbers, dates, versions, and identifiers shown on screen

Rules:
- Keep the original wording whenever the screenshot does not clearly justify a change.
- Do not summarize, paraphrase, or rewrite for style.
- Do not add new ideas, sentences, or details that are not already implied by the transcript.
- Preserve sentence structure unless a small correction requires a local change.
- If you are unsure, keep the transcript exactly as written.

Return JSON only.
"""


@dataclass
class ScreenContextResult:
    text: str
    corrected: bool = False
    confidence: float | None = None
    screen_context: str = ""
    evidence_terms: list[str] | None = None
    error: str | None = None


_cached_api_key: str | None = None
_last_key_error: str | None = None
_next_key_retry_at: float = 0.0


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


def refine_transcript_with_screen_context(
    transcript: str,
    screenshot_path: str,
) -> ScreenContextResult:
    if not transcript.strip():
        return ScreenContextResult(text=transcript)

    try:
        api_key = _get_openrouter_api_key()
        data_url = _image_path_to_data_url(screenshot_path)
        raw_response = _call_openrouter(api_key=api_key, transcript=transcript, data_url=data_url)
        payload = _extract_json_payload(raw_response)

        corrected_text = str(payload.get("corrected_text") or "").strip() or transcript
        corrected_text = _validate_candidate_text(original=transcript, candidate=corrected_text)

        confidence = payload.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_value = float(confidence)
        else:
            confidence_value = None

        evidence_terms = payload.get("evidence_terms")
        if not isinstance(evidence_terms, list):
            evidence_terms = []
        evidence_terms = [str(term).strip() for term in evidence_terms if str(term).strip()]

        screen_context = str(payload.get("screen_context") or "").strip()
        changed = bool(payload.get("changed")) and corrected_text != transcript

        return ScreenContextResult(
            text=corrected_text,
            corrected=changed,
            confidence=confidence_value,
            screen_context=screen_context,
            evidence_terms=evidence_terms,
        )
    except Exception as exc:
        return ScreenContextResult(text=transcript, error=str(exc))


def _get_openrouter_api_key() -> str:
    global _cached_api_key, _last_key_error, _next_key_retry_at

    env_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("VOICE_TRANSCRIBE_OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()

    if _cached_api_key:
        return _cached_api_key

    now = time.time()
    if _last_key_error and now < _next_key_retry_at:
        raise RuntimeError(_last_key_error)

    aws_bin = None
    for candidate in (
        shutil.which("aws"),
        "/opt/homebrew/bin/aws",
        "/usr/local/bin/aws",
        "/usr/bin/aws",
    ):
        if candidate and os.path.exists(candidate):
            aws_bin = candidate
            break

    if not aws_bin:
        _remember_key_error("aws CLI not found for OpenRouter secret lookup")
        raise RuntimeError(_last_key_error or "aws CLI not found")

    try:
        secret_string = subprocess.check_output(
            [
                aws_bin,
                "secretsmanager",
                "get-secret-value",
                "--region",
                AWS_REGION,
                "--secret-id",
                OPENROUTER_SECRET_ID,
                "--query",
                "SecretString",
                "--output",
                "text",
            ],
            text=True,
            timeout=10,
        ).strip()
    except subprocess.CalledProcessError as exc:
        _remember_key_error(f"failed to fetch OpenRouter key from AWS Secrets Manager: {exc}")
        raise RuntimeError(_last_key_error or "failed to fetch OpenRouter key")
    except subprocess.TimeoutExpired:
        _remember_key_error("timed out fetching OpenRouter key from AWS Secrets Manager")
        raise RuntimeError(_last_key_error or "timed out fetching OpenRouter key")

    if not secret_string:
        _remember_key_error("empty OpenRouter secret returned from AWS Secrets Manager")
        raise RuntimeError(_last_key_error or "empty OpenRouter secret")

    try:
        parsed = json.loads(secret_string)
        if isinstance(parsed, dict):
            for key_name in ("api_key", "OPENROUTER_API_KEY", "openrouter_api_key", "key"):
                value = parsed.get(key_name)
                if isinstance(value, str) and value.strip():
                    _cached_api_key = value.strip()
                    _last_key_error = None
                    return _cached_api_key
    except json.JSONDecodeError:
        pass

    _cached_api_key = secret_string.strip()
    _last_key_error = None
    return _cached_api_key


def _remember_key_error(message: str) -> None:
    global _last_key_error, _next_key_retry_at
    _last_key_error = message
    _next_key_retry_at = time.time() + KEY_FETCH_COOLDOWN_SECONDS


def _image_path_to_data_url(path: str) -> str:
    if not os.path.exists(path):
        raise RuntimeError("screenshot file missing")

    with open(path, "rb") as handle:
        raw = handle.read()

    if not raw:
        raise RuntimeError("screenshot file was empty")

    mime_type = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _call_openrouter(*, api_key: str, transcript: str, data_url: str) -> str:
    user_prompt = (
        "Original transcript:\n"
        f"{transcript}\n\n"
        "Correct the transcript only where the screenshot provides strong visual evidence.\n"
        "Return JSON with keys corrected_text, changed, confidence, screen_context, evidence_terms.\n"
        "If unsure, set changed to false and return the original transcript exactly.\n"
    )

    body = {
        "model": OPENROUTER_MODEL,
        "temperature": 0,
        "max_tokens": 350,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.voice-transcribe",
            "X-OpenRouter-Title": "Voice Transcribe",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {details[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    raise RuntimeError("OpenRouter returned an unreadable completion payload")


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise RuntimeError("screen context response was not valid JSON")


def _validate_candidate_text(*, original: str, candidate: str) -> str:
    if not candidate.strip():
        raise RuntimeError("screen context returned an empty transcript")

    original_words = original.split()
    candidate_words = candidate.split()
    if original_words and candidate_words:
        if len(candidate_words) > max(len(original_words) * 2 + 6, 16):
            raise RuntimeError("screen context rewrite was implausibly long")
        if len(candidate_words) < max(1, len(original_words) // 2 - 3):
            raise RuntimeError("screen context rewrite was implausibly short")

    return candidate.strip()
