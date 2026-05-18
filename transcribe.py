#!/Users/zackseyun/voice-transcribe/.venv/bin/python3
"""
Voice Transcribe — hold Fn to record, release to transcribe & paste.

Menu bar app using local ASR with optional screenshot-aware context.

Architecture:
  - Main process: rumps menu bar app + always-on audio stream
  - Key monitor subprocess: Quartz CGEvent tap for Fn detection
  - Transcription worker subprocess: loads local ASR model, transcribes on demand
  - Optional screen assist: prefetches frontmost-window screenshots and injects fast local text context into ASR

IMPORTANT — CoreAudio threading constraints:
  The audio stream (sounddevice/PortAudio) is opened at startup and is NEVER
  stopped or closed. This is intentional. PortAudio's Pa_StopStream and
  Pa_AbortStream both call AudioDeviceStop, which tries to acquire CoreAudio's
  internal HALB_Mutex. If the audio callback is running (which it is every ~10ms),
  the callback thread already holds that mutex → DEADLOCK. This happens regardless
  of which thread calls stop/abort. The safe refresh path is to open a new input
  stream and retire the old object until process exit, never stop it in-place.
  Recording is controlled by toggling a boolean flag that the active callback checks.
"""

import json
import multiprocessing
import os
import queue
import re
import shutil
import shlex
import threading
import subprocess
import tempfile
import time
import wave
import fcntl
from datetime import datetime
from pathlib import Path

import numpy as np
import rumps
import sounddevice as sd

# Subprocess target functions live in separate files (no rumps/sd imports)
import key_monitor
import transcribe_worker
from format_text import format_transcription
from hud_overlay import get_controller as get_hud_controller
from main_window import get_controller as get_main_window_controller
from screen_context import (
    capture_frontmost_window_snapshot,
    extract_screen_text_context,
)

# -- Constants --
SAMPLE_RATE = 16000
CHANNELS = 1
HISTORY_FILE = Path(__file__).parent / "history.json"
GLOSSARY_MEMORY_FILE = Path(__file__).parent / "screen_glossary_memory.json"
SETTINGS_FILE = Path(__file__).parent / "settings.json"
LAST_RECORDING_FILE = Path(__file__).parent / "last_recording.wav"
FAILED_RECORDINGS_DIR = Path(__file__).parent / "failed_recordings"
LOCK_FILE = Path("/tmp/voice-transcribe.lock")
MAX_HISTORY = 100
ICON_IDLE = "🎙"
ICON_RECORDING = "🔴"
ICON_PROCESSING = "⏳"

# Thermal state labels and icons (NSProcessInfoThermalState values)
THERMAL_STATES = {
    0: ("Normal", "✅"),
    1: ("Fair", "⚠️"),
    2: ("Serious", "🔥"),
    3: ("Critical", "🛑"),
}
THERMAL_ICON_SUFFIX = {
    0: "",       # Normal — no indicator
    1: "",       # Fair — no indicator (minor)
    2: "🔥",    # Serious — show fire on menu bar icon
    3: "🛑",    # Critical — show stop on menu bar icon
}
TRANSCRIBE_TIMEOUT = float(os.getenv("VOICE_TRANSCRIBE_TIMEOUT_SECONDS", "900"))
PASTEBOARD_PRE_PASTE_DELAY = float(os.getenv("VOICE_TRANSCRIBE_PASTEBOARD_PRE_PASTE_DELAY", "0.03"))
PASTEBOARD_RESTORE_DELAY = float(os.getenv("VOICE_TRANSCRIBE_PASTEBOARD_RESTORE_DELAY", "0.10"))
SCREEN_CONTEXT_PREFETCH_INTERVAL = float(
    os.getenv("VOICE_TRANSCRIBE_SCREEN_PREFETCH_INTERVAL_SECONDS", "5")
)
SCREEN_CONTEXT_MAX_AGE = float(
    os.getenv("VOICE_TRANSCRIBE_SCREEN_MAX_AGE_SECONDS", "15")
)
SCREEN_CONTEXT_RECORDING_REFRESH_INTERVAL = float(
    os.getenv("VOICE_TRANSCRIBE_SCREEN_RECORDING_REFRESH_INTERVAL_SECONDS", "2")
)
GLOSSARY_MEMORY_MIN_COUNT = int(os.getenv("VOICE_TRANSCRIBE_GLOSSARY_MEMORY_MIN_COUNT", "2"))
GLOSSARY_MEMORY_TOP_TERMS = int(os.getenv("VOICE_TRANSCRIBE_GLOSSARY_MEMORY_TOP_TERMS", "12"))
GLOSSARY_MEMORY_MAX_TERMS = int(os.getenv("VOICE_TRANSCRIBE_GLOSSARY_MEMORY_MAX_TERMS", "256"))
SCREEN_CONTEXT_MAX_CHARS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_CONTEXT_MAX_CHARS", "320"))
RELEASE_DEBOUNCE_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_RELEASE_DEBOUNCE_SECONDS", "0.2"))

# Audio health / mic refresh. We still avoid PortAudio stop/abort/close because
# those can deadlock CoreAudio while the callback thread is active. A refresh
# therefore opens a new input stream and retires the old object until process
# exit. This is intentionally rare and cooldown-gated.
AUDIO_CALLBACK_STALE_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_CALLBACK_STALE_SECONDS", "2.5"))
AUDIO_HEALTH_CHECK_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_HEALTH_CHECK_SECONDS", "5"))
AUDIO_REFRESH_COOLDOWN_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_REFRESH_COOLDOWN_SECONDS", "4"))
AUDIO_MAX_RETIRED_STREAMS = int(os.getenv("VOICE_TRANSCRIBE_AUDIO_MAX_RETIRED_STREAMS", "3"))
AUDIO_RELAUNCH_ON_REFRESH_CAP = os.getenv(
    "VOICE_TRANSCRIBE_AUDIO_RELAUNCH_ON_REFRESH_CAP", "true"
).strip().lower() not in {"0", "false", "off", "no"}
AUDIO_RECORDING_WATCHDOG_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_RECORDING_WATCHDOG_SECONDS", "0.35"))
AUDIO_RECORDING_FLAT_REFRESH_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_RECORDING_FLAT_REFRESH_SECONDS", "1.6"))
AUDIO_RECORDING_MIN_ACTIVE_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_RECORDING_MIN_ACTIVE_SECONDS", "0.15"))
AUDIO_RELAUNCH_DELAY_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_RELAUNCH_DELAY_SECONDS", "0.8"))
AUDIO_RELAUNCH_LOG_FILE = Path(os.getenv("VOICE_TRANSCRIBE_AUDIO_RELAUNCH_LOG_FILE", "/tmp/voice-transcribe.log"))
AUDIO_DEAD_INPUT_MIN_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_DEAD_INPUT_MIN_SECONDS", "0.90"))
AUDIO_DEAD_INPUT_MAX_RMS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_DEAD_INPUT_MAX_RMS", "0.008"))
AUDIO_DEAD_INPUT_ACTIVE_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_AUDIO_DEAD_INPUT_ACTIVE_SECONDS", "0.12"))
INPUT_VOLUME_FORCE_ON_RECORDING = os.getenv(
    "VOICE_TRANSCRIBE_FORCE_INPUT_VOLUME", "true"
).strip().lower() not in {"0", "false", "off", "no"}
INPUT_VOLUME_OSASCRIPT_TIMEOUT_SECONDS = float(
    os.getenv("VOICE_TRANSCRIBE_INPUT_VOLUME_TIMEOUT_SECONDS", "1.5")
)

BACKGROUND_WARM_INTERVAL_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_WARM_PING_SECONDS", "240"))
BACKGROUND_WARM_LOW_POWER_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_WARM_PING_LOW_POWER_SECONDS", "900"))
BACKGROUND_WARM_LOW_BATTERY_PERCENT = int(os.getenv("VOICE_TRANSCRIBE_WARM_LOW_BATTERY_PERCENT", "25"))

# Static vocabulary opt-in — prepended to the Cohere decoder prompt. Disabled
# by default because biasing the decoder toward "Claude"/"Cartha" tokens on
# every transcription causes degeneration on low-confidence audio (the model
# starts cycling "Claudia, Claus, Claire, Claw, Clap, …"). Brand/homophone
# corrections are now handled deterministically in format_text.py instead.
# Set "vocabulary" in settings.json to opt back in for specialised dictation.
STATIC_VOCABULARY_DEFAULT = ""

DEFAULT_MODEL_MODE = "cohere"
MODEL_LABELS = {
    "granite": "Granite Speech 4.1 NAR",
    "cohere": "Cohere Transcribe 2B",
    "fast": "Qwen3-ASR 0.6B",
    "accurate": "Qwen3-ASR 1.7B",
}
MENU_MODEL_MODES = ("cohere", "granite")

# Silence gate — if the loudest 200ms window in the recording has RMS below
# this threshold, the audio is treated as silent and no transcription runs.
# Prevents Whisper-style ASR hallucinations on silent/near-silent input
# ("Thank you.", ".", "you", etc.). Tuned for MacBook Pro internal mic noise
# floor (~0.002-0.005 RMS); real speech is typically 0.02+.
SILENCE_RMS_THRESHOLD = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_RMS", "0.008"))
SILENCE_WINDOW_SECONDS = 0.2
SILENCE_FRAME_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_FRAME_SECONDS", "0.03"))
SILENCE_ACTIVE_RMS_THRESHOLD = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_ACTIVE_RMS", "0.012"))
SILENCE_MIN_RECORDING_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_MIN_RECORDING_SECONDS", "0.25"))
SILENCE_MIN_ACTIVE_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_MIN_ACTIVE_SECONDS", "0.28"))
SILENCE_LOW_CONFIDENCE_FULL_RMS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_LOW_FULL_RMS", "0.010"))
SILENCE_LOW_CONFIDENCE_MAX_RMS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_LOW_MAX_RMS", "0.040"))
SILENCE_LOW_CONFIDENCE_ACTIVE_RATIO = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_LOW_ACTIVE_RATIO", "0.16"))
SILENCE_SHORT_BLIP_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_SHORT_BLIP_SECONDS", "0.50"))
SILENCE_SHORT_BLIP_MAX_RMS = float(os.getenv("VOICE_TRANSCRIBE_SILENCE_SHORT_BLIP_MAX_RMS", "0.050"))

# Known ASR hallucination outputs on silent/near-silent audio. These are the
# most common ones Whisper-family and Cohere Transcribe emit from training
# data contamination (YouTube captions). Normalized lowercase, no trailing
# punctuation. If the transcription collapses to one of these, drop it.
ASR_HALLUCINATIONS = frozenset({
    "",
    "you",
    "thank you",
    "thanks",
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you so much",
    "thank you very much",
    "bye",
    "bye bye",
    "okay",
    "ok",
    "mm",
    "mhm",
    "uh",
    "um",
    ".",
    "...",
    "♪",
    "♪♪",
    "[music]",
    "[silence]",
    "(silence)",
    "!",
    "!!",
    "!!!",
    "! !",
    "! ! !",
})


def _env_flag_value(raw, default=False):
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "off", "no"}


def _env_flag(name, default=False):
    return _env_flag_value(os.getenv(name), default=default)


def _parse_input_volume_osascript_output(output):
    """Parse osascript output as (previous_volume, current_volume).

    The AppleScript returns comma-separated integers, but this parser is
    intentionally tolerant so failures can be logged instead of crashing the
    recording path.
    """
    numbers = re.findall(r"\d+", output or "")
    if len(numbers) < 2:
        return None, None
    try:
        previous = max(0, min(100, int(numbers[0])))
        current = max(0, min(100, int(numbers[1])))
        return previous, current
    except (TypeError, ValueError):
        return None, None


def _input_volume_should_log(previous, current, error=None):
    """Return True when the volume bump path should emit a log line."""
    if error:
        return True
    if previous is None or current is None:
        return True
    return previous != current


def _longest_true_run_seconds(mask, frame_seconds):
    longest = 0
    current = 0
    for active in mask:
        if bool(active):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * frame_seconds)


def _analyze_audio_volume(audio):
    """Return cheap volume stats used before sending audio to ASR.

    Max RMS alone is too optimistic for silent clips because one key click,
    mic pop, or CoreAudio blip can cross the old threshold and send the app
    into Granite/Cohere fallback work. These stats also look for sustained
    active audio so true silence/no-volume releases can unlock immediately.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return {
            "duration": 0.0,
            "peak": 0.0,
            "full_rms": 0.0,
            "max_rms": 0.0,
            "p95_rms": 0.0,
            "active_seconds": 0.0,
            "active_ratio": 0.0,
            "longest_active_seconds": 0.0,
        }

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    duration = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio)))
    full_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

    window_samples = max(1, int(SILENCE_WINDOW_SECONDS * SAMPLE_RATE))
    if len(audio) >= window_samples:
        num_windows = len(audio) // window_samples
        trimmed = audio[: num_windows * window_samples]
        window_rms = np.sqrt(
            np.mean(trimmed.reshape(num_windows, window_samples) ** 2, axis=1)
        )
        max_rms = float(np.max(window_rms)) if num_windows > 0 else full_rms
        p95_rms = float(np.percentile(window_rms, 95)) if num_windows > 0 else full_rms
    else:
        max_rms = full_rms
        p95_rms = full_rms

    frame_samples = max(1, int(SILENCE_FRAME_SECONDS * SAMPLE_RATE))
    if len(audio) >= frame_samples:
        num_frames = len(audio) // frame_samples
        frame_audio = audio[: num_frames * frame_samples]
        frame_rms = np.sqrt(
            np.mean(frame_audio.reshape(num_frames, frame_samples) ** 2, axis=1)
        )
        active_mask = frame_rms >= SILENCE_ACTIVE_RMS_THRESHOLD
        active_seconds = float(np.sum(active_mask) * SILENCE_FRAME_SECONDS)
        active_ratio = float(np.mean(active_mask)) if num_frames > 0 else 0.0
        longest_active_seconds = _longest_true_run_seconds(
            active_mask, SILENCE_FRAME_SECONDS
        )
    else:
        active = full_rms >= SILENCE_ACTIVE_RMS_THRESHOLD
        active_seconds = duration if active else 0.0
        active_ratio = 1.0 if active else 0.0
        longest_active_seconds = active_seconds

    return {
        "duration": float(duration),
        "peak": peak,
        "full_rms": full_rms,
        "max_rms": max_rms,
        "p95_rms": p95_rms,
        "active_seconds": active_seconds,
        "active_ratio": active_ratio,
        "longest_active_seconds": longest_active_seconds,
    }


def _no_volume_skip_reason(stats):
    """Return a reason when a recording should end without ASR, else None."""
    duration = stats["duration"]
    max_rms = stats["max_rms"]
    full_rms = stats["full_rms"]
    active_seconds = stats["active_seconds"]
    longest_active_seconds = stats["longest_active_seconds"]

    if duration <= 0:
        return "no audio samples"
    if duration < SILENCE_MIN_RECORDING_SECONDS:
        return f"too short ({duration:.2f}s < {SILENCE_MIN_RECORDING_SECONDS:.2f}s)"
    if max_rms < SILENCE_RMS_THRESHOLD:
        return f"max_rms={max_rms:.4f} < {SILENCE_RMS_THRESHOLD:.4f}"
    if (
        duration < SILENCE_SHORT_BLIP_SECONDS
        and active_seconds < 0.12
        and max_rms < SILENCE_SHORT_BLIP_MAX_RMS
    ):
        return (
            f"short blip only (active={active_seconds:.2f}s, "
            f"max_rms={max_rms:.4f})"
        )
    if (
        duration >= 0.75
        and active_seconds < SILENCE_MIN_ACTIVE_SECONDS
        and longest_active_seconds < SILENCE_MIN_ACTIVE_SECONDS
        and full_rms < SILENCE_LOW_CONFIDENCE_FULL_RMS
        and max_rms < SILENCE_LOW_CONFIDENCE_MAX_RMS
    ):
        return (
            f"no sustained volume (active={active_seconds:.2f}s, "
            f"longest={longest_active_seconds:.2f}s, full_rms={full_rms:.4f})"
        )
    return None


def _is_low_confidence_no_volume(stats):
    """True for clips where ASR fallback should not keep the app busy."""
    return (
        (
            stats["full_rms"] < SILENCE_LOW_CONFIDENCE_FULL_RMS
            and stats["max_rms"] < SILENCE_LOW_CONFIDENCE_MAX_RMS
            and stats["active_seconds"] < max(0.45, SILENCE_MIN_ACTIVE_SECONDS)
        )
        or (
            stats["duration"] >= 1.0
            and stats["full_rms"] < SILENCE_LOW_CONFIDENCE_FULL_RMS
            and stats["max_rms"] < SILENCE_LOW_CONFIDENCE_MAX_RMS
            and stats["active_ratio"] < SILENCE_LOW_CONFIDENCE_ACTIVE_RATIO
        )
    )


def _dead_input_refresh_reason(stats):
    """Return a mic-refresh reason when a real hold captured almost no signal.

    A quick accidental tap should just end. A long hold with flat audio is more
    suspicious: the mic stream may have wedged, the selected input may have
    disappeared, or CoreAudio may be delivering near-zero samples. In that case
    we refresh the input stream before the next recording.
    """
    duration = stats.get("duration", 0.0)
    if duration < AUDIO_DEAD_INPUT_MIN_SECONDS:
        return None

    max_rms = stats.get("max_rms", 0.0)
    full_rms = stats.get("full_rms", 0.0)
    active_seconds = stats.get("active_seconds", 0.0)
    longest_active_seconds = stats.get("longest_active_seconds", 0.0)

    if max_rms < AUDIO_DEAD_INPUT_MAX_RMS:
        return (
            f"flat input for {duration:.1f}s "
            f"(max_rms={max_rms:.4f} < {AUDIO_DEAD_INPUT_MAX_RMS:.4f})"
        )
    if (
        active_seconds < AUDIO_DEAD_INPUT_ACTIVE_SECONDS
        and longest_active_seconds < AUDIO_DEAD_INPUT_ACTIVE_SECONDS
        and full_rms < SILENCE_LOW_CONFIDENCE_FULL_RMS
        and max_rms < SILENCE_LOW_CONFIDENCE_MAX_RMS
    ):
        return (
            f"no sustained input for {duration:.1f}s "
            f"(active={active_seconds:.2f}s, full_rms={full_rms:.4f})"
        )
    return None


def _parse_pmset_battery_output(output):
    """Parse `pmset -g batt` enough for low-power diagnostics/logging."""
    text = output or ""
    source = "unknown"
    source_match = re.search(r"Now drawing from '([^']+)'", text)
    if source_match:
        source = source_match.group(1)

    percent = None
    percent_match = re.search(r"(\d{1,3})%;", text)
    if percent_match:
        try:
            percent = int(percent_match.group(1))
        except ValueError:
            percent = None

    return {"source": source, "percent": percent}


def _device_name_matches(device, preferred):
    if not preferred:
        return False
    try:
        return preferred.lower() in str(device.get("name", "")).lower()
    except Exception:
        return False


def _is_input_device(device):
    try:
        return int(device.get("max_input_channels", 0)) > 0
    except Exception:
        return False


def _force_macos_input_volume_to_max(
    timeout_seconds=INPUT_VOLUME_OSASCRIPT_TIMEOUT_SECONDS,
    runner=subprocess.run,
):
    """Best-effort macOS input-volume bump before recording starts.

    Uses AppleScript's public `set volume input volume 100` path. The caller
    treats every error as non-fatal so dictation can still start if macOS is
    slow, osascript is unavailable, or permissions behave unexpectedly.
    """
    script = """
set previousInputVolume to input volume of (get volume settings)
if previousInputVolume is not equal to 100 then
    set volume input volume 100
end if
set currentInputVolume to input volume of (get volume settings)
return (previousInputVolume as text) & "," & (currentInputVolume as text)
""".strip()
    result = {"previous": None, "current": None, "error": None}
    try:
        completed = runner(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=max(0.05, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"osascript timed out after {timeout_seconds:.2f}s"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result

    previous, current = _parse_input_volume_osascript_output(completed.stdout)
    result["previous"] = previous
    result["current"] = current
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        result["error"] = stderr or f"osascript exited {completed.returncode}"
    elif previous is None or current is None:
        output = (completed.stdout or "").strip()
        result["error"] = f"unparseable osascript output: {output!r}"
    elif current != 100:
        result["error"] = f"input volume remained at {current}"
    return result


def _build_self_relaunch_script(repo_dir, log_path, delay_seconds):
    """Build a relaunch command that preserves the repo's Python environment."""
    repo_dir = Path(repo_dir)
    log_path = Path(log_path)
    launcher = repo_dir / "run.sh"
    venv_python = repo_dir / ".venv" / "bin" / "python3"
    script = repo_dir / "transcribe.py"

    if launcher.exists() and os.access(launcher, os.X_OK):
        exec_part = f"exec {shlex.quote(str(launcher))}"
    elif venv_python.exists() and os.access(venv_python, os.X_OK):
        exec_part = (
            f"exec {shlex.quote(str(venv_python))} "
            f"{shlex.quote(str(script))}"
        )
    else:
        exec_part = f"exec python3 {shlex.quote(str(script))}"

    return (
        f"sleep {max(0.1, float(delay_seconds)):.2f}; "
        f"cd {shlex.quote(str(repo_dir))}; "
        f"{exec_part} >> {shlex.quote(str(log_path))} 2>&1"
    )


def _choose_input_device_index(devices, default_device=None, preferred_name=None):
    """Choose a CoreAudio input device in the same order macOS users expect.

    Prefer an explicit env setting, then the system default input from Sound
    Settings, then a built-in MacBook mic, then the first available input.
    This avoids pinning the app to a stale numeric CoreAudio index.
    """
    devices = list(devices or [])

    def valid_index(idx):
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(devices) and _is_input_device(devices[idx]):
            return idx
        return None

    explicit = os.getenv("VOICE_TRANSCRIBE_AUDIO_DEVICE")
    if explicit:
        explicit_idx = valid_index(explicit)
        if explicit_idx is not None:
            return explicit_idx
        for i, dev in enumerate(devices):
            if _is_input_device(dev) and _device_name_matches(dev, explicit):
                return i

    if preferred_name:
        for i, dev in enumerate(devices):
            if _is_input_device(dev) and _device_name_matches(dev, preferred_name):
                return i

    default_input = None
    if isinstance(default_device, (list, tuple)) and default_device:
        default_input = default_device[0]
    else:
        default_input = default_device
    default_idx = valid_index(default_input)
    if default_idx is not None:
        return default_idx

    for i, dev in enumerate(devices):
        if _is_input_device(dev) and _device_name_matches(dev, "MacBook"):
            return i

    for i, dev in enumerate(devices):
        if _is_input_device(dev):
            return i

    return None


class VoiceTranscribeApp(rumps.App):
    def __init__(self, key_pipe):
        super().__init__(ICON_IDLE, quit_button=None)

        self.key_pipe = key_pipe
        self._key_monitor = None  # set by caller after construction
        self.is_recording = False
        self.is_processing = False
        self.audio_buffer = []
        self.history = self._load_history()
        self.settings = self._load_settings()
        self._pending_title = None
        self._recording_start_time = None
        self._last_heartbeat = time.time()
        self._main_thread_actions = queue.SimpleQueue()
        self._last_release_handled_at = 0.0
        self.screen_context_enabled = bool(self.settings.get("screen_context_enabled", False))
        self.sound_effects_enabled = bool(self.settings.get("sound_effects_enabled", True))
        self.vocabulary = str(self.settings.get("vocabulary", STATIC_VOCABULARY_DEFAULT))
        self.default_model_mode = self._normalize_model_mode(
            self.settings.get("default_model_mode", DEFAULT_MODEL_MODE)
        )
        if self.settings.get("default_model_mode") != self.default_model_mode:
            self.settings["default_model_mode"] = self.default_model_mode
            self._save_settings()
        self._screen_assist_selftest_enabled = False
        self._screen_context_cache_lock = threading.Lock()
        self._screen_context_cached_path = None
        self._screen_context_cached_at = 0.0
        self._screen_context_cached_app_name = ""
        self._screen_context_cached_window_title = ""
        self._screen_context_cache_ready = threading.Event()
        self._screen_text_cache_lock = threading.Lock()
        self._screen_text_cached_for_path = None
        self._screen_text_cached_at = 0.0
        self._screen_text_cached_glossary = ""
        self._screen_text_cached_terms = []
        self._screen_text_cached_lines = []
        self._screen_text_cache_ready = threading.Event()
        self._screen_ocr_requests = queue.Queue(maxsize=1)
        self._last_recording_refresh = 0.0
        self._glossary_memory = self._load_glossary_memory()

        # Visual feedback — floating HUD near cursor and optional main window
        self._hud = get_hud_controller()
        self._main_window = get_main_window_controller()
        self._main_window.attachApp_(self)
        self._hud_level_peak = 0.0
        self._sound_cache = {}
        self._main_window_shown_once = False
        # Per-model "has this ever completed a transcription this session" flag.
        # Used to label the HUD "Loading model…" on first use (cold MLX load ≈ 7s).
        self._model_warmed = self._new_model_warm_state()

        # Audio stream health. We keep the capture stream always-on, but track
        # callback freshness and can soft-refresh by opening a replacement stream
        # if CoreAudio/device state goes stale.
        self._audio_stream = None
        self._retired_audio_streams = []
        self._audio_generation = 0
        self._audio_refresh_count = 0
        self._audio_stream_lock = threading.Lock()
        self._audio_refresh_lock = threading.Lock()
        self._last_audio_callback_at = 0.0
        self._last_audio_nonzero_at = 0.0
        self._last_audio_status = ""
        self._last_audio_status_at = 0.0
        self._last_audio_rms = 0.0
        self._last_audio_refresh_at = 0.0
        self._audio_stream_device_idx = None
        self._audio_stream_device_name = "unknown"
        self._recording_started_monotonic = None
        self._recording_frames_seen = 0
        self._recording_active_frames = 0
        self._recording_max_callback_rms = 0.0
        self._recording_last_active_at = 0.0
        self._recording_auto_refresh_requested = False
        self._relaunch_requested = False

        # Make the system input gain safe before the worker starts its heavy
        # model preload. We still repeat this on each Fn recording, but doing it
        # here avoids the first dictation racing osascript against model load.
        self._force_input_volume_before_recording(context="startup")

        # Transcription worker subprocess (holds the local ASR model(s) in memory)
        self._tx_req_parent, self._tx_req_child = multiprocessing.Pipe()
        self._tx_res_parent, self._tx_res_child = multiprocessing.Pipe()
        self._tx_worker = None
        self._spawn_transcribe_worker()

        self._rebuild_menu()

        # Open always-on audio stream. See module docstring for why we never close it.
        self._open_audio_stream()

        # Poll pipe for key events in background thread
        threading.Thread(target=self._poll_pipe, daemon=True).start()
        threading.Thread(target=self._audio_health_watchdog_loop, daemon=True).start()
        threading.Thread(target=self._recording_audio_watchdog_loop, daemon=True).start()
        threading.Thread(target=self._screen_context_prefetch_loop, daemon=True).start()
        threading.Thread(target=self._screen_context_ocr_loop, daemon=True).start()

        if self._screen_assist_selftest_enabled:
            threading.Thread(target=self._run_screen_assist_self_test, daemon=True).start()

        # Wake-from-sleep hook + periodic warm ping. Both call _send_warm_signal,
        # which the worker treats as a no-op if the model is already hot and as
        # a full re-warm if GPU state is cold. See _send_warm_signal docstring.
        self._install_wake_observer()
        threading.Thread(target=self._warm_ping_loop, daemon=True).start()
        self._disable_app_nap()

    # ── Thermal Monitoring ──

    def _get_thermal_state(self):
        """Return macOS thermal state as (level_int, label, icon)."""
        try:
            from Foundation import NSProcessInfo
            level = NSProcessInfo.processInfo().thermalState()
        except Exception:
            level = 0
        label, icon = THERMAL_STATES.get(level, ("Unknown", "❓"))
        return level, label, icon

    def _get_power_state(self):
        """Return battery / low-power state for performance diagnostics."""
        state = {"source": "unknown", "percent": None, "low_power": None}

        try:
            from Foundation import NSProcessInfo

            pi = NSProcessInfo.processInfo()
            if hasattr(pi, "isLowPowerModeEnabled"):
                state["low_power"] = bool(pi.isLowPowerModeEnabled())
        except Exception:
            pass

        try:
            proc = subprocess.run(
                ["pmset", "-g", "batt"],
                text=True,
                capture_output=True,
                timeout=1.0,
            )
            if proc.stdout:
                state.update(_parse_pmset_battery_output(proc.stdout))
        except Exception:
            pass

        return state

    def _power_log_summary(self, power_state=None):
        power_state = power_state or self._get_power_state()
        source = power_state.get("source") or "unknown"
        percent = power_state.get("percent")
        low_power = power_state.get("low_power")

        parts = []
        if source != "unknown" or percent is not None:
            suffix = f" {percent}%" if percent is not None else ""
            parts.append(f"power={source}{suffix}")
        if low_power is not None:
            parts.append(f"low_power={'on' if low_power else 'off'}")
        return ", ".join(parts) if parts else "power=unknown"

    def _performance_log_summary(self, thermal_level=None, thermal_label=None, power_state=None):
        if thermal_level is None or thermal_label is None:
            thermal_level, thermal_label, _ = self._get_thermal_state()
        power_state = power_state or self._get_power_state()
        parts = [f"thermal={thermal_label}", self._power_log_summary(power_state)]

        source = str(power_state.get("source") or "").lower()
        percent = power_state.get("percent")
        low_power = bool(power_state.get("low_power"))
        battery_low = "battery" in source and percent is not None and percent <= BACKGROUND_WARM_LOW_BATTERY_PERCENT
        if thermal_level >= 2 or low_power or battery_low:
            parts.append("energy-throttle-likely")
        else:
            parts.append("if slow, likely cold GPU/model state")
        return ", ".join(part for part in parts if part)

    def _should_skip_background_warm(self, power_state=None, thermal_level=None):
        power_state = power_state or self._get_power_state()
        if thermal_level is None:
            thermal_level, _, _ = self._get_thermal_state()

        source = str(power_state.get("source") or "").lower()
        percent = power_state.get("percent")
        low_power = bool(power_state.get("low_power"))
        battery_low = "battery" in source and percent is not None and percent <= BACKGROUND_WARM_LOW_BATTERY_PERCENT
        return thermal_level >= 2 or low_power or battery_low

    def _thermal_menu_title(self):
        level, label, icon = self._get_thermal_state()
        suffix = ""
        if level == 1:
            suffix = " (minor slowdowns possible)"
        elif level == 2:
            suffix = " (GPU throttling — expect slow transcriptions)"
        elif level >= 3:
            suffix = " (heavily throttled — transcriptions will be very slow)"
        return f"{icon} Thermal: {label}{suffix}"

    def _idle_icon_with_thermal(self):
        """Return the idle icon, appending a thermal warning if needed."""
        level, _, _ = self._get_thermal_state()
        suffix = THERMAL_ICON_SUFFIX.get(level, "")
        return f"{ICON_IDLE}{suffix}" if suffix else ICON_IDLE

    # ── Key Monitor Management ──
    # The key monitor runs as a subprocess with a Quartz CGEvent tap.
    # macOS can disable event taps after sleep/wake — the monitor auto-recovers,
    # but if it dies completely, the heartbeat watchdog here restarts it.

    def _restart_key_monitor(self):
        """Kill and restart the key monitor subprocess."""
        if self._key_monitor and self._key_monitor.is_alive():
            self._key_monitor.kill()
            self._key_monitor.join(timeout=2)

        # If we were recording when monitor died, stop recording
        if self.is_recording:
            self._stop_recording()

        parent_conn, child_conn = multiprocessing.Pipe()
        self.key_pipe = parent_conn
        self._key_monitor = multiprocessing.Process(
            target=key_monitor.run, args=(child_conn,), daemon=True
        )
        self._key_monitor.start()
        self._last_heartbeat = time.time()
        print(f"Key monitor restarted (PID {self._key_monitor.pid})", flush=True)

    # ── Transcription Worker Management ──
    # The worker holds the local ASR model(s) in GPU memory.
    # It auto-restarts after 50 transcriptions or 4GB active memory to prevent
    # MLX Metal buffer leaks from accumulating (see transcribe_worker.py).

    def _spawn_transcribe_worker(self):
        """Start (or restart) the transcription worker process."""
        if self._tx_worker and self._tx_worker.is_alive():
            self._tx_worker.kill()
            self._tx_worker.join(timeout=2)

        # Fresh pipes for clean state
        self._tx_req_parent, self._tx_req_child = multiprocessing.Pipe()
        self._tx_res_parent, self._tx_res_child = multiprocessing.Pipe()

        self._tx_worker = multiprocessing.Process(
            target=transcribe_worker.run,
            args=(self._tx_req_child, self._tx_res_child),
            daemon=True,
        )
        self._tx_worker.start()
        # Fresh worker → models are cold again. Reset warm state so the HUD
        # shows "Loading model…" on the next use.
        if hasattr(self, "_model_warmed"):
            self._model_warmed = self._new_model_warm_state()

    # ── Wake / keep-warm ──
    # macOS tears down Metal GPU state during sleep. The weights stay in the
    # worker's Python heap, but the first inference after wake has to recompile
    # kernels (~5-7s). A single dummy inference fixes it — so we fire one on
    # wake events and on a slow periodic tick.

    def _install_wake_observer(self):
        try:
            from AppKit import NSWorkspace
            from Foundation import NSObject

            app_self = self

            class _WakeObserver(NSObject):
                def wokeUp_(self, _notification):  # noqa: N802
                    print("System woke from sleep → warming ASR model", flush=True)
                    app_self._send_warm_signal()

            self._wake_observer = _WakeObserver.alloc().init()
            NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
                self._wake_observer,
                b"wokeUp:",
                "NSWorkspaceDidWakeNotification",
                None,
            )
        except Exception as exc:
            print(f"Wake observer install failed: {exc}", flush=True)

    def _warm_ping_loop(self):
        """Periodically ping the worker to warm when power/thermal state allows."""
        while True:
            thermal_level, _, _ = self._get_thermal_state()
            power_state = self._get_power_state()
            low_power_mode = self._should_skip_background_warm(
                power_state=power_state,
                thermal_level=thermal_level,
            )
            interval = (
                BACKGROUND_WARM_LOW_POWER_SECONDS
                if low_power_mode
                else BACKGROUND_WARM_INTERVAL_SECONDS
            )
            time.sleep(interval)
            try:
                thermal_level, thermal_label, _ = self._get_thermal_state()
                power_state = self._get_power_state()
                if self._should_skip_background_warm(
                    power_state=power_state,
                    thermal_level=thermal_level,
                ):
                    print(
                        "Background ASR warm skipped "
                        f"({self._performance_log_summary(thermal_level, thermal_label, power_state)})",
                        flush=True,
                    )
                    continue
                self._send_warm_signal()
            except Exception as exc:
                print(f"Warm ping failed: {exc}", flush=True)

    def _disable_app_nap(self):
        """Keep the main process active — macOS won't page it out or throttle it.

        Stored as self._app_nap_activity so the activity doesn't get GC'd.
        """
        try:
            from Foundation import NSProcessInfo

            # Flags: background-ok + latency-critical + disable idle system sleep
            # Combined bitmask = 0x00FFFFFF | (1 << 20) | (1 << 19) etc. Easier to
            # use the documented constants.
            pi = NSProcessInfo.processInfo()
            # NSActivityUserInitiated (0x00FFFFFF) | NSActivityLatencyCritical (0xFF00000000)
            # See NSProcessInfo.h. Encoded as one large long in Objective-C.
            NSActivityUserInitiated = 0x00FFFFFF
            NSActivityLatencyCritical = 0xFF00000000
            options = NSActivityUserInitiated | NSActivityLatencyCritical
            self._app_nap_activity = pi.beginActivityWithOptions_reason_(
                options, "Keep ASR worker responsive across sleep/wake"
            )
        except Exception as exc:
            print(f"App Nap disable failed: {exc}", flush=True)

    def _send_warm_signal(self, model_mode=None):
        """Tell the worker to pre-warm the selected model while recording.

        Fired on Fn key down. Cohere warms MPS kernels; Granite starts or reuses
        the persistent CrispASR server so the 3GB GGUF is not reloaded per run.
        """
        try:
            self._tx_req_parent.send({
                "__warm__": True,
                "model_mode": self._normalize_model_mode(
                    model_mode or self.default_model_mode
                ),
            })
        except (BrokenPipeError, OSError):
            # Worker is dead — the next real request will respawn it.
            pass

    def _transcribe_via_worker(self, wav_path, model_mode="fast", screen_context=""):
        """Send wav to worker and wait for result with timeout. Returns dict or None."""
        request = {
            "wav_path": wav_path,
            "model_mode": model_mode,
            "screen_context": screen_context,
        }
        try:
            self._tx_req_parent.send(request)
        except (BrokenPipeError, OSError):
            print("Worker pipe broken, respawning...", flush=True)
            self._spawn_transcribe_worker()
            self._tx_req_parent.send(request)

        # Wait for result with timeout
        if self._tx_res_parent.poll(timeout=TRANSCRIBE_TIMEOUT):
            result = self._tx_res_parent.recv()

            # Worker may send a restart signal after the result
            if self._tx_res_parent.poll(timeout=0.1):
                extra = self._tx_res_parent.recv()
                if isinstance(extra, dict) and extra.get("__restart__"):
                    print("Worker requested restart (memory pressure), respawning...", flush=True)
                    self._spawn_transcribe_worker()

            # Handle case where the restart signal came instead of a result
            if isinstance(result, dict) and result.get("__restart__"):
                print("Worker requested restart (memory pressure), respawning...", flush=True)
                self._spawn_transcribe_worker()
                return None

            return result
        else:
            print(f"Transcription timed out after {TRANSCRIBE_TIMEOUT}s, killing worker...", flush=True)
            self._spawn_transcribe_worker()
            return None

    # ── UI Helpers ──

    def _set_title(self, icon):
        """Queue title change for main thread."""
        self._pending_title = icon

    @rumps.timer(0.2)
    def _tick(self, _):
        """Main-thread timer: apply pending title + update recording duration."""
        while True:
            try:
                action = self._main_thread_actions.get_nowait()
            except queue.Empty:
                break
            try:
                action()
            except Exception as exc:
                print(f"Main-thread action failed: {exc}", flush=True)

        if self._pending_title is not None:
            self.title = self._pending_title
            self._pending_title = None
        elif self.is_recording and self._recording_start_time:
            elapsed = time.time() - self._recording_start_time
            self.title = f"🔴 {elapsed:.0f}s"

        # Open main window once the runloop is live (can't call before .run()).
        if not self._main_window_shown_once:
            self._main_window_shown_once = True
            try:
                self._main_window.showWindow()
            except Exception as exc:
                print(f"Main window open failed: {exc}", flush=True)

    def _run_on_main_thread(self, action):
        if threading.current_thread() is threading.main_thread():
            action()
        else:
            self._main_thread_actions.put(action)

    # Per-sound dedupe — guards against any duplicate trigger paths (key event
    # echoes, race windows, synthetic Cmd+V observed by tap, etc.). 250ms is well
    # under intentional repeat cadence but covers OS-level event bounces.
    _SOUND_DEDUPE_WINDOW = 0.25

    def _play_sound(self, name):
        """Play a system sound (non-blocking, deduped within 250ms per name)."""
        if not self.sound_effects_enabled:
            return
        now = time.monotonic()
        last = getattr(self, "_last_sound_at", {})
        if now - last.get(name, 0.0) < self._SOUND_DEDUPE_WINDOW:
            return
        last[name] = now
        self._last_sound_at = last
        try:
            from AppKit import NSSound

            sound = self._sound_cache.get(name)
            if sound is None:
                sound = NSSound.soundNamed_(name)
                if sound is not None:
                    self._sound_cache[name] = sound
            if sound is not None:
                # Rewind if the user taps quickly; NSSound.play is async.
                try:
                    sound.stop()
                except Exception:
                    pass
                sound.play()
                return
        except Exception:
            pass

        # Fallback for environments where NSSound is unavailable.
        subprocess.Popen(
            ["afplay", f"/System/Library/Sounds/{name}.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _screen_context_prefetch_loop(self):
        """Keep a fresh screenshot ready while Screen Assist is enabled."""
        while True:
            try:
                if not self.screen_context_enabled:
                    self._clear_screen_context_cache()
                    time.sleep(1)
                    continue

                if self.is_recording or self.is_processing:
                    time.sleep(0.5)
                    continue

                needs_refresh = False
                with self._screen_context_cache_lock:
                    if not self._screen_context_cached_path:
                        needs_refresh = True
                    else:
                        age = time.time() - self._screen_context_cached_at
                        needs_refresh = age >= SCREEN_CONTEXT_PREFETCH_INTERVAL

                if needs_refresh:
                    self._refresh_screen_context_cache(reason="prefetch")

                time.sleep(1)
            except Exception as exc:
                print(f"Screen assist prefetch skipped: {exc}", flush=True)
                time.sleep(2)

    def _refresh_screen_context_cache(self, reason="prefetch"):
        """Capture and store a fresh screenshot for later transcription use."""
        snapshot = capture_frontmost_window_snapshot()
        screenshot_path = snapshot.path
        captured_at = time.time()
        previous_path = None
        had_previous = False

        with self._screen_context_cache_lock:
            previous_path = self._screen_context_cached_path
            had_previous = previous_path is not None
            self._screen_context_cached_path = screenshot_path
            self._screen_context_cached_at = captured_at
            self._screen_context_cached_app_name = snapshot.app_name
            self._screen_context_cached_window_title = snapshot.window_title
            self._screen_context_cache_ready.set()

        self._submit_screen_extract_request(
            screenshot_path,
            captured_at,
            reason,
            snapshot.app_name,
            snapshot.window_title,
            snapshot.accessibility_lines or [],
        )

        if previous_path and previous_path != screenshot_path and os.path.exists(previous_path):
            os.unlink(previous_path)

        if reason != "prefetch" or not had_previous:
            print(
                "Screen assist cached screenshot "
                f"({reason}, {os.path.getsize(screenshot_path)} bytes)",
                flush=True,
            )

    def _submit_screen_extract_request(self, screenshot_path, captured_at, reason, app_name, window_title, accessibility_lines):
        request = (screenshot_path, captured_at, reason, app_name, window_title, accessibility_lines)
        try:
            self._screen_ocr_requests.put_nowait(request)
        except queue.Full:
            try:
                self._screen_ocr_requests.get_nowait()
            except queue.Empty:
                pass
            self._screen_ocr_requests.put_nowait(request)

    def _screen_context_ocr_loop(self):
        """Extract fast screen glossary text from prefetched screenshots in the background."""
        while True:
            screenshot_path, captured_at, reason, app_name, window_title, accessibility_lines = self._screen_ocr_requests.get()
            try:
                result = extract_screen_text_context(
                    screenshot_path,
                    app_name=app_name,
                    window_title=window_title,
                    accessibility_lines=accessibility_lines,
                )
                if result.error:
                    print(f"Screen assist extraction skipped ({reason}): {result.error}", flush=True)
                    continue

                glossary = result.glossary or ""
                terms = result.terms or []
                if not glossary:
                    continue

                with self._screen_text_cache_lock:
                    had_previous = bool(self._screen_text_cached_glossary)
                    if captured_at >= self._screen_text_cached_at:
                        self._screen_text_cached_for_path = screenshot_path
                        self._screen_text_cached_at = captured_at
                        self._screen_text_cached_glossary = glossary
                        self._screen_text_cached_terms = terms
                        self._screen_text_cached_lines = []
                        self._screen_text_cache_ready.set()

                self._remember_glossary_terms(terms)

                if reason != "prefetch" or not had_previous:
                    print(
                        "Screen assist extracted screen context "
                        f"({reason}, source={result.source or 'unknown'}, {result.recognition_time_ms}ms, {len(terms)} terms): "
                        f"{glossary[:180]}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"Screen assist extraction crashed ({reason}): {exc}", flush=True)

    def _clear_screen_context_cache(self):
        """Delete any cached screenshot when Screen Assist is disabled."""
        cached_path = None
        with self._screen_context_cache_lock:
            cached_path = self._screen_context_cached_path
            self._screen_context_cached_path = None
            self._screen_context_cached_at = 0.0
            self._screen_context_cached_app_name = ""
            self._screen_context_cached_window_title = ""
            self._screen_context_cache_ready.clear()

        if cached_path and os.path.exists(cached_path):
            os.unlink(cached_path)

        with self._screen_text_cache_lock:
            self._screen_text_cached_for_path = None
            self._screen_text_cached_at = 0.0
            self._screen_text_cached_glossary = ""
            self._screen_text_cached_terms = []
            self._screen_text_cached_lines = []
            self._screen_text_cache_ready.clear()

    def _take_screen_context_snapshot(self):
        """Return a ready screenshot path, preferring a prefetched screenshot."""
        stale_path = None

        with self._screen_context_cache_lock:
            if self._screen_context_cached_path:
                age = time.time() - self._screen_context_cached_at
                path = self._screen_context_cached_path
                self._screen_context_cached_path = None
                self._screen_context_cached_at = 0.0
                self._screen_context_cache_ready.clear()

                if age <= SCREEN_CONTEXT_MAX_AGE:
                    return path, "prefetched", age

                stale_path = path

        if stale_path and os.path.exists(stale_path):
            os.unlink(stale_path)

        snapshot = capture_frontmost_window_snapshot()
        self._submit_screen_extract_request(
            snapshot.path,
            time.time(),
            "on-demand",
            snapshot.app_name,
            snapshot.window_title,
            snapshot.accessibility_lines or [],
        )
        return snapshot.path, "on-demand", None

    def _get_screen_context_glossary(self, screenshot_path):
        """Return merged fresh + retained glossary context without blocking transcription."""
        with self._screen_text_cache_lock:
            glossary = self._screen_text_cached_glossary
            cached_for_path = self._screen_text_cached_for_path
            cached_at = self._screen_text_cached_at
            fresh_terms = list(self._screen_text_cached_terms or [])

        retained_terms = self._get_retained_glossary_terms()
        if not glossary and not retained_terms:
            return "", "none", None

        age = time.time() - cached_at if cached_at else None
        if glossary and age is not None and age > SCREEN_CONTEXT_MAX_AGE:
            glossary = ""
            fresh_terms = []

        merged_terms = []
        seen = set()
        for term in fresh_terms + retained_terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_terms.append(term)

        if not merged_terms:
            return "", "stale", age

        merged_context = self._build_screen_context_string(glossary, fresh_terms, retained_terms)
        memory_suffix = "+memory" if retained_terms else ""

        if fresh_terms and cached_for_path == screenshot_path:
            return merged_context, f"matching-prefetch{memory_suffix}", age
        if fresh_terms:
            return merged_context, f"recent-prefetch{memory_suffix}", age
        return merged_context, "memory-only", None

    def _run_screen_assist_self_test(self):
        """Run a one-shot fast screen-context probe inside the app process for verification."""
        screenshot_path = None
        audio_path = None
        try:
            print(
                "Screen assist self-test starting "
                f"(screen_context_enabled={self.screen_context_enabled})",
                flush=True,
            )
            self._screen_context_cache_ready.wait(timeout=6)
            screenshot_path, source, age = self._take_screen_context_snapshot()
            self._screen_text_cache_ready.wait(timeout=12)
            print(
                "Screen assist self-test using "
                f"{source} screenshot "
                f"({os.path.getsize(screenshot_path)} bytes"
                f"{'' if age is None else f', age={age:.1f}s'})",
                flush=True,
            )
            glossary, glossary_source, glossary_age = self._get_screen_context_glossary(screenshot_path)
            if not glossary:
                print("Screen assist self-test failed: screen glossary unavailable", flush=True)
                return

            print(
                "Screen assist self-test screen context ready "
                f"({glossary_source}"
                f"{'' if glossary_age is None else f', age={glossary_age:.1f}s'}): "
                f"{glossary[:220]}",
                flush=True,
            )

            audio_path = tempfile.mktemp(suffix=".wav")
            subprocess.run(
                [
                    "say",
                    "-o",
                    audio_path,
                    "--file-format=WAVE",
                    "--data-format=LEI16@16000",
                    "Open Claude transcription test",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(
                f"Screen assist self-test injecting screen glossary into ASR ({len(glossary)} chars)",
                flush=True,
            )
            result = self._transcribe_via_worker(
                audio_path,
                model_mode="fast",
                screen_context=glossary,
            )
            if result is None:
                print("Screen assist self-test ASR timed out", flush=True)
            elif result.get("error"):
                print(f"Screen assist self-test ASR failed: {result['error']}", flush=True)
            else:
                print(
                    "Screen assist self-test ASR result: "
                    f"{result.get('text', '').strip()}",
                    flush=True,
                )
        except Exception as exc:
            print(f"Screen assist self-test crashed: {exc}", flush=True)
        finally:
            if screenshot_path and os.path.exists(screenshot_path):
                os.unlink(screenshot_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    # ── Pipe Polling ──
    # Runs in a background thread. Receives messages from the key monitor subprocess.
    # Messages: "heartbeat", "down:default", "up"

    def _poll_pipe(self):
        self._last_heartbeat = time.time()
        self._pending_model = None
        while True:
            try:
                # Use poll with timeout so we can detect a dead key monitor
                if self.key_pipe.poll(timeout=30):
                    msg = self.key_pipe.recv()
                else:
                    if time.time() - self._last_heartbeat > 30:
                        print("Key monitor appears dead (no heartbeat), restarting...", flush=True)
                        self._restart_key_monitor()
                    continue

                if msg == "heartbeat":
                    self._last_heartbeat = time.time()
                    continue
                elif msg.startswith("down:"):
                    if not self.is_recording and not self.is_processing:
                        model_mode = self.default_model_mode
                        self._pending_model = model_mode
                        model_label = self._model_display_name(model_mode)
                        print(f"Fn hold → recording ({model_label})", flush=True)
                        self._play_sound("Tink")
                        # Speculatively warm the selected model while user is recording.
                        # Cohere warms MPS; Granite starts/reuses the resident CrispASR
                        # server so cold-start latency is hidden inside recording time.
                        self._send_warm_signal(model_mode)
                        self._start_recording()
                elif msg == "up":
                    if self.is_recording:
                        now = time.monotonic()
                        if now - self._last_release_handled_at < RELEASE_DEBOUNCE_SECONDS:
                            print("Ignoring duplicate release event", flush=True)
                            continue
                        self._last_release_handled_at = now
                        elapsed = time.time() - self._recording_start_time if self._recording_start_time else 0
                        model_label = self._model_display_name(
                            self._pending_model or self.default_model_mode
                        )
                        print(f"Release → stop ({elapsed:.1f}s), transcribing with {model_label}...", flush=True)
                        self._play_sound("Pop")
                        self._stop_recording()
            except (EOFError, OSError):
                print("Key monitor pipe broken, restarting...", flush=True)
                self._restart_key_monitor()
                continue

    # ── Audio Stream (always-on) ──
    # DO NOT add stream.stop(), stream.abort(), or stream.close() anywhere.
    # See module docstring for the CoreAudio HALB_Mutex deadlock explanation.

    def _select_input_device(self):
        device_idx = None
        devices = []
        try:
            devices = sd.query_devices()
            device_idx = _choose_input_device_index(
                devices,
                default_device=sd.default.device,
                preferred_name=self.settings.get("audio_input_device_name"),
            )
        except Exception as exc:
            print(f"Audio device query failed; using PortAudio default input: {exc}", flush=True)

        try:
            dev_name = devices[device_idx]["name"] if device_idx is not None else "default"
        except Exception:
            dev_name = "default"
        return device_idx, dev_name

    def _open_audio_stream(self, reason="startup"):
        """Open a persistent audio input stream.

        The callback always fires (~every 10ms). When is_recording is False, it
        discards the data (negligible CPU). When True, it appends to audio_buffer.

        Refresh note: we do not stop/close old streams because that can deadlock
        CoreAudio/PortAudio. A refresh opens a replacement stream and retires the
        previous object until process exit.
        """
        device_idx, dev_name = self._select_input_device()
        generation = self._audio_generation + 1

        def audio_callback(indata, frames, time_info, status):
            if generation != self._audio_generation:
                return
            now = time.monotonic()
            self._last_audio_callback_at = now
            if status:
                self._last_audio_status = str(status)
                self._last_audio_status_at = now
                print(f"Audio status: {status}", flush=True)
            try:
                rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
                self._last_audio_rms = rms
                if rms >= 0.001:
                    self._last_audio_nonzero_at = now
            except Exception:
                rms = 0.0
            if self.is_recording:
                self.audio_buffer.append(indata.copy())
                self._recording_frames_seen += int(frames or len(indata))
                self._recording_max_callback_rms = max(self._recording_max_callback_rms, rms)
                if rms >= SILENCE_ACTIVE_RMS_THRESHOLD:
                    self._recording_active_frames += int(frames or len(indata))
                    self._recording_last_active_at = now
                # Feed HUD waveform — one level per callback is enough for 30fps draw.
                try:
                    # Noise gate below the mic noise floor (~0.004-0.006 RMS on
                    # MacBook mics) → bars flatten when silent rather than
                    # jittering on ambient noise. Real speech (~0.02+ RMS) still
                    # produces full-height bars.
                    level = max(0.0, min(1.0, (rms - 0.006) * 12.0))
                    self._hud.pushLevel_(level)
                except Exception:
                    pass

        stream = sd.InputStream(
            device=device_idx,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()

        with self._audio_stream_lock:
            previous = self._audio_stream
            self._audio_generation = generation
            self._audio_stream = stream
            self._audio_stream_device_idx = device_idx
            self._audio_stream_device_name = dev_name
            if previous is not None:
                self._retired_audio_streams.append(previous)
                self._audio_refresh_count += 1
                self._last_audio_refresh_at = time.monotonic()

        retired_note = (
            f", retired_streams={len(self._retired_audio_streams)}"
            if self._retired_audio_streams
            else ""
        )
        print(
            f"Audio stream opened on device {device_idx}: {dev_name} "
            f"(reason={reason}, generation={generation}{retired_note})",
            flush=True,
        )

    def _audio_health_snapshot(self):
        now = time.monotonic()
        callback_age = None
        if self._last_audio_callback_at:
            callback_age = now - self._last_audio_callback_at
        status_age = None
        if self._last_audio_status_at:
            status_age = now - self._last_audio_status_at
        return {
            "callback_age": callback_age,
            "status_age": status_age,
            "last_status": self._last_audio_status,
            "last_rms": self._last_audio_rms,
            "device_idx": self._audio_stream_device_idx,
            "device_name": self._audio_stream_device_name,
            "generation": self._audio_generation,
            "refresh_count": self._audio_refresh_count,
            "retired_streams": len(self._retired_audio_streams),
        }

    def _refresh_audio_stream_async(self, reason, force=False):
        threading.Thread(
            target=self._refresh_audio_stream,
            kwargs={"reason": reason, "force": force},
            daemon=True,
        ).start()

    def _schedule_self_relaunch(self, reason):
        """Relaunch the app when CoreAudio needs a full process-level reset."""
        if self._relaunch_requested:
            return
        self._relaunch_requested = True
        print(
            "Audio recovery needs a clean app relaunch "
            f"({reason}). Restarting Voice Transcribe now...",
            flush=True,
        )

        repo_dir = Path(__file__).resolve().parent
        relaunch_script = _build_self_relaunch_script(
            repo_dir,
            AUDIO_RELAUNCH_LOG_FILE,
            AUDIO_RELAUNCH_DELAY_SECONDS,
        )

        try:
            subprocess.Popen(
                ["/bin/bash", "-lc", relaunch_script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as exc:
            self._relaunch_requested = False
            print(f"Audio recovery relaunch failed to schedule: {exc}", flush=True)
            return

        for proc in multiprocessing.active_children():
            try:
                proc.kill()
                proc.join(timeout=1)
            except Exception:
                pass
        os._exit(75)

    def _refresh_audio_stream(self, reason, force=False):
        """Soft-refresh input without stopping the old CoreAudio stream."""
        if not self._audio_refresh_lock.acquire(blocking=False):
            return False
        try:
            now = time.monotonic()
            if (
                not force
                and self._last_audio_refresh_at
                and now - self._last_audio_refresh_at < AUDIO_REFRESH_COOLDOWN_SECONDS
            ):
                return False
            if len(self._retired_audio_streams) >= AUDIO_MAX_RETIRED_STREAMS:
                print(
                    "Audio stream refresh skipped: retired-stream safety cap reached; "
                    "a full app relaunch is needed to reset CoreAudio cleanly.",
                    flush=True,
                )
                if AUDIO_RELAUNCH_ON_REFRESH_CAP:
                    self._schedule_self_relaunch(
                        f"retired stream cap reached during {reason}"
                    )
                return False

            snapshot = self._audio_health_snapshot()
            age = snapshot["callback_age"]
            age_text = "never" if age is None else f"{age:.1f}s ago"
            print(
                "Refreshing audio input stream "
                f"({reason}; last_callback={age_text}, "
                f"last_rms={snapshot['last_rms']:.4f}, "
                f"device={snapshot['device_idx']}:{snapshot['device_name']})",
                flush=True,
            )
            self._open_audio_stream(reason=reason)
            return True
        except Exception as exc:
            print(f"Audio stream refresh failed ({reason}): {exc}", flush=True)
            return False
        finally:
            self._audio_refresh_lock.release()

    def _force_input_volume_before_recording(self, context="before recording"):
        if not INPUT_VOLUME_FORCE_ON_RECORDING:
            return
        result = _force_macos_input_volume_to_max()
        previous = result.get("previous")
        current = result.get("current")
        error = result.get("error")
        if not _input_volume_should_log(previous, current, error=error):
            return

        if error:
            print(
                f"Input volume max check failed ({context}) "
                f"(previous={previous}, current={current}): {error}",
                flush=True,
            )
        else:
            print(
                f"Input volume forced to 100 ({context}) "
                f"(previous={previous}, current={current})",
                flush=True,
            )

    def _ensure_audio_stream_healthy(self, reason):
        snapshot = self._audio_health_snapshot()
        callback_age = snapshot["callback_age"]
        if self._audio_stream is None:
            return self._refresh_audio_stream(reason=f"{reason}: no stream", force=True)
        if callback_age is None or callback_age > AUDIO_CALLBACK_STALE_SECONDS:
            age_text = "never" if callback_age is None else f"{callback_age:.1f}s"
            return self._refresh_audio_stream(
                reason=f"{reason}: callback stale ({age_text})",
                force=True,
            )
        return False

    def _audio_health_watchdog_loop(self):
        while True:
            time.sleep(AUDIO_HEALTH_CHECK_SECONDS)
            try:
                if self.is_recording or self.is_processing:
                    continue
                self._ensure_audio_stream_healthy("watchdog")
            except Exception as exc:
                print(f"Audio health watchdog failed: {exc}", flush=True)

    def _recording_audio_watchdog_loop(self):
        """Refresh during a hold if the user is talking but CoreAudio is flat.

        The older recovery only happened after release, which meant one bad hold
        still failed. This nudges the input stream while Fn is still down so the
        same dictation can often recover before the user lets go.
        """
        while True:
            time.sleep(AUDIO_RECORDING_WATCHDOG_SECONDS)
            try:
                if not self.is_recording or self.is_processing:
                    continue
                if self._recording_auto_refresh_requested:
                    continue
                started = self._recording_started_monotonic
                if not started:
                    continue

                held = time.monotonic() - started
                if held < AUDIO_RECORDING_FLAT_REFRESH_SECONDS:
                    continue

                active_seconds = self._recording_active_frames / float(SAMPLE_RATE)
                max_rms = self._recording_max_callback_rms
                if (
                    active_seconds < AUDIO_RECORDING_MIN_ACTIVE_SECONDS
                    and max_rms < AUDIO_DEAD_INPUT_MAX_RMS
                ):
                    self._recording_auto_refresh_requested = True
                    print(
                        "Mic input is flat while recording; refreshing input stream "
                        f"mid-hold (held={held:.1f}s, active={active_seconds:.2f}s, "
                        f"max_rms={max_rms:.4f}).",
                        flush=True,
                    )
                    self._refresh_audio_stream_async(
                        "recording-live flat input",
                        force=False,
                    )
            except Exception as exc:
                print(f"Recording audio watchdog failed: {exc}", flush=True)

    # ── Recording ──
    # Start/stop are just flag toggles — no CoreAudio calls, no mutex, no deadlock.

    def _start_recording(self):
        if self.is_recording or self.is_processing:
            return
        self._force_input_volume_before_recording()
        self._ensure_audio_stream_healthy("recording-start")
        self.audio_buffer = []
        self._recording_started_monotonic = time.monotonic()
        self._recording_frames_seen = 0
        self._recording_active_frames = 0
        self._recording_max_callback_rms = 0.0
        self._recording_last_active_at = 0.0
        self._recording_auto_refresh_requested = False
        self.is_recording = True
        self._recording_start_time = time.time()
        self._run_on_main_thread(self._hud.show)
        if self.screen_context_enabled:
            now = time.time()
            if now - self._last_recording_refresh >= SCREEN_CONTEXT_RECORDING_REFRESH_INTERVAL:
                self._last_recording_refresh = now
                threading.Thread(
                    target=self._refresh_screen_context_cache,
                    kwargs={"reason": "recording-start"},
                    daemon=True,
                ).start()
        self._set_title(ICON_RECORDING)

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._recording_started_monotonic = None
        self._recording_start_time = None
        # Set is_processing synchronously here (not inside the worker thread) so
        # there's no race window where both is_recording and is_processing are
        # False — otherwise a stray flagsChanged echo arriving in that gap could
        # trigger a duplicate Tink + recording start.
        self.is_processing = True
        self._set_title(ICON_PROCESSING)

        audio_data = self.audio_buffer
        self.audio_buffer = []

        if not audio_data:
            print(
                "No audio callbacks captured during recording; refreshing mic input.",
                flush=True,
            )
            self._refresh_audio_stream_async("empty-recording", force=True)
            self.is_processing = False
            self._set_title(self._idle_icon_with_thermal())
            self._run_on_main_thread(self._hud.hide)
            return

        model_mode = self._pending_model or self.default_model_mode
        loading = not self._model_warmed.get(model_mode, False)
        hud_label = "Loading model…" if loading else "Transcribing…"
        self._run_on_main_thread(lambda lbl=hud_label: self._hud.setProcessingWithLabel_(lbl))
        threading.Thread(
            target=self._transcribe_and_paste, args=(audio_data, model_mode), daemon=True
        ).start()

    # ── Transcription ──

    def _transcribe_and_paste(self, audio_data, model_mode="fast"):
        self.is_processing = True
        wav_path = None
        screenshot_path = None
        screen_context = ""
        history_queued = False
        latency_t0 = time.perf_counter()
        latency = {}
        try:
            prep_t0 = time.perf_counter()
            audio = np.concatenate(audio_data, axis=0).flatten()
            volume_stats = _analyze_audio_volume(audio)
            duration = volume_stats["duration"]
            peak = volume_stats["peak"]
            max_rms = volume_stats["max_rms"]
            low_confidence_no_volume = _is_low_confidence_no_volume(volume_stats)

            thermal_level, thermal_label, thermal_icon = self._get_thermal_state()
            thermal_note = f", thermal={thermal_label}" if thermal_level > 0 else ""
            print(
                f"Audio: {duration:.1f}s, peak={peak:.4f}, max_rms={max_rms:.4f} "
                f"full_rms={volume_stats['full_rms']:.4f}, "
                f"active={volume_stats['active_seconds']:.2f}s/"
                f"{volume_stats['longest_active_seconds']:.2f}s "
                f"({'SILENT' if peak < 0.001 else 'OK'}){thermal_note}",
                flush=True,
            )

            # Silence gate — skip ASR entirely on silent audio. This prevents
            # the model from hallucinating "Thank you."-style outputs, and
            # also saves a GPU roundtrip. Short-but-loud recordings (e.g. "yes",
            # "no") pass through; short-and-quiet keypress artifacts don't.
            no_volume_reason = _no_volume_skip_reason(volume_stats)
            if no_volume_reason:
                print(
                    f"No usable volume ({no_volume_reason}), ending immediately.",
                    flush=True,
                )
                refresh_reason = _dead_input_refresh_reason(volume_stats)
                if refresh_reason:
                    print(
                        f"Mic input looked wedged ({refresh_reason}); refreshing before next recording.",
                        flush=True,
                    )
                    self._refresh_audio_stream_async(
                        f"dead-input after no-volume: {refresh_reason}",
                        force=False,
                    )
                self._set_title(self._idle_icon_with_thermal())
                return

            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            self._save_last_recording(wav_path)
            latency["prep"] = time.perf_counter() - prep_t0

            if self.screen_context_enabled:
                context_t0 = time.perf_counter()
                try:
                    screenshot_path, screenshot_source, screenshot_age = self._take_screen_context_snapshot()
                    if screenshot_source == "prefetched":
                        print(
                            f"Screen assist using prefetched screenshot (age={screenshot_age:.1f}s)",
                            flush=True,
                        )
                    else:
                        print("Screen assist cache miss; captured screenshot on demand", flush=True)

                    screen_context, context_source, context_age = self._get_screen_context_glossary(
                        screenshot_path
                    )
                    if screen_context:
                        print(
                            "Screen assist injecting screen context "
                            f"({context_source}"
                            f"{'' if context_age is None else f', age={context_age:.1f}s'}): "
                            f"{screen_context[:220]}",
                            flush=True,
                        )
                    else:
                        print(
                            "Screen assist screen glossary unavailable; transcribing without screen context",
                            flush=True,
                        )
                except Exception as screen_err:
                    print(f"Screen context capture skipped: {screen_err}", flush=True)
                finally:
                    latency["context"] = time.perf_counter() - context_t0

            # Cohere-only: prepend the static vocabulary to whatever screen
            # context was assembled. Qwen3 modes get the screen context as-is.
            if model_mode == "cohere" and self.vocabulary:
                screen_context = (
                    f"{self.vocabulary} {screen_context}".strip()
                    if screen_context
                    else self.vocabulary
                )

            asr_t0 = time.perf_counter()
            result = self._transcribe_via_worker(
                wav_path,
                model_mode,
                screen_context=screen_context,
            )
            latency["asr_wall"] = time.perf_counter() - asr_t0

            if result is None:
                print("Transcription timed out, skipping.", flush=True)
                rumps.notification("Voice Transcribe", "Timeout", "Transcription took too long. Try again.")
                return

            if result.get("error"):
                if model_mode == "granite":
                    if low_confidence_no_volume:
                        print(
                            "Granite failed on low-volume audio; treating as no clip "
                            "instead of falling back to Cohere.",
                            flush=True,
                        )
                        return
                    print(
                        f"Granite transcription error, falling back to Cohere: {result['error']}",
                        flush=True,
                    )
                    fallback_t0 = time.perf_counter()
                    fallback = self._transcribe_via_worker(
                        wav_path,
                        "cohere",
                        screen_context=screen_context,
                    )
                    latency["fallback_wall"] = time.perf_counter() - fallback_t0
                    if fallback is not None and not fallback.get("error"):
                        result = fallback
                        model_mode = "cohere"
                    else:
                        self._preserve_failed_recording(wav_path, "granite_error")
                        if fallback is None:
                            raise Exception(result["error"])
                        raise Exception(fallback.get("error") or result["error"])
                else:
                    self._preserve_failed_recording(wav_path, "error")
                    raise Exception(result["error"])

            # Model produced a result → it's warm now. Subsequent HUDs show
            # "Transcribing…" instead of "Loading model…".
            self._model_warmed[model_mode] = True

            text = result.get("text", "")
            transcribe_time = result.get("time", 0)

            if not text:
                if model_mode == "granite":
                    if low_confidence_no_volume:
                        print(
                            "Granite returned no text on low-volume audio; ending immediately.",
                            flush=True,
                        )
                        return
                    print("Granite returned no text; falling back to Cohere...", flush=True)
                    fallback_t0 = time.perf_counter()
                    fallback = self._transcribe_via_worker(
                        wav_path,
                        "cohere",
                        screen_context=screen_context,
                    )
                    latency["fallback_wall"] = time.perf_counter() - fallback_t0
                    if fallback is not None and not fallback.get("error"):
                        model_mode = "cohere"
                        self._model_warmed["cohere"] = True
                        text = fallback.get("text", "")
                        transcribe_time = fallback.get("time", 0)
                    else:
                        self._preserve_failed_recording(wav_path, "empty")
                        print("No speech detected.", flush=True)
                        return
                if not text:
                    self._preserve_failed_recording(wav_path, "empty")
                    print("No speech detected.", flush=True)
                    return

            # Hallucination filter — ASR models trained on YouTube captions
            # frequently emit "Thank you.", ".", "you", etc. on near-silent
            # input. Collapse to a normalized form and drop if it matches a
            # known hallucination.
            normalized = text.strip().lower().rstrip(".!?,;: ").strip()
            if normalized in ASR_HALLUCINATIONS and model_mode == "granite":
                if low_confidence_no_volume:
                    print(
                        f"Granite returned low-content output {text!r} on low-volume audio; "
                        "ending immediately.",
                        flush=True,
                    )
                    return
                print(
                    f"Granite returned low-content output {text!r}; falling back to Cohere...",
                    flush=True,
                )
                fallback_t0 = time.perf_counter()
                fallback = self._transcribe_via_worker(
                    wav_path,
                    "cohere",
                    screen_context=screen_context,
                )
                latency["fallback_wall"] = time.perf_counter() - fallback_t0
                if fallback is not None and not fallback.get("error"):
                    model_mode = "cohere"
                    self._model_warmed["cohere"] = True
                    text = fallback.get("text", "")
                    transcribe_time = fallback.get("time", 0)
                    normalized = text.strip().lower().rstrip(".!?,;: ").strip()
                else:
                    self._preserve_failed_recording(wav_path, "granite_hallucination")
                    return

            if not text:
                self._preserve_failed_recording(wav_path, "empty")
                print("No speech detected.", flush=True)
                return

            if normalized in ASR_HALLUCINATIONS:
                print(
                    f"Dropping ASR hallucination on low-content audio: {text!r}",
                    flush=True,
                )
                self._preserve_failed_recording(wav_path, "hallucination")
                return

            format_t0 = time.perf_counter()
            text = format_transcription(text)
            latency["format"] = time.perf_counter() - format_t0
            post_thermal_level, post_thermal_label, _ = self._get_thermal_state()
            slow_note = ""
            if transcribe_time > 3.0:
                slow_note = (
                    " ⚠️ SLOW "
                    f"({self._performance_log_summary(post_thermal_level, post_thermal_label)})"
                )
            print(f"Transcribed ({duration:.1f}s audio → {transcribe_time:.1f}s{slow_note}): {text}", flush=True)
            self._add_to_history(text)
            history_queued = True
            paste_t0 = time.perf_counter()
            self._paste_text(text)
            latency["paste"] = time.perf_counter() - paste_t0
            latency["total"] = time.perf_counter() - latency_t0
            latency_parts = [
                f"total={latency.get('total', 0):.2f}s",
                f"prep={latency.get('prep', 0):.2f}s",
                f"asr_wall={latency.get('asr_wall', 0):.2f}s",
                f"asr_worker={transcribe_time:.2f}s",
                f"format={latency.get('format', 0):.2f}s",
                f"paste={latency.get('paste', 0):.2f}s",
            ]
            if latency.get("context"):
                latency_parts.insert(2, f"context={latency['context']:.2f}s")
            if latency.get("fallback_wall"):
                latency_parts.insert(3, f"fallback_wall={latency['fallback_wall']:.2f}s")
            print(f"Latency: model={model_mode} " + " ".join(latency_parts), flush=True)

        except Exception as e:
            print(f"Transcription error: {e}", flush=True)
            rumps.notification("Voice Transcribe", "Error", str(e))
        finally:
            self.is_processing = False
            self._set_title(self._idle_icon_with_thermal())
            if not history_queued:
                self._run_on_main_thread(self._rebuild_menu)
            self._run_on_main_thread(self._hud.hide)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            if screenshot_path and os.path.exists(screenshot_path):
                os.unlink(screenshot_path)

    # ── Paste ──

    def _paste_text(self, text):
        """Paste text at cursor without clobbering the user's clipboard.

        Saves the current clipboard, pastes the transcription via Cmd+V,
        then restores the original clipboard contents. The user's previous
        copy is preserved — transcription just gets inserted inline.
        """
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags,
            CGEventSourceCreate, kCGEventFlagMaskCommand,
        )
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()

        # Save current clipboard contents
        old_contents = pb.stringForType_(NSPasteboardTypeString)
        old_change_count = pb.changeCount()

        # Set clipboard to transcription and paste
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)

        time.sleep(PASTEBOARD_PRE_PASTE_DELAY)

        source = CGEventSourceCreate(0)
        key_down = CGEventCreateKeyboardEvent(source, 9, True)
        key_up = CGEventCreateKeyboardEvent(source, 9, False)
        CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
        CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
        CGEventPost(0, key_down)
        CGEventPost(0, key_up)

        # Wait for paste to complete, then restore original clipboard
        time.sleep(PASTEBOARD_RESTORE_DELAY)
        if old_contents is not None:
            pb.clearContents()
            pb.setString_forType_(old_contents, NSPasteboardTypeString)

    def _save_last_recording(self, wav_path):
        """Keep a copy of the latest recording for manual recovery/debugging."""
        try:
            shutil.copy2(wav_path, LAST_RECORDING_FILE)
        except Exception as exc:
            print(f"Failed to save last recording: {exc}", flush=True)

    def _preserve_failed_recording(self, wav_path, reason):
        """Archive a failed recording before temp cleanup so audio is recoverable."""
        if not wav_path or not os.path.exists(wav_path):
            return None
        try:
            FAILED_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = FAILED_RECORDINGS_DIR / f"{stamp}-{reason}.wav"
            shutil.copy2(wav_path, dest)
            print(f"Preserved failed recording: {dest}", flush=True)
            return dest
        except Exception as exc:
            print(f"Failed to preserve failed recording: {exc}", flush=True)
            return None

    # ── History ──

    def _load_history(self):
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_history(self):
        try:
            HISTORY_FILE.write_text(json.dumps(self.history, indent=2))
        except IOError as e:
            print(f"Failed to save history: {e}", flush=True)

    def _load_settings(self):
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "screen_context_enabled": False,
            "sound_effects_enabled": True,
            "default_model_mode": DEFAULT_MODEL_MODE,
        }

    def _save_settings(self):
        try:
            SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2))
        except IOError as e:
            print(f"Failed to save settings: {e}", flush=True)

    def _load_glossary_memory(self):
        if GLOSSARY_MEMORY_FILE.exists():
            try:
                data = json.loads(GLOSSARY_MEMORY_FILE.read_text())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return {"terms": {}}

    def _save_glossary_memory(self):
        try:
            GLOSSARY_MEMORY_FILE.write_text(json.dumps(self._glossary_memory, indent=2))
        except IOError as e:
            print(f"Failed to save glossary memory: {e}", flush=True)

    def _should_persist_term(self, term):
        cleaned = term.strip()
        if len(cleaned) < 3:
            return False
        if not any(ch.isalpha() for ch in cleaned):
            return False
        if cleaned.islower() and " " not in cleaned and len(cleaned) < 5:
            return False
        return True

    def _remember_glossary_terms(self, terms):
        if not terms:
            return

        now = time.time()
        bucket = self._glossary_memory.setdefault("terms", {})
        changed = False
        for term in terms:
            if not self._should_persist_term(term):
                continue
            record = bucket.setdefault(term, {"count": 0, "last_seen": 0})
            record["count"] += 1
            record["last_seen"] = now
            changed = True

        if not changed:
            return

        sorted_items = sorted(
            bucket.items(),
            key=lambda item: (item[1].get("count", 0), item[1].get("last_seen", 0)),
            reverse=True,
        )[:GLOSSARY_MEMORY_MAX_TERMS]
        self._glossary_memory["terms"] = {key: value for key, value in sorted_items}
        self._save_glossary_memory()

    def _get_retained_glossary_terms(self):
        bucket = self._glossary_memory.get("terms", {})
        if not isinstance(bucket, dict):
            return []

        retained = [
            (term, meta.get("count", 0), meta.get("last_seen", 0))
            for term, meta in bucket.items()
            if meta.get("count", 0) >= GLOSSARY_MEMORY_MIN_COUNT
        ]
        retained.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [term for term, _, _ in retained[:GLOSSARY_MEMORY_TOP_TERMS]]

    def _build_screen_context_string(self, fresh_glossary, fresh_terms, retained_terms):
        parts = []
        if fresh_glossary:
            parts.append(fresh_glossary)

        retained_unique = []
        fresh_keys = {term.lower() for term in fresh_terms}
        for term in retained_terms:
            if term.lower() not in fresh_keys:
                retained_unique.append(term)

        if retained_unique:
            parts.append("Frequent terms: " + ", ".join(retained_unique))

        context = " | ".join(parts).strip()
        if len(context) > SCREEN_CONTEXT_MAX_CHARS:
            context = context[: SCREEN_CONTEXT_MAX_CHARS - 1].rstrip() + "…"
        return context

    def _add_to_history(self, text):
        if threading.current_thread() is not threading.main_thread():
            self._run_on_main_thread(lambda: self._add_to_history(text))
            return

        entry = {"text": text, "timestamp": datetime.now().isoformat()}
        self.history.insert(0, entry)
        self.history = self.history[:MAX_HISTORY]
        self._save_history()
        self._rebuild_menu()

    def _rebuild_menu(self):
        if threading.current_thread() is not threading.main_thread():
            self._run_on_main_thread(self._rebuild_menu)
            return

        self.menu.clear()

        self.menu.add(rumps.MenuItem("Open Window", callback=self._open_main_window))
        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem(
            f"Fn = {self._model_display_name(self.default_model_mode)}",
            callback=None,
        ))
        self.menu.add(rumps.MenuItem("Default Model", callback=None))
        for mode in MENU_MODEL_MODES:
            item = rumps.MenuItem(
                self._default_model_menu_title(mode),
                callback=self._set_default_model,
            )
            item.representedObject = mode
            try:
                item.state = 1 if mode == self.default_model_mode else 0
            except Exception:
                pass
            self.menu.add(item)
        self.menu.add(rumps.MenuItem(self._screen_context_menu_title(), callback=self._toggle_screen_context))
        self.menu.add(rumps.MenuItem(self._sound_effects_menu_title(), callback=self._toggle_sound_effects))
        self.menu.add(rumps.MenuItem(self._thermal_menu_title(), callback=None))
        self.menu.add(rumps.separator)

        if self.history:
            self.menu.add(rumps.MenuItem("Recent Transcriptions", callback=None))
            for entry in self.history[:20]:
                text = entry["text"]
                display = text[:80] + "…" if len(text) > 80 else text
                ts = entry.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        display = f"[{dt.strftime('%H:%M')}] {display}"
                    except ValueError:
                        pass
                item = rumps.MenuItem(display, callback=self._copy_history_item)
                item.representedObject = text
                self.menu.add(item)
            self.menu.add(rumps.separator)
            self.menu.add(rumps.MenuItem("Clear History", callback=self._clear_history))
        else:
            self.menu.add(rumps.MenuItem("No transcriptions yet", callback=None))

        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Quit", callback=rumps.quit_application))

    def _open_main_window(self, _sender=None):
        self._run_on_main_thread(self._main_window.showWindow)

    def _copy_history_item(self, sender):
        from AppKit import NSPasteboard, NSPasteboardTypeString
        text = getattr(sender, "representedObject", sender.title)
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)
        rumps.notification("Voice Transcribe", "Copied", text[:100])

    def _normalize_model_mode(self, mode):
        mode = str(mode or "").strip().lower()
        return mode if mode in MODEL_LABELS else DEFAULT_MODEL_MODE

    def _new_model_warm_state(self):
        return {mode: False for mode in MODEL_LABELS}

    def _model_display_name(self, mode):
        return MODEL_LABELS.get(self._normalize_model_mode(mode), MODEL_LABELS[DEFAULT_MODEL_MODE])

    def _default_model_menu_title(self, mode):
        prefix = "✓ " if mode == self.default_model_mode else "   "
        return f"{prefix}{self._model_display_name(mode)}"

    def _set_default_model(self, sender):
        mode = self._normalize_model_mode(getattr(sender, "representedObject", None))
        if mode == self.default_model_mode:
            return
        self.default_model_mode = mode
        self.settings["default_model_mode"] = mode
        self._save_settings()
        self._rebuild_menu()
        rumps.notification(
            "Voice Transcribe",
            "Default Model",
            f"Fn now uses {self._model_display_name(mode)}.",
        )

    def _screen_context_menu_title(self):
        status = "On" if self.screen_context_enabled else "Off"
        return f"Screen Assist: {status} (fast local text → ASR context)"

    def _toggle_screen_context(self, _):
        self.screen_context_enabled = not self.screen_context_enabled
        self.settings["screen_context_enabled"] = self.screen_context_enabled
        self._save_settings()
        if not self.screen_context_enabled:
            self._clear_screen_context_cache()
        self._rebuild_menu()
        detail = (
            "Enabled — frontmost-window screenshots will be prefetched, fast local text will be extracted, and fresh + frequent terms will be injected into ASR."
            if self.screen_context_enabled
            else "Disabled — only local ASR will be used."
        )
        if self.screen_context_enabled:
            threading.Thread(
                target=self._refresh_screen_context_cache,
                kwargs={"reason": "enable"},
                daemon=True,
            ).start()
        rumps.notification("Voice Transcribe", "Screen Assist", detail)

    def _sound_effects_menu_title(self):
        status = "On" if self.sound_effects_enabled else "Off"
        return f"Sound Effects: {status}"

    def _toggle_sound_effects(self, _):
        self.sound_effects_enabled = not self.sound_effects_enabled
        self.settings["sound_effects_enabled"] = self.sound_effects_enabled
        self._save_settings()
        self._rebuild_menu()

    def _clear_history(self, _):
        def clear():
            self.history = []
            self._save_history()
            self._rebuild_menu()

        self._run_on_main_thread(clear)


if __name__ == "__main__":
    import atexit
    import signal

    lock_handle = LOCK_FILE.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another Voice Transcribe instance is already running; exiting duplicate launch.", flush=True)
        raise SystemExit(0)

    lock_handle.write(str(os.getpid()))
    lock_handle.flush()

    # Regular app: shows in Dock, appears in ⌘-Tab. Also keeps the menu bar icon.
    from AppKit import NSApplication, NSApplicationActivationPolicyRegular
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyRegular)

    parent_conn, child_conn = multiprocessing.Pipe()
    monitor = multiprocessing.Process(target=key_monitor.run, args=(child_conn,), daemon=True)
    monitor.start()

    app = VoiceTranscribeApp(parent_conn)
    app._key_monitor = monitor

    # Ensure all child processes are killed on exit (prevents orphan buildup).
    # Without this, each restart leaves zombie key_monitor/worker/tracker processes
    # reparented to launchd (PID 1) that accumulate indefinitely.
    def _cleanup():
        for proc in multiprocessing.active_children():
            proc.kill()
            proc.join(timeout=2)
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            lock_handle.close()
        except OSError:
            pass

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), os._exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), os._exit(0)))

    app.run()
