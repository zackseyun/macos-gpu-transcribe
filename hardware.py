"""Hardware profile used to pick per-machine defaults.

This checkout runs on very different Apple Silicon Macs (a 128GB M4 Max MacBook
Pro and a MacBook Air). Hard-coding a single model default in git meant every
pull flipped the other machine's Fn behavior, so defaults derive from the
machine instead:

- Cohere Transcribe MLX 8-bit is a 2B model: ~3.8GB of weights plus ~1.4GB of
  scratch, ~40x real-time on an M4 Max. Best accuracy; the right default when
  the machine has memory to spare.
- Qwen3-ASR 0.6B 4-bit is ~0.5GB and faster per clip but weaker on names; the
  right default on 8-24GB machines where wiring 5GB+ would starve everything.

Every rule here is a pure function of the numbers so tests can cover the
thresholds without depending on the test machine.
"""
from __future__ import annotations

import os
import subprocess

GIB = 1024 ** 3

# Machines at or above this much unified memory default to Cohere 8-bit and
# wire MLX memory so the weights stay resident between dictations.
LARGE_MEMORY_GB = float(os.getenv("VOICE_TRANSCRIBE_COHERE_MIN_MEMORY_GB", "32"))
LARGE_MEMORY_MODEL_MODE = "cohere"
SMALL_MEMORY_MODEL_MODE = "fast"
DEFAULT_WIRED_LIMIT_GB = 8.0


def unified_memory_bytes() -> int:
    """Physical (unified) memory in bytes, or 0 when it cannot be determined."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0


def chip_name() -> str:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        return proc.stdout.strip() or "unknown chip"
    except Exception:
        return "unknown chip"


def describe(memory_bytes: int | None = None, chip: str | None = None) -> str:
    memory_bytes = unified_memory_bytes() if memory_bytes is None else memory_bytes
    chip = chip_name() if chip is None else chip
    return f"{chip}, {memory_bytes / GIB:.0f}GB unified memory"


def is_large_memory_machine(memory_bytes: int | None = None) -> bool:
    memory_bytes = unified_memory_bytes() if memory_bytes is None else memory_bytes
    return memory_bytes >= LARGE_MEMORY_GB * GIB


def recommended_default_model_mode(
    memory_bytes: int | None = None,
    valid_modes=None,
) -> str:
    """Model mode the Fn key should use when settings.json says "auto".

    VOICE_TRANSCRIBE_DEFAULT_MODEL_MODE forces a mode (when it is one of
    valid_modes, if given); otherwise the memory rule above decides.
    """
    forced = os.getenv("VOICE_TRANSCRIBE_DEFAULT_MODEL_MODE", "").strip().lower()
    if forced and (valid_modes is None or forced in valid_modes):
        return forced
    if is_large_memory_machine(memory_bytes):
        return LARGE_MEMORY_MODEL_MODE
    return SMALL_MEMORY_MODEL_MODE


def recommended_wired_limit_bytes(
    memory_bytes: int | None = None,
    requested_gb: float = DEFAULT_WIRED_LIMIT_GB,
) -> int:
    """MLX wired-memory limit: keep Cohere resident on roomy machines, off elsewhere."""
    if is_large_memory_machine(memory_bytes):
        return int(requested_gb * GIB)
    return 0
