"""Fn-key monitor subprocess for hold-to-record.

Polling the physical Quartz key state is deliberately used instead of a
CGEvent tap. Recent macOS versions can repeatedly disable listen-only event
taps even when Input Monitoring is granted; key-state polling stays active and
is sufficient because this app only needs the Fn press/release state.
"""

import os
import time


POLL_INTERVAL = 0.01
HEARTBEAT_INTERVAL = 10.0
FN_KEYCODE = 63


def run(pipe):
    """Send ``down:fast`` and ``up`` messages as the Fn modifier changes."""
    print(f"Key monitor process started (PID {os.getpid()})", flush=True)

    try:
        from Quartz import (
            CGEventSourceKeyState,
            kCGEventSourceStateHIDSystemState,
        )
        print("Key monitor: Quartz imported OK", flush=True)
    except Exception as exc:
        print(f"Key monitor: Quartz import failed: {exc}", flush=True)
        return

    is_held = False
    last_heartbeat = time.monotonic()
    print("Key monitor: polling active (Fn=Qwen3 0.6B)", flush=True)

    while True:
        try:
            # Arrow/navigation keys can also set SecondaryFn. Query the actual
            # Fn key instead so moving the cursor never starts a recording.
            fn_now = bool(CGEventSourceKeyState(
                kCGEventSourceStateHIDSystemState, FN_KEYCODE
            ))

            if fn_now and not is_held:
                is_held = True
                pipe.send("down:fast")
            elif is_held and not fn_now:
                is_held = False
                pipe.send("up")

            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                pipe.send("heartbeat")
                last_heartbeat = now

            time.sleep(POLL_INTERVAL)
        except (BrokenPipeError, EOFError, OSError):
            return
        except Exception as exc:
            print(f"Key monitor: polling error: {exc}", flush=True)
            time.sleep(1.0)
