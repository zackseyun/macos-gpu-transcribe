"""Key monitor subprocess — Fn key hold-to-record via pynput."""
import os
import threading
import time


# If recording started but no "up" within this many seconds, auto-reset.
# Prevents permanent desync from dropped Fn events.
RECORDING_TIMEOUT = 120  # 2 minutes max recording


def run(pipe):
    """Monitor Fn key. First event = start, second event = stop (toggle behavior).
    Includes auto-reset timeout to recover from state desync."""
    print(f"Key monitor process started (PID {os.getpid()})", flush=True)
    try:
        from pynput import keyboard
        print("Key monitor: pynput imported OK", flush=True)
    except Exception as e:
        print(f"Key monitor: pynput import failed: {e}", flush=True)
        return

    state = {"recording": False, "last_time": 0.0, "record_start": 0.0}
    lock = threading.Lock()

    def _auto_reset():
        """Background thread: auto-sends 'up' if recording runs too long (desync recovery)."""
        while True:
            time.sleep(5)
            with lock:
                if state["recording"] and (time.monotonic() - state["record_start"]) > RECORDING_TIMEOUT:
                    print(f"Key monitor: auto-reset after {RECORDING_TIMEOUT}s timeout", flush=True)
                    state["recording"] = False
                    try:
                        pipe.send("up")
                    except (BrokenPipeError, OSError):
                        pass

    reset_thread = threading.Thread(target=_auto_reset, daemon=True)
    reset_thread.start()

    def on_release(key):
        is_fn = (hasattr(key, 'vk') and key.vk == 63) or str(key) == '<63>'
        if not is_fn:
            return

        now = time.monotonic()
        with lock:
            # Ignore events too close together (< 50ms) — hardware bounce
            if now - state["last_time"] < 0.05:
                return
            state["last_time"] = now

            try:
                if not state["recording"]:
                    pipe.send("down")
                    state["recording"] = True
                    state["record_start"] = now
                else:
                    pipe.send("up")
                    state["recording"] = False
            except (BrokenPipeError, OSError):
                return False

    while True:
        try:
            with keyboard.Listener(on_release=on_release) as listener:
                print("Key monitor: listener active (tap Fn to start/stop recording)", flush=True)
                listener.join()
            print("Key monitor: listener stopped, restarting...", flush=True)
        except Exception as e:
            print(f"Key monitor: listener crashed ({e}), restarting...", flush=True)
        with lock:
            state["recording"] = False
        time.sleep(1)
