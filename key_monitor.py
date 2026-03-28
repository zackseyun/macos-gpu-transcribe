"""Key monitor subprocess — Fn key hold-to-record via pynput."""
import os
import time


def run(pipe):
    """Monitor Fn key. First event = start, second event = stop (hold behavior)."""
    print(f"Key monitor process started (PID {os.getpid()})", flush=True)
    try:
        from pynput import keyboard
        print("Key monitor: pynput imported OK", flush=True)
    except Exception as e:
        print(f"Key monitor: pynput import failed: {e}", flush=True)
        return

    # macOS sends TWO flagsChanged (release) events per Fn press/release:
    #   press down  → release event #1
    #   release up  → release event #2
    # We use odd count = "down" (start), even count = "up" (stop)
    state = {"recording": False, "last_time": 0.0}

    def on_release(key):
        is_fn = (hasattr(key, 'vk') and key.vk == 63) or str(key) == '<63>'
        if not is_fn:
            return

        now = time.monotonic()
        # Ignore events too close together (< 50ms) — hardware bounce
        if now - state["last_time"] < 0.05:
            return
        state["last_time"] = now

        try:
            if not state["recording"]:
                pipe.send("down")
                state["recording"] = True
            else:
                pipe.send("up")
                state["recording"] = False
        except (BrokenPipeError, OSError):
            return False

    while True:
        try:
            with keyboard.Listener(on_release=on_release) as listener:
                print("Key monitor: listener active (hold Fn to record)", flush=True)
                listener.join()
            print("Key monitor: listener stopped, restarting...", flush=True)
        except Exception as e:
            print(f"Key monitor: listener crashed ({e}), restarting...", flush=True)
        state["recording"] = False
        time.sleep(1)
