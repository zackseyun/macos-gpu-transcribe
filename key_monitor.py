"""Key monitor subprocess — hold-to-record via Quartz CGEvent tap.

Supports three trigger keys:
  - Hold Fn (Globe) → fast model (Qwen3 0.6B)
  - Hold Right Option → accurate model (Qwen3 1.7B)
  - Hold Right Command → Cohere Transcribe (2B)

Uses low-level Quartz event tap to detect actual key press/release state.
Auto-recovers when macOS disables the event tap (sleep/wake, screen lock).
"""
import os
import threading
import time


# Send heartbeats so parent can detect a dead monitor
HEARTBEAT_INTERVAL = 10  # seconds

# macOS key codes
KEYCODE_RIGHT_OPTION = 61
KEYCODE_RIGHT_COMMAND = 54


def run(pipe):
    """Monitor Fn and Right Option key press/release. Hold = record, release = stop.
    Auto-recovers when macOS disables the event tap."""
    print(f"Key monitor process started (PID {os.getpid()})", flush=True)

    try:
        from Quartz import (
            CGEventTapCreate,
            CGEventMaskBit,
            CGEventGetFlags,
            CGEventGetIntegerValueField,
            CGEventTapIsEnabled,
            kCGEventFlagsChanged,
            kCGEventFlagMaskSecondaryFn,
            kCGEventFlagMaskAlternate,
            kCGEventFlagMaskCommand,
            kCGEventTapDisabledByTimeout,
            kCGEventTapDisabledByUserInput,
            kCGHeadInsertEventTap,
            kCGSessionEventTap,
            kCGKeyboardEventKeycode,
            CGEventTapEnable,
        )
        from CoreFoundation import (
            CFMachPortCreateRunLoopSource,
            CFRunLoopAddSource,
            CFRunLoopGetCurrent,
            CFRunLoopRunInMode,
            CFRunLoopStop,
            kCFRunLoopDefaultMode,
            kCFRunLoopCommonModes,
        )
        print("Key monitor: Quartz imported OK", flush=True)
    except Exception as e:
        print(f"Key monitor: Quartz import failed: {e}", flush=True)
        return

    # Track which key is currently held for recording
    # Only one can be active at a time
    active_key = [None]  # None, "fn", "ropt", or "rcmd"
    last_event_time = [0.0]
    tap_ref = [None]

    def callback(proxy, event_type, event, refcon):
        # macOS sends special event types when the tap is disabled
        if event_type == kCGEventTapDisabledByTimeout or event_type == kCGEventTapDisabledByUserInput:
            print(f"Key monitor: tap disabled (type={event_type}), re-enabling...", flush=True)
            if tap_ref[0] is not None:
                CGEventTapEnable(tap_ref[0], True)
            return event

        now = time.monotonic()
        if now - last_event_time[0] < 0.03:
            return event
        last_event_time[0] = now

        flags = CGEventGetFlags(event)
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)

        fn_now = bool(flags & kCGEventFlagMaskSecondaryFn)
        opt_now = bool(flags & kCGEventFlagMaskAlternate)
        cmd_now = bool(flags & kCGEventFlagMaskCommand)
        is_right_opt = (keycode == KEYCODE_RIGHT_OPTION)
        is_right_cmd = (keycode == KEYCODE_RIGHT_COMMAND)

        try:
            if active_key[0] is None:
                # No key held — check if one just pressed
                if fn_now:
                    active_key[0] = "fn"
                    pipe.send("down:fast")
                elif opt_now and is_right_opt:
                    active_key[0] = "ropt"
                    pipe.send("down:accurate")
                elif cmd_now and is_right_cmd:
                    active_key[0] = "rcmd"
                    pipe.send("down:cohere")
            elif active_key[0] == "fn":
                if not fn_now:
                    active_key[0] = None
                    pipe.send("up")
            elif active_key[0] == "ropt":
                if not opt_now:
                    active_key[0] = None
                    pipe.send("up")
            elif active_key[0] == "rcmd":
                if not cmd_now:
                    active_key[0] = None
                    pipe.send("up")
        except (BrokenPipeError, OSError):
            pass

        return event

    def _heartbeat():
        """Send periodic heartbeats so parent knows we're alive."""
        while True:
            time.sleep(HEARTBEAT_INTERVAL)
            try:
                pipe.send("heartbeat")
            except (BrokenPipeError, OSError):
                os._exit(0)

    threading.Thread(target=_heartbeat, daemon=True).start()

    while True:
        mask = CGEventMaskBit(kCGEventFlagsChanged)
        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            0,
            mask,
            callback,
            None,
        )

        if tap is None:
            print("Key monitor: failed to create event tap. Grant Accessibility permission in System Settings.", flush=True)
            time.sleep(5)
            continue

        tap_ref[0] = tap
        source = CFMachPortCreateRunLoopSource(None, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
        CGEventTapEnable(tap, True)

        print("Key monitor: listener active (Fn=Qwen3 0.6B, Right Opt=Qwen3 1.7B, Right Cmd=Cohere 2B)", flush=True)

        while True:
            result = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 5.0, False)

            if not CGEventTapIsEnabled(tap):
                print("Key monitor: tap was disabled by macOS, re-enabling...", flush=True)
                CGEventTapEnable(tap, True)
                if not CGEventTapIsEnabled(tap):
                    print("Key monitor: re-enable failed, recreating tap...", flush=True)
                    if active_key[0] is not None:
                        active_key[0] = None
                        try:
                            pipe.send("up")
                        except (BrokenPipeError, OSError):
                            pass
                    break

        time.sleep(1)
