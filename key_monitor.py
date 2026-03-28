"""Key monitor subprocess — Fn key hold-to-record via Quartz CGEvent tap.

Uses low-level Quartz event tap to detect actual Fn key press/release state
(not toggle). This is more reliable than pynput for modifier keys on macOS.
"""
import os
import time


def run(pipe):
    """Monitor Fn key press/release. Hold = record, release = stop."""
    print(f"Key monitor process started (PID {os.getpid()})", flush=True)

    try:
        from Quartz import (
            CGEventTapCreate,
            CGEventMaskBit,
            CGEventGetFlags,
            kCGEventFlagsChanged,
            kCGEventFlagMaskSecondaryFn,
            kCGHeadInsertEventTap,
            kCGSessionEventTap,
            CGEventTapEnable,
        )
        from CoreFoundation import (
            CFMachPortCreateRunLoopSource,
            CFRunLoopAddSource,
            CFRunLoopGetCurrent,
            CFRunLoopRun,
            kCFRunLoopCommonModes,
        )
        print("Key monitor: Quartz imported OK", flush=True)
    except Exception as e:
        print(f"Key monitor: Quartz import failed: {e}", flush=True)
        return

    fn_held = [False]
    last_event_time = [0.0]

    def callback(proxy, event_type, event, refcon):
        now = time.monotonic()
        # Debounce: ignore events within 30ms of each other
        if now - last_event_time[0] < 0.03:
            return event
        last_event_time[0] = now

        flags = CGEventGetFlags(event)
        fn_now = bool(flags & kCGEventFlagMaskSecondaryFn)

        if fn_now and not fn_held[0]:
            # Fn just pressed — start recording
            fn_held[0] = True
            try:
                pipe.send("down")
            except (BrokenPipeError, OSError):
                pass
        elif not fn_now and fn_held[0]:
            # Fn just released — stop recording
            fn_held[0] = False
            try:
                pipe.send("up")
            except (BrokenPipeError, OSError):
                pass

        return event

    # Create event tap for flagsChanged events (modifier keys like Fn)
    mask = CGEventMaskBit(kCGEventFlagsChanged)
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        0,  # listenOnly=0 means we can observe (not block) events
        mask,
        callback,
        None,
    )

    if tap is None:
        print("Key monitor: failed to create event tap. Grant Accessibility permission in System Settings.", flush=True)
        return

    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)

    print("Key monitor: listener active (hold Fn to record, release to transcribe)", flush=True)
    CFRunLoopRun()
    print("Key monitor: run loop exited", flush=True)
