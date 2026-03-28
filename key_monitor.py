"""Key monitor subprocess — Fn key hold-to-record via Quartz CGEvent tap.

Uses low-level Quartz event tap to detect actual Fn key press/release state
(not toggle). This is more reliable than pynput for modifier keys on macOS.

macOS disables event taps after sleep/wake or screen lock. This module
automatically re-creates the tap when that happens.
"""
import os
import threading
import time


# Send heartbeats so parent can detect a dead monitor
HEARTBEAT_INTERVAL = 10  # seconds


def run(pipe):
    """Monitor Fn key press/release. Hold = record, release = stop.
    Auto-recovers when macOS disables the event tap."""
    print(f"Key monitor process started (PID {os.getpid()})", flush=True)

    try:
        from Quartz import (
            CGEventTapCreate,
            CGEventMaskBit,
            CGEventGetFlags,
            CGEventTapIsEnabled,
            kCGEventFlagsChanged,
            kCGEventFlagMaskSecondaryFn,
            kCGEventTapDisabledByTimeout,
            kCGEventTapDisabledByUserInput,
            kCGHeadInsertEventTap,
            kCGSessionEventTap,
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

    fn_held = [False]
    last_event_time = [0.0]
    tap_ref = [None]  # store tap reference for re-enable checks

    def callback(proxy, event_type, event, refcon):
        # macOS sends special event types when the tap is disabled
        if event_type == kCGEventTapDisabledByTimeout or event_type == kCGEventTapDisabledByUserInput:
            print(f"Key monitor: tap disabled (type={event_type}), re-enabling...", flush=True)
            if tap_ref[0] is not None:
                CGEventTapEnable(tap_ref[0], True)
            return event

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
        # Create event tap for flagsChanged events (modifier keys like Fn)
        mask = CGEventMaskBit(kCGEventFlagsChanged)
        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            0,  # active tap (can observe events)
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

        print("Key monitor: listener active (hold Fn to record, release to transcribe)", flush=True)

        # Run the event loop, but periodically check if the tap is still enabled
        while True:
            # Run for 5 seconds at a time, then check tap health
            result = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 5.0, False)

            # Check if tap is still enabled
            if not CGEventTapIsEnabled(tap):
                print("Key monitor: tap was disabled by macOS, re-enabling...", flush=True)
                CGEventTapEnable(tap, True)
                if not CGEventTapIsEnabled(tap):
                    print("Key monitor: re-enable failed, recreating tap...", flush=True)
                    # Reset Fn state in case it was held when tap died
                    if fn_held[0]:
                        fn_held[0] = False
                        try:
                            pipe.send("up")
                        except (BrokenPipeError, OSError):
                            pass
                    break  # break inner loop to recreate tap

        # Small delay before recreating
        time.sleep(1)
