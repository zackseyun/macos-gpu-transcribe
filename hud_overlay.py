"""Floating HUD overlay — follows cursor, shows live waveform + recording state.

Runs entirely on the main thread (AppKit requirement). `push_level` is
thread-safe so the PortAudio callback can feed it levels directly.
"""

import math
import threading

import objc
from AppKit import (
    NSBezierPath,
    NSColor,
    NSEvent,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSScreenSaverWindowLevel,
    NSString,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSMakeRect, NSObject, NSTimer

NSBackingStoreBuffered = 2
NSWindowStyleMaskBorderless = 0

HUD_WIDTH = 200.0
HUD_HEIGHT = 52.0
BAR_COUNT = 36
BAR_WIDTH = 2.0
BAR_GAP = 2.0
CURSOR_OFFSET_X = 22.0
CURSOR_OFFSET_Y = -HUD_HEIGHT - 8.0  # below the cursor
# Visual smoothing — each new level is blended with the existing rolling peak
# so bars rise fast but decay smoothly instead of flickering frame-to-frame.
LEVEL_ATTACK = 0.85   # how quickly a rising level is accepted (higher = snappier)
LEVEL_RELEASE = 0.25  # how quickly bars decay when audio drops
# Push throttle: limit updates to ~25 Hz so the waveform scrolls calmly.
MIN_PUSH_INTERVAL = 1.0 / 25.0


class WaveformView(NSView):
    def initWithFrame_(self, frame):  # noqa: N802
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._levels = [0.0] * BAR_COUNT
        self._state = "recording"  # recording | processing
        self._label = ""  # text shown when in processing state
        self._lock = threading.Lock()
        self._pulse_phase = 0.0
        self._last_push_time = 0.0
        return self

    def pushLevel_(self, level):  # noqa: N802
        import time as _time
        now = _time.monotonic()
        with self._lock:
            if now - self._last_push_time < MIN_PUSH_INTERVAL:
                # Throttle: bump the last bar up toward this level without adding
                # a new column. Keeps the waveform from flickering at 100 Hz.
                if self._levels:
                    prev = self._levels[-1]
                    self._levels[-1] = _blend(prev, float(level))
                return
            self._last_push_time = now
            blended = _blend(self._levels[-1] if self._levels else 0.0, float(level))
            self._levels.append(blended)
            if len(self._levels) > BAR_COUNT:
                self._levels = self._levels[-BAR_COUNT:]

    def settleActiveBar(self):  # noqa: N802
        """Decay only the rightmost (active) bar toward 0 when no audio arrives.

        Older bars are snapshots of past audio — they scroll off the left edge
        unchanged, preserving the shape of what was spoken. The active bar
        gradually falls during silence instead of pinning at the last peak.
        """
        with self._lock:
            if self._levels:
                self._levels[-1] = max(0.0, self._levels[-1] * (1.0 - LEVEL_RELEASE * 0.5))

    def resetLevels(self):  # noqa: N802
        """Clear the waveform so a newly shown HUD starts visually empty."""
        with self._lock:
            self._levels = [0.0] * BAR_COUNT
            self._pulse_phase = 0.0
            self._last_push_time = 0.0

    def setState_(self, state):  # noqa: N802
        with self._lock:
            self._state = state

    def setLabel_(self, label):  # noqa: N802
        with self._lock:
            self._label = label or ""

    def drawRect_(self, _rect):  # noqa: N802
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height

        # Background: dark rounded rect with subtle border
        bg = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, 12.0, 12.0)
        NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.82).setFill()
        bg.fill()
        NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.12).setStroke()
        bg.setLineWidth_(1.0)
        bg.stroke()

        with self._lock:
            levels = list(self._levels)
            state = self._state
            label = self._label
            self._pulse_phase = (self._pulse_phase + 0.10) % (2 * math.pi)
            pulse = self._pulse_phase

        # State dot (left side)
        dot_d = 8.0
        dot_x = 10.0
        dot_y = h / 2 - dot_d / 2
        dot_path = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(dot_x, dot_y, dot_d, dot_d)
        )
        if state == "recording":
            alpha = 0.75 + 0.25 * math.sin(pulse)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.22, 0.22, alpha).setFill()
        else:  # processing
            alpha = 0.75 + 0.25 * math.sin(pulse * 1.4)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.78, 0.2, alpha).setFill()
        dot_path.fill()

        content_x = dot_x + dot_d + 8.0
        content_w = w - content_x - 10.0

        if state == "processing":
            # Draw text label centered in the right-hand area
            if not label:
                label = "Transcribing…"
            text_attrs = {
                NSFontAttributeName: NSFont.systemFontOfSize_(12),
                NSForegroundColorAttributeName: NSColor.colorWithCalibratedWhite_alpha_(
                    0.92, 0.9
                ),
            }
            ns_label = NSString.stringWithString_(label)
            size = ns_label.sizeWithAttributes_(text_attrs)
            tx = content_x + (content_w - size.width) / 2
            ty = (h - size.height) / 2
            ns_label.drawAtPoint_withAttributes_((tx, ty), text_attrs)
            return

        # Waveform bars (recording state)
        total_w = BAR_COUNT * (BAR_WIDTH + BAR_GAP) - BAR_GAP
        start_x = content_x + (content_w - total_w) / 2
        mid_y = h / 2
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.35, 0.35, 0.95).setFill()

        for i, lvl in enumerate(levels):
            lvl = max(0.0, min(1.0, lvl))
            # Tiny baseline so fully silent frames show a faint line, not nothing.
            bar_h = max(1.0, lvl * (h * 0.65))
            x = start_x + i * (BAR_WIDTH + BAR_GAP)
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, mid_y - bar_h / 2, BAR_WIDTH, bar_h), 1.0, 1.0
            )
            path.fill()


def _blend(prev, new):
    """Attack/release smoothing: fast on rising edges, slow on decay."""
    if new >= prev:
        return prev + (new - prev) * LEVEL_ATTACK
    return prev + (new - prev) * LEVEL_RELEASE


class HUDController(NSObject):
    """Controls show/hide, cursor tracking, and audio level pushing."""

    def init(self):
        self = objc.super(HUDController, self).init()
        if self is None:
            return None
        self._window = None
        self._view = None
        self._visible = False
        self._follow_timer = None
        self._redraw_timer = None
        self._follow_cursor = True
        return self

    def _ensure_window(self):
        if self._window is not None:
            return
        rect = NSMakeRect(0, 0, HUD_WIDTH, HUD_HEIGHT)
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setLevel_(NSScreenSaverWindowLevel)
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(True)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        # Hide from ⌘-Tab and window menu
        self._window.setHidesOnDeactivate_(False)
        self._view = WaveformView.alloc().initWithFrame_(rect)
        self._window.setContentView_(self._view)

    def show(self):
        self._ensure_window()
        if self._view is not None:
            self._view.resetLevels()
            self._view.setState_("recording")
            self._view.setLabel_("")
            self._view.setNeedsDisplay_(True)
        self._visible = True
        self._positionNearCursor()
        self._window.orderFrontRegardless()
        self._startTimers()

    def setProcessing(self):  # noqa: N802
        self.setProcessingWithLabel_("Transcribing…")

    def setProcessingWithLabel_(self, label):  # noqa: N802
        self._ensure_window()
        if self._view is not None:
            self._view.setState_("processing")
            self._view.setLabel_(label)
        if not self._visible:
            # Show even if not previously visible (covers the silence-gate path)
            self._visible = True
            self._positionNearCursor()
            self._window.orderFrontRegardless()
            self._startTimers()

    def hide(self):
        self._visible = False
        self._stopTimers()
        if self._window is not None:
            self._window.orderOut_(None)

    def pushLevel_(self, level):  # noqa: N802
        if self._view is not None and self._visible:
            self._view.pushLevel_(level)

    def _startTimers(self):  # noqa: N802
        if self._follow_timer is None and self._follow_cursor:
            self._follow_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / 30, self, b"_followTick:", None, True
            )
        if self._redraw_timer is None:
            self._redraw_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / 30, self, b"_redrawTick:", None, True
            )

    def _stopTimers(self):  # noqa: N802
        if self._follow_timer is not None:
            self._follow_timer.invalidate()
            self._follow_timer = None
        if self._redraw_timer is not None:
            self._redraw_timer.invalidate()
            self._redraw_timer = None

    def _positionNearCursor(self):  # noqa: N802
        if self._window is None:
            return
        loc = NSEvent.mouseLocation()
        x = loc.x + CURSOR_OFFSET_X
        y = loc.y + CURSOR_OFFSET_Y
        # Keep inside the main screen bounds. AppKit uses lower-left origin.
        frame = self._window.frame()
        screen = self._window.screen() or _main_screen()
        if screen is not None:
            vis = screen.visibleFrame()
            x = max(vis.origin.x + 4, min(x, vis.origin.x + vis.size.width - frame.size.width - 4))
            y = max(vis.origin.y + 4, min(y, vis.origin.y + vis.size.height - frame.size.height - 4))
        self._window.setFrameOrigin_((x, y))

    def _followTick_(self, _timer):  # noqa: N802
        if self._visible:
            self._positionNearCursor()

    def _redrawTick_(self, _timer):  # noqa: N802
        if self._view is not None and self._visible:
            # Let only the rightmost (active) bar drift to 0 during silence.
            # Past bars stay at their captured level and just scroll off-screen,
            # preserving the shape of what was spoken rather than melting away.
            self._view.settleActiveBar()
            self._view.setNeedsDisplay_(True)


def _main_screen():
    from AppKit import NSScreen
    return NSScreen.mainScreen()


_controller = None
_controller_lock = threading.Lock()


def get_controller():
    """Lazy-initialize a shared HUDController. Must be called from main thread first."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = HUDController.alloc().init()
        return _controller
