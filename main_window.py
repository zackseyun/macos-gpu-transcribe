"""Main app window — visible status, settings toggles, and transcription history.

Opens when the Dock icon is clicked or from the menu bar's "Open Window" item.
State updates by polling the app via a 0.25s timer; this is plenty fast for a
status UI and avoids threading churn.
"""

import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSScrollView,
    NSStackView,
    NSSwitchButton,
    NSTableColumn,
    NSTableView,
    NSTextField,
    NSUserInterfaceLayoutOrientationVertical,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSMakeRect, NSObject, NSTimer

WINDOW_WIDTH = 420
WINDOW_HEIGHT = 520


def _label(text, size=13, bold=False, color=None):
    tf = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 0, 0))
    tf.setStringValue_(text)
    tf.setBezeled_(False)
    tf.setDrawsBackground_(False)
    tf.setEditable_(False)
    tf.setSelectable_(False)
    if bold:
        tf.setFont_(NSFont.boldSystemFontOfSize_(size))
    else:
        tf.setFont_(NSFont.systemFontOfSize_(size))
    if color is not None:
        tf.setTextColor_(color)
    return tf


class HistoryDataSource(NSObject):
    """NSTableView data source backed by the app's history list."""

    def initWithApp_(self, app):  # noqa: N802
        self = objc.super(HistoryDataSource, self).init()
        if self is None:
            return None
        self._app = app
        return self

    def numberOfRowsInTableView_(self, _tv):  # noqa: N802
        return len(self._app.history)

    def tableView_objectValueForTableColumn_row_(self, _tv, column, row):  # noqa: N802
        try:
            entry = self._app.history[row]
        except IndexError:
            return ""
        col_id = str(column.identifier())
        if col_id == "time":
            # entries are dicts with "timestamp" (ISO) and "text"
            ts = entry.get("timestamp", "")
            # keep only "HH:MM"
            return ts.split("T")[1][:5] if "T" in ts else ts
        return entry.get("text", "")


class MainWindowController(NSObject):
    """Builds + owns the main window. Attach via `attach_app_(app)` after init."""

    def init(self):
        self = objc.super(MainWindowController, self).init()
        if self is None:
            return None
        self._app = None
        self._window = None
        self._status_label = None
        self._model_label = None
        self._sound_switch = None
        self._screen_ctx_switch = None
        self._history_table = None
        self._history_source = None
        self._refresh_timer = None
        self._last_history_len = -1
        return self

    def attachApp_(self, app):  # noqa: N802
        self._app = app

    # ── Building ──

    def buildWindow(self):  # noqa: N802
        rect = NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False
        )
        self._window.setTitle_("Voice Transcribe")
        self._window.setReleasedWhenClosed_(False)
        self._window.center()
        self._window.setMinSize_((360, 360))

        # Root vertical stack (padding from content view)
        content = NSView.alloc().initWithFrame_(rect)
        self._window.setContentView_(content)

        stack = NSStackView.alloc().initWithFrame_(rect)
        stack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        stack.setSpacing_(10)
        stack.setEdgeInsets_((16, 16, 16, 16))
        stack.setAlignment_(1)  # NSLayoutAttributeLeading
        stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        content.addSubview_(stack)

        # Pin stack to content edges
        content.addConstraints_([
            _pin(stack, content, "top", 0),
            _pin(stack, content, "bottom", 0),
            _pin(stack, content, "leading", 0),
            _pin(stack, content, "trailing", 0),
        ])

        # ── Header ──
        stack.addArrangedSubview_(_label("Voice Transcribe", size=17, bold=True))
        self._status_label = _label("●  Idle", size=13, color=NSColor.secondaryLabelColor())
        stack.addArrangedSubview_(self._status_label)
        self._model_label = _label("", size=11, color=NSColor.tertiaryLabelColor())
        stack.addArrangedSubview_(self._model_label)

        stack.addArrangedSubview_(_divider())

        # ── Hotkeys ──
        stack.addArrangedSubview_(_label("Hotkeys", size=13, bold=True))
        stack.addArrangedSubview_(
            _label("Hold Fn  →  record (Cohere 2B)", size=12, color=NSColor.secondaryLabelColor())
        )
        stack.addArrangedSubview_(
            _label(
                "Hold Right ⌥  →  record (Qwen 1.7B)",
                size=12,
                color=NSColor.secondaryLabelColor(),
            )
        )

        stack.addArrangedSubview_(_divider())

        # ── Settings ──
        stack.addArrangedSubview_(_label("Settings", size=13, bold=True))

        self._sound_switch = _checkbox("Sound effects", self, b"toggleSound:")
        stack.addArrangedSubview_(self._sound_switch)

        self._screen_ctx_switch = _checkbox(
            "Screen context (inject on-screen text into ASR)", self, b"toggleScreenCtx:"
        )
        stack.addArrangedSubview_(self._screen_ctx_switch)

        stack.addArrangedSubview_(_divider())

        # ── History ──
        stack.addArrangedSubview_(_label("Recent transcriptions", size=13, bold=True))

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, WINDOW_WIDTH - 32, 180))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(1)  # NSLineBorder
        scroll.setTranslatesAutoresizingMaskIntoConstraints_(False)

        table = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, WINDOW_WIDTH - 32, 180))
        table.setUsesAlternatingRowBackgroundColors_(True)
        table.setRowHeight_(20)

        time_col = NSTableColumn.alloc().initWithIdentifier_("time")
        time_col.setWidth_(58)
        time_col.headerCell().setStringValue_("Time")
        table.addTableColumn_(time_col)

        text_col = NSTableColumn.alloc().initWithIdentifier_("text")
        text_col.setWidth_(320)
        text_col.headerCell().setStringValue_("Text")
        table.addTableColumn_(text_col)

        scroll.setDocumentView_(table)
        stack.addArrangedSubview_(scroll)

        # Pin history width + flexible height
        scroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        scroll.heightAnchor().constraintGreaterThanOrEqualToConstant_(120).setActive_(True)

        self._history_table = table

        if self._app is not None:
            self._history_source = HistoryDataSource.alloc().initWithApp_(self._app)
            table.setDataSource_(self._history_source)
            self._syncSettingsFromApp()

        return self._window

    # ── Show / lifecycle ──

    def showWindow(self):  # noqa: N802
        if self._window is None:
            self.buildWindow()
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        if self._refresh_timer is None:
            self._refresh_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.25, self, b"refreshTick:", None, True
            )
        # First time: observe app activation so clicking the Dock icon reopens
        # this window if the user closed it.
        if not getattr(self, "_did_observe_activation", False):
            from Foundation import NSNotificationCenter
            NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
                self, b"appDidBecomeActive:", "NSApplicationDidBecomeActiveNotification", None
            )
            self._did_observe_activation = True

    def appDidBecomeActive_(self, _notification):  # noqa: N802
        # Reopen the main window if no visible windows (typical after user closes it).
        if self._window is None or not self._window.isVisible():
            if self._window is None:
                self.buildWindow()
            self._window.makeKeyAndOrderFront_(None)

    def refreshTick_(self, _timer):  # noqa: N802
        if self._app is None or self._window is None:
            return
        if not self._window.isVisible():
            return

        # Status
        if self._app.is_recording:
            self._status_label.setStringValue_("●  Recording")
            self._status_label.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.25, 0.25, 1.0)
            )
        elif self._app.is_processing:
            self._status_label.setStringValue_("●  Transcribing…")
            self._status_label.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.72, 0.2, 1.0)
            )
        else:
            self._status_label.setStringValue_("●  Idle")
            self._status_label.setTextColor_(NSColor.secondaryLabelColor())

        mode = getattr(self._app, "_pending_model", None) or "(none)"
        self._model_label.setStringValue_(f"Model: {mode}")

        if len(self._app.history) != self._last_history_len:
            self._last_history_len = len(self._app.history)
            if self._history_table is not None:
                self._history_table.reloadData()

    def _syncSettingsFromApp(self):  # noqa: N802
        self._sound_switch.setState_(1 if self._app.sound_effects_enabled else 0)
        self._screen_ctx_switch.setState_(1 if self._app.screen_context_enabled else 0)

    # ── Switch callbacks ──

    def toggleSound_(self, sender):  # noqa: N802
        if self._app is None:
            return
        self._app.sound_effects_enabled = bool(sender.state())
        self._app.settings["sound_effects_enabled"] = self._app.sound_effects_enabled
        self._app._save_settings()

    def toggleScreenCtx_(self, sender):  # noqa: N802
        if self._app is None:
            return
        self._app.screen_context_enabled = bool(sender.state())
        self._app.settings["screen_context_enabled"] = self._app.screen_context_enabled
        self._app._save_settings()


# ── Layout helpers ──


def _divider():
    v = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WINDOW_WIDTH - 32, 1))
    v.setWantsLayer_(True)
    v.layer().setBackgroundColor_(
        NSColor.colorWithCalibratedWhite_alpha_(0.5, 0.22).CGColor()
    )
    v.heightAnchor().constraintEqualToConstant_(1).setActive_(True)
    return v


def _checkbox(title, target, action):
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 0, 0))
    btn.setButtonType_(NSSwitchButton)
    btn.setTitle_(title)
    btn.setTarget_(target)
    btn.setAction_(action)
    return btn


def _pin(child, parent, edge, constant):
    from AppKit import NSLayoutConstraint

    attr_map = {
        "top": 3,
        "bottom": 4,
        "leading": 5,
        "trailing": 6,
    }
    a = attr_map[edge]
    return NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
        child, a, 0, parent, a, 1.0, constant
    )


_controller = None


def get_controller():
    global _controller
    if _controller is None:
        _controller = MainWindowController.alloc().init()
    return _controller
