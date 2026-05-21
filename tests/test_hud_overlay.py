import unittest
from types import SimpleNamespace

import hud_overlay


def rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def size(width, height):
    return SimpleNamespace(width=width, height=height)


class FakeScreen:
    def __init__(self, frame_rect, visible_rect=None):
        self._frame = frame_rect
        self._visible = visible_rect or frame_rect

    def frame(self):
        return self._frame

    def visibleFrame(self):
        return self._visible


class HudOverlayPositionTest(unittest.TestCase):
    def test_cursor_on_monitor_above_uses_that_monitor_bounds(self):
        screens = [
            FakeScreen(rect(0, 0, 1440, 900)),
            FakeScreen(rect(0, 900, 1440, 900)),
        ]

        x, y = hud_overlay._window_origin_near_cursor(
            point(500, 1300),
            size(hud_overlay.HUD_WIDTH, hud_overlay.HUD_HEIGHT),
            screens=screens,
        )

        self.assertEqual(x, 522.0)
        self.assertEqual(y, 1240.0)
        self.assertGreaterEqual(y, 900.0)

    def test_cursor_on_left_monitor_can_move_to_negative_coordinates(self):
        screens = [
            FakeScreen(rect(-1920, 0, 1920, 1080)),
            FakeScreen(rect(0, 0, 1440, 900)),
        ]

        x, y = hud_overlay._window_origin_near_cursor(
            point(-1000, 500),
            size(hud_overlay.HUD_WIDTH, hud_overlay.HUD_HEIGHT),
            screens=screens,
        )

        self.assertEqual(x, -978.0)
        self.assertEqual(y, 440.0)
        self.assertLess(x, 0.0)

    def test_clamps_to_cursor_screen_edge_not_previous_window_screen(self):
        screens = [
            FakeScreen(rect(0, 0, 1440, 900)),
            FakeScreen(rect(0, 900, 1440, 900)),
        ]

        x, y = hud_overlay._window_origin_near_cursor(
            point(1438, 1750),
            size(hud_overlay.HUD_WIDTH, hud_overlay.HUD_HEIGHT),
            screens=screens,
        )

        self.assertEqual(x, 1236.0)
        self.assertEqual(y, 1690.0)


if __name__ == "__main__":
    unittest.main()
