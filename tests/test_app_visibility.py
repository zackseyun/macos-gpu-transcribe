import unittest

import transcribe


class AppVisibilityDefaultsTest(unittest.TestCase):
    def test_quiet_menu_bar_mode_is_default(self):
        env = {}
        argv = ["transcribe.py"]

        self.assertFalse(transcribe._should_show_main_window_on_launch(argv=argv, environ=env))
        self.assertFalse(transcribe._should_show_dock_icon(argv=argv, environ=env))
        self.assertFalse(transcribe._should_show_main_window_on_launch())
        self.assertFalse(transcribe._should_show_dock_icon())

    def test_window_can_be_requested_explicitly(self):
        self.assertTrue(
            transcribe._should_show_main_window_on_launch(
                argv=["transcribe.py", "--show-window"],
                environ={},
            )
        )
        self.assertTrue(
            transcribe._should_show_main_window_on_launch(
                argv=["transcribe.py"],
                environ={"VOICE_TRANSCRIBE_SHOW_WINDOW_ON_LAUNCH": "1"},
            )
        )

    def test_dock_icon_can_be_requested_explicitly(self):
        self.assertTrue(
            transcribe._should_show_dock_icon(
                argv=["transcribe.py", "--show-dock"],
                environ={},
            )
        )
        self.assertTrue(
            transcribe._should_show_dock_icon(
                argv=["transcribe.py"],
                environ={"VOICE_TRANSCRIBE_SHOW_DOCK_ICON": "1"},
            )
        )


if __name__ == "__main__":
    unittest.main()
