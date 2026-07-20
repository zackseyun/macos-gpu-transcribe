import unittest
from io import StringIO

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

    def test_launchd_managed_only_for_the_persistent_agent(self):
        self.assertTrue(
            transcribe._is_launchd_managed(
                {"XPC_SERVICE_NAME": "com.zack.voice-transcribe"}
            )
        )
        self.assertFalse(
            transcribe._is_launchd_managed(
                {"XPC_SERVICE_NAME": "application.com.zack.voice-transcribe.123"}
            )
        )
        self.assertFalse(transcribe._is_launchd_managed({}))

    def test_lock_owner_pid_is_read_without_truncation(self):
        lock_handle = StringIO("12345")
        lock_handle.seek(5)

        self.assertEqual(12345, transcribe._read_lock_owner_pid(lock_handle))
        self.assertEqual("12345", lock_handle.getvalue())

    def test_invalid_lock_owner_pid_is_ignored(self):
        self.assertIsNone(transcribe._read_lock_owner_pid(StringIO("")))
        self.assertIsNone(transcribe._read_lock_owner_pid(StringIO("not-a-pid")))
        self.assertIsNone(transcribe._read_lock_owner_pid(StringIO("-7")))


if __name__ == "__main__":
    unittest.main()
