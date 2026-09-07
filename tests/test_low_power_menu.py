import sys
import types
import unittest
from unittest import mock

import transcribe


class LowPowerMenuTest(unittest.TestCase):
    def test_low_power_title_explains_slowdown(self):
        title = transcribe._low_power_menu_title(True)
        self.assertIn("Low Power Mode: ON", title)
        self.assertIn("slower", title)

    def test_off_and_unknown_titles(self):
        self.assertIn("off", transcribe._low_power_menu_title(False))
        self.assertIn("unknown", transcribe._low_power_menu_title(None))

    def test_menu_title_uses_process_info_without_pmset(self):
        class FakeProcessInfo:
            @staticmethod
            def processInfo():
                return FakeProcessInfo()

            def isLowPowerModeEnabled(self):
                return True

        class Stub(transcribe.VoiceTranscribeApp):
            def __init__(self):  # skip rumps.App setup; only the helpers are exercised
                pass

        foundation = types.SimpleNamespace(NSProcessInfo=FakeProcessInfo)
        app = Stub()
        with mock.patch.dict(sys.modules, {"Foundation": foundation}), \
                mock.patch.object(transcribe.subprocess, "run", side_effect=AssertionError("pmset must not run")):
            self.assertTrue(app._is_low_power_mode())
            self.assertIn("ON", app._power_menu_title())


if __name__ == "__main__":
    unittest.main()
