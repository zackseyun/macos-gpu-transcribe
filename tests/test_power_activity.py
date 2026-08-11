import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import transcribe


class PowerActivityTest(unittest.TestCase):
    def test_app_nap_activity_explicitly_allows_idle_system_sleep(self):
        calls = []
        activity = object()

        class FakeProcessInfo:
            @staticmethod
            def processInfo():
                return FakeProcessInfo()

            def beginActivityWithOptions_reason_(self, options, reason):
                calls.append((options, reason))
                return activity

        foundation = types.SimpleNamespace(NSProcessInfo=FakeProcessInfo)
        app = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {"Foundation": foundation}):
            transcribe.VoiceTranscribeApp._disable_app_nap_allowing_system_sleep(app)

        options, reason = calls[0]
        self.assertEqual(0, options & (1 << 20))
        self.assertEqual(0xFF00000000, options & 0xFF00000000)
        self.assertEqual(
            "Keep ASR hotkey responsive while system is awake", reason
        )
        self.assertIs(activity, app._app_nap_activity)

    def test_display_state_probe_fails_open_for_recording(self):
        with mock.patch.dict(sys.modules, {"Quartz": None}):
            self.assertFalse(transcribe._is_main_display_asleep())

    def test_display_wake_clears_marker_and_reopens_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "display-asleep"
            marker.touch()
            refreshes = []
            app = types.SimpleNamespace(
                _audio_stream=None,
                _refresh_audio_stream=lambda reason, force=False: refreshes.append(
                    (reason, force)
                ),
            )
            with mock.patch.object(transcribe, "DISPLAY_SLEEP_MARKER", marker):
                transcribe.VoiceTranscribeApp._resume_audio_after_display_wake(app)

            self.assertFalse(marker.exists())
            self.assertEqual([("display-wake: no stream", True)], refreshes)


if __name__ == "__main__":
    unittest.main()
