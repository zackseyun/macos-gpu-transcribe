import unittest

import transcribe


class PasteSafetyTest(unittest.TestCase):
    def test_restore_delay_defaults_to_slow_target_grace_window(self):
        self.assertGreaterEqual(transcribe.PASTEBOARD_RESTORE_DELAY, 2.0)

    def test_restore_allowed_only_for_current_unchanged_pasteboard(self):
        self.assertTrue(
            transcribe._should_restore_clipboard_after_paste(
                paste_generation=7,
                current_generation=7,
                current_contents="new transcription",
                expected_text="new transcription",
                old_contents="previous clipboard",
            )
        )

    def test_restore_skips_old_generation(self):
        self.assertFalse(
            transcribe._should_restore_clipboard_after_paste(
                paste_generation=6,
                current_generation=7,
                current_contents="new transcription",
                expected_text="new transcription",
                old_contents="previous clipboard",
            )
        )

    def test_restore_skips_when_user_or_app_changed_clipboard(self):
        self.assertFalse(
            transcribe._should_restore_clipboard_after_paste(
                paste_generation=7,
                current_generation=7,
                current_contents="user copied something else",
                expected_text="new transcription",
                old_contents="previous clipboard",
            )
        )

    def test_restore_skips_non_text_previous_clipboard(self):
        self.assertFalse(
            transcribe._should_restore_clipboard_after_paste(
                paste_generation=7,
                current_generation=7,
                current_contents="new transcription",
                expected_text="new transcription",
                old_contents=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
