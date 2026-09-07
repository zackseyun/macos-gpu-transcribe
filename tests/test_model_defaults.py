import os
import sys
import types
import unittest
from unittest import mock

import hardware
import transcribe

GIB = 1024 ** 3


class HardwareDefaultTest(unittest.TestCase):
    def test_large_memory_machines_default_to_cohere_8bit(self):
        with mock.patch.dict(os.environ, {"VOICE_TRANSCRIBE_DEFAULT_MODEL_MODE": ""}):
            for gb in (32, 36, 48, 64, 128, 192):
                self.assertEqual("cohere", hardware.recommended_default_model_mode(gb * GIB), gb)

    def test_small_memory_machines_default_to_qwen_fast(self):
        with mock.patch.dict(os.environ, {"VOICE_TRANSCRIBE_DEFAULT_MODEL_MODE": ""}):
            for gb in (8, 16, 18, 24):
                self.assertEqual("fast", hardware.recommended_default_model_mode(gb * GIB), gb)

    def test_env_override_wins_only_when_valid(self):
        with mock.patch.dict(os.environ, {"VOICE_TRANSCRIBE_DEFAULT_MODEL_MODE": "fast"}):
            self.assertEqual("fast", hardware.recommended_default_model_mode(128 * GIB))
        with mock.patch.dict(os.environ, {"VOICE_TRANSCRIBE_DEFAULT_MODEL_MODE": "bogus"}):
            self.assertEqual(
                "cohere",
                hardware.recommended_default_model_mode(128 * GIB, valid_modes=transcribe.MODEL_LABELS),
            )

    def test_wired_limit_only_on_large_memory_machines(self):
        self.assertEqual(8 * GIB, hardware.recommended_wired_limit_bytes(128 * GIB))
        self.assertEqual(8 * GIB, hardware.recommended_wired_limit_bytes(32 * GIB))
        self.assertEqual(0, hardware.recommended_wired_limit_bytes(24 * GIB))
        self.assertEqual(0, hardware.recommended_wired_limit_bytes(16 * GIB))

    def test_module_default_matches_this_machine(self):
        self.assertEqual(
            hardware.recommended_default_model_mode(valid_modes=transcribe.MODEL_LABELS),
            transcribe.DEFAULT_MODEL_MODE,
        )
        self.assertEqual(transcribe.DEFAULT_MODEL_MODE, transcribe.MENU_MODEL_MODES[0])
        self.assertEqual(set(transcribe.MENU_MODEL_MODES),
                         {"cohere", "fast", "cohere-swift-4bit", "cohere-pytorch", "granite"})
        self.assertIn("GB unified memory", transcribe.HARDWARE_PROFILE)


class ModelLabelsTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(transcribe.MODEL_LABELS["fast"], "Qwen3-ASR 0.6B (MLX)")
        self.assertEqual(transcribe.MODEL_LABELS["cohere"], "Cohere Transcribe MLX 8-bit")
        self.assertEqual(transcribe.MODEL_LABELS["cohere-swift-4bit"], "Cohere Transcribe Swift 4-bit")


class DefaultModelSettingTest(unittest.TestCase):
    def test_auto_missing_or_unknown_follow_hardware_rule(self):
        for value in ("auto", "AUTO", None, "", "bogus"):
            self.assertEqual((True, transcribe.DEFAULT_MODEL_MODE), transcribe._resolve_default_model_setting(value), value)

    def test_known_modes_are_pins(self):
        self.assertEqual((False, "fast"), transcribe._resolve_default_model_setting("fast"))
        self.assertEqual((False, "cohere"), transcribe._resolve_default_model_setting(" Cohere "))

    def _stub_app(self, is_auto, mode):
        class Stub(transcribe.VoiceTranscribeApp):
            def __init__(self):  # skip rumps.App setup; only the default-model logic runs
                self.default_model_is_auto = is_auto
                self.default_model_mode = mode
                self.settings = {}
                self.saved = 0
                self.rebuilt = 0

            def _save_settings(self):
                self.saved += 1

            def _rebuild_menu(self):
                self.rebuilt += 1

        return Stub()

    def test_picking_a_model_pins_it_and_auto_unpins(self):
        app = self._stub_app(True, transcribe.DEFAULT_MODEL_MODE)
        other = "fast" if transcribe.DEFAULT_MODEL_MODE != "fast" else "cohere"
        with mock.patch.object(transcribe.rumps, "notification"):
            app._set_default_model(types.SimpleNamespace(representedObject=other))
            self.assertFalse(app.default_model_is_auto)
            self.assertEqual(other, app.default_model_mode)
            self.assertEqual(other, app.settings["default_model_mode"])
            self.assertIn("✓", app._default_model_menu_title(other))
            self.assertNotIn("✓", app._auto_model_menu_title())

            app._set_default_model(types.SimpleNamespace(representedObject="auto"))
            self.assertTrue(app.default_model_is_auto)
            self.assertEqual(transcribe.DEFAULT_MODEL_MODE, app.default_model_mode)
            self.assertEqual("auto", app.settings["default_model_mode"])
            self.assertIn("✓", app._auto_model_menu_title())
            self.assertNotIn("✓", app._default_model_menu_title(transcribe.DEFAULT_MODEL_MODE))
        self.assertEqual(2, app.saved)
        self.assertEqual(2, app.rebuilt)


if __name__ == "__main__":
    unittest.main()
