import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import transcribe


def sine(seconds, amplitude=0.05, hz=180):
    samples = int(transcribe.SAMPLE_RATE * seconds)
    t = np.arange(samples, dtype=np.float32) / transcribe.SAMPLE_RATE
    return (amplitude * np.sin(2 * math.pi * hz * t)).astype(np.float32)


class SilenceGateTest(unittest.TestCase):
    def decision_for(self, audio):
        stats = transcribe._analyze_audio_volume(audio)
        return stats, transcribe._no_volume_skip_reason(stats)

    def test_empty_recording_skips_immediately(self):
        stats, reason = self.decision_for(np.zeros(0, dtype=np.float32))
        self.assertEqual(stats["duration"], 0.0)
        self.assertIsNotNone(reason)

    def test_short_key_click_blip_skips_immediately(self):
        audio = np.zeros(int(transcribe.SAMPLE_RATE * 0.42), dtype=np.float32)
        audio[: int(transcribe.SAMPLE_RATE * 0.06)] = sine(0.06, amplitude=0.025)
        stats, reason = self.decision_for(audio)
        self.assertLess(stats["active_seconds"], 0.12)
        self.assertIsNotNone(reason)

    def test_sparse_low_volume_clip_does_not_trigger_slow_fallback(self):
        audio = np.zeros(int(transcribe.SAMPLE_RATE * 4.7), dtype=np.float32)
        start = int(transcribe.SAMPLE_RATE * 2.0)
        burst = sine(0.60, amplitude=0.024)
        audio[start : start + len(burst)] = burst
        stats, reason = self.decision_for(audio)
        self.assertIsNone(reason)
        self.assertTrue(transcribe._is_low_confidence_no_volume(stats))

    def test_sustained_speech_like_audio_is_allowed(self):
        audio = sine(0.55, amplitude=0.06)
        stats, reason = self.decision_for(audio)
        self.assertGreaterEqual(stats["active_seconds"], 0.28)
        self.assertIsNone(reason)
        self.assertFalse(transcribe._is_low_confidence_no_volume(stats))
        self.assertIsNone(transcribe._dead_input_refresh_reason(stats))

    def test_long_flat_recording_requests_mic_refresh(self):
        audio = np.zeros(int(transcribe.SAMPLE_RATE * 2.0), dtype=np.float32)
        stats, reason = self.decision_for(audio)
        self.assertIsNotNone(reason)
        self.assertIn("flat input", transcribe._dead_input_refresh_reason(stats))

    def test_input_volume_env_flag_defaults_on_and_can_disable(self):
        self.assertTrue(transcribe._env_flag_value(None, default=True))
        for raw in ("0", "false", "off", "no"):
            self.assertFalse(transcribe._env_flag_value(raw, default=True))
        self.assertTrue(transcribe._env_flag_value("1", default=False))

    def test_input_volume_osascript_output_parser(self):
        self.assertEqual(
            transcribe._parse_input_volume_osascript_output("42,100\n"),
            (42, 100),
        )
        self.assertEqual(
            transcribe._parse_input_volume_osascript_output("previous=18 current=100"),
            (18, 100),
        )
        self.assertEqual(
            transcribe._parse_input_volume_osascript_output("not a volume"),
            (None, None),
        )

    def test_input_volume_log_decision_only_changes_or_failures(self):
        self.assertFalse(transcribe._input_volume_should_log(100, 100))
        self.assertTrue(transcribe._input_volume_should_log(25, 100))
        self.assertTrue(transcribe._input_volume_should_log(None, None))
        self.assertTrue(transcribe._input_volume_should_log(100, 100, error="boom"))

    def test_force_macos_input_volume_uses_osascript_and_short_timeout(self):
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="35,100\n", stderr="")

        result = transcribe._force_macos_input_volume_to_max(
            timeout_seconds=0.12,
            runner=fake_runner,
        )
        self.assertEqual(result, {"previous": 35, "current": 100, "error": None})
        cmd, kwargs = calls[0]
        self.assertEqual(cmd[0], "/usr/bin/osascript")
        self.assertIn("set volume input volume 100", cmd[-1])
        self.assertEqual(kwargs["timeout"], 0.12)
        self.assertFalse(kwargs["check"])

    def test_force_macos_input_volume_reports_failure_nonfatally(self):
        def fake_runner(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="20,20\n", stderr="denied")

        result = transcribe._force_macos_input_volume_to_max(
            timeout_seconds=0.12,
            runner=fake_runner,
        )
        self.assertEqual(result["previous"], 20)
        self.assertEqual(result["current"], 20)
        self.assertEqual(result["error"], "denied")

    def test_self_relaunch_prefers_repo_run_sh(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            launcher = repo / "run.sh"
            launcher.write_text("#!/bin/bash\n")
            launcher.chmod(0o755)
            script = transcribe._build_self_relaunch_script(
                repo,
                "/tmp/voice-transcribe.log",
                0.8,
            )
            self.assertIn("run.sh", script)
            self.assertNotIn("sys.executable", script)
            self.assertIn("sleep 0.80", script)

    def test_audio_refresh_failure_triggers_clean_relaunch(self):
        self.assertTrue(
            transcribe._should_relaunch_after_audio_refresh_failure(
                "empty-recording",
                "Error opening InputStream: Internal PortAudio error [PaErrorCode -9986]",
            )
        )
        self.assertTrue(
            transcribe._should_relaunch_after_audio_refresh_failure(
                "recording-live flat input",
                "Audio Unit: Invalid Property Value",
            )
        )
        self.assertFalse(
            transcribe._should_relaunch_after_audio_refresh_failure(
                "watchdog",
                "temporary query timeout",
            )
        )

    def test_pmset_battery_parser(self):
        parsed = transcribe._parse_pmset_battery_output(
            "Now drawing from 'Battery Power'\\n -InternalBattery-0 (id=1234567)\\t42%; discharging;"
        )
        self.assertEqual(parsed["source"], "Battery Power")
        self.assertEqual(parsed["percent"], 42)

    def test_audio_device_prefers_system_default_input(self):
        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "External USB Mic", "max_input_channels": 1},
            {"name": "Display Speakers", "max_input_channels": 0},
        ]
        with mock.patch.dict("os.environ", {"VOICE_TRANSCRIBE_AUDIO_DEVICE": ""}):
            self.assertEqual(
                transcribe._choose_input_device_index(devices, default_device=[1, 2]),
                1,
            )

    def test_audio_device_falls_back_to_macbook_input(self):
        devices = [
            {"name": "Display Speakers", "max_input_channels": 0},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "Jump Desktop Microphone", "max_input_channels": 8},
        ]
        with mock.patch.dict("os.environ", {"VOICE_TRANSCRIBE_AUDIO_DEVICE": ""}):
            self.assertEqual(
                transcribe._choose_input_device_index(devices, default_device=[None, 0]),
                1,
            )

    def test_audio_device_can_be_pinned_by_name(self):
        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "Studio Mic", "max_input_channels": 1},
        ]
        with mock.patch.dict("os.environ", {"VOICE_TRANSCRIBE_AUDIO_DEVICE": ""}):
            self.assertEqual(
                transcribe._choose_input_device_index(
                    devices,
                    default_device=[0, 0],
                    preferred_name="studio",
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
