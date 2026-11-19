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

    def test_quiet_dictation_below_old_cutoff_is_not_discarded(self):
        stats, reason = self.decision_for(sine(3.0, amplitude=0.011))
        self.assertLess(stats["max_rms"], 0.008)
        self.assertIsNone(reason)

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


class QuietEdgeTrimTest(unittest.TestCase):
    """_speech_bounds drops long quiet starts/ends so Nemotron doesn't loop."""

    rng = np.random.default_rng(7)

    def room(self, seconds, amplitude=0.004):
        samples = int(transcribe.SAMPLE_RATE * seconds)
        return (amplitude * self.rng.standard_normal(samples)).astype(np.float32)

    def voice(self, seconds, amplitude=0.05):
        # 1 kHz sits in the speech band; the shared sine() helper is 180 Hz.
        return sine(seconds, amplitude=amplitude, hz=1000) + self.room(seconds)

    def bounds(self, *parts):
        audio = np.concatenate(parts)
        start, end = transcribe._speech_bounds(audio)
        return start / transcribe.SAMPLE_RATE, end / transcribe.SAMPLE_RATE, len(audio) / transcribe.SAMPLE_RATE

    def test_long_quiet_ending_is_cut_with_a_margin(self):
        start, end, total = self.bounds(self.voice(1.5), self.room(19.0))
        self.assertEqual(start, 0.0)
        self.assertAlmostEqual(end, 1.5 + transcribe.EDGE_TRIM_PAD_SECONDS, delta=0.26)
        self.assertGreater(total, 20.0)

    def test_long_quiet_start_is_cut_with_a_margin(self):
        start, end, total = self.bounds(self.room(4.0), self.voice(3.0))
        self.assertAlmostEqual(start, 4.0 - transcribe.EDGE_TRIM_PAD_SECONDS, delta=0.26)
        self.assertEqual(end, total)

    def test_continuous_speech_and_short_pauses_are_untouched(self):
        for parts in (
            (self.voice(12.0),),
            (self.voice(3.0), self.room(1.5)),
            (self.room(1.5), self.voice(3.0), self.room(1.0), self.voice(2.0)),
        ):
            start, end, total = self.bounds(*parts)
            self.assertEqual((start, end), (0.0, total))

    def test_low_rumble_does_not_count_as_speech(self):
        rumble = sine(10.0, amplitude=0.05, hz=60) + self.room(10.0)
        start, end, total = self.bounds(self.voice(2.0), rumble)
        self.assertLess(end, 3.5)

    def test_soft_last_words_survive_a_loud_knock(self):
        knock = self.voice(0.25, amplitude=0.8)
        start, end, total = self.bounds(
            self.voice(2.0), knock, self.voice(3.0, amplitude=0.012), self.room(0.5)
        )
        self.assertEqual((start, end), (0.0, total))

    def test_short_or_silent_audio_is_untouched(self):
        for audio in (np.zeros(0, dtype=np.float32), np.zeros(8000, dtype=np.float32), self.room(5.0)):
            self.assertEqual(transcribe._speech_bounds(audio), (0, len(audio)))


if __name__ == "__main__":
    unittest.main()
