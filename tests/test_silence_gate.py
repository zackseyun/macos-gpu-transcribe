import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
