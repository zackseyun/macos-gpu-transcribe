import tempfile
import unittest
from pathlib import Path

import numpy as np

import transcribe
import transcribe_worker


class LatencyHelperTest(unittest.TestCase):
    def test_cohere_uses_in_memory_audio_under_threshold(self):
        self.assertTrue(transcribe._should_send_audio_in_memory("cohere", 3.0))
        self.assertTrue(transcribe._should_send_audio_in_memory("fast", 3.0))

    def test_granite_still_requires_wav_file(self):
        self.assertFalse(transcribe._should_send_audio_in_memory("granite", 3.0))

    def test_very_long_audio_uses_file_path_to_avoid_large_pipe_payloads(self):
        self.assertFalse(
            transcribe._should_send_audio_in_memory(
                "cohere",
                transcribe.IN_MEMORY_AUDIO_MAX_SECONDS + 1,
            )
        )

    def test_worker_accepts_raw_16khz_audio_tuple_for_cohere(self):
        audio = np.zeros(160, dtype=np.float32)
        out = transcribe_worker._cohere_audio_from_input((audio, 16000))
        self.assertIs(out, audio)

    def test_worker_rejects_wrong_sample_rate_for_raw_audio(self):
        audio = np.zeros(160, dtype=np.float32)
        with self.assertRaises(ValueError):
            transcribe_worker._cohere_audio_from_input((audio, 8000))

    def test_write_wav_file_creates_reusable_debug_audio(self):
        audio = np.zeros(160, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            transcribe._write_wav_file(audio, path)
            self.assertGreater(path.stat().st_size, 44)


if __name__ == "__main__":
    unittest.main()
