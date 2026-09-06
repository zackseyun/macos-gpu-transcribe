import io
import os
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from swift_asr import MODEL, SwiftASR, _split_wav, _wav_bytes


class SwiftASRTests(unittest.TestCase):
    def test_pause_splitting_preserves_all_samples(self):
        speech = np.full(10 * 16000, 0.2, dtype=np.float32)
        audio = np.concatenate([speech, np.zeros(16000), speech, np.zeros(16000), speech])
        original = _wav_bytes(audio)
        chunks = _split_wav(original)
        self.assertEqual(len(chunks), 3)
        frames = []
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk)) as wav:
                self.assertLessEqual(wav.getnframes(), 14 * 16000)
                frames.append(wav.readframes(wav.getnframes()))
        with wave.open(io.BytesIO(original)) as wav:
            self.assertEqual(b"".join(frames), wav.readframes(wav.getnframes()))

    def test_continuous_long_speech_requests_full_clip_fallback(self):
        self.assertIsNone(_split_wav(_wav_bytes(np.full(30 * 16000, 0.2))))

    def test_short_pauses_above_noise_floor_stay_on_swift(self):
        speech = np.full(8 * 16000, 0.02, dtype=np.float32)
        pause = np.full(int(0.08 * 16000), 0.003, dtype=np.float32)
        chunks = _split_wav(_wav_bytes(np.concatenate([speech, pause, speech, pause, speech])))
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 3)

    def test_sample_rate_preserved(self):
        with wave.open(io.BytesIO(_wav_bytes((np.zeros(8000), 8000)))) as wav:
            self.assertEqual(wav.getframerate(), 8000)

    def test_context_and_accurate_mode_use_existing_backend(self):
        for kwargs in ({"context": "Mishaal"}, {"fast": False}):
            client = SwiftASR()
            client._start = Mock()
            fallback = Mock(return_value={"text": "retained context"})
            self.assertEqual(client.transcribe(np.zeros(16000), fallback, **kwargs)["text"], "retained context")
            fallback.assert_called_once()
            client._start.assert_not_called()

    def test_bad_model_response_disables_swift_and_retries_full_audio(self):
        client = SwiftASR()
        client.enabled = True
        client.binary = Mock()
        client._start = Mock()
        client._json = Mock(return_value={"model": "unexpected", "text": "wrong"})
        client.stop = Mock()
        fallback = Mock(return_value={"text": "correct"})
        with patch("swift_asr.os.access", return_value=True):
            result = client.transcribe(np.zeros(16000), fallback)
            self.assertEqual(result["text"], "correct")
            self.assertFalse(client.enabled)
            client.transcribe(np.zeros(16000), fallback)
        self.assertEqual(client._json.call_count, 1)
        self.assertEqual(fallback.call_count, 2)
        client.stop.assert_called_once()

    def test_python_override_never_starts_native_process(self):
        with patch.dict(os.environ, {"VOICE_TRANSCRIBE_QWEN_BACKEND": "python"}):
            client = SwiftASR()
        client._start = Mock()
        client.transcribe(np.zeros(16000), lambda: {"text": "python"})
        client._start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
