import threading
import unittest

import numpy as np

import transcribe

SR = transcribe.SAMPLE_RATE


class Stub(transcribe.VoiceTranscribeApp):
    def __init__(self, replies):  # skip rumps.App setup; only the helpers are exercised
        self.replies = list(replies)
        self.requests = []
        self._worker_request_lock = threading.RLock()
        self._worker_send_lock = threading.Lock()
        self.live_pieces_enabled = True
        self.audio_buffer = []

    def _transcribe_via_worker(self, wav_path=None, model_mode="fast", screen_context="", audio=None):
        self.requests.append((model_mode, None if audio is None else len(audio[0])))
        return self.replies.pop(0)


class FakeLive:
    pieces = 1
    context_seconds = 2.0

    def __init__(self, text, start, error=None):
        self.result = (text, start, error)

    def finish(self, timeout=None):
        return self.result


def loud(seconds):
    rng = np.random.default_rng(0)
    return (0.1 * np.sin(np.arange(int(seconds * SR)) / SR * 2 * np.pi * 200)
            + rng.normal(0, 0.01, int(seconds * SR))).astype(np.float32)


class LivePiecesAppTest(unittest.TestCase):
    def test_joins_pieces_with_tail(self):
        # The tail request re-hears ~2 s of context, then is stitched in the overlap.
        app = Stub([{"text": "Of the first piece and the tail.", "time": 0.7}])
        audio = loud(14)
        latency = {}
        result = app._finish_live_pieces(
            FakeLive("This is the end of the first piece.", 10 * SR), audio, "nemotron35", latency)
        self.assertEqual(result["text"], "This is the end of the first piece and the tail.")
        self.assertEqual(len(app.requests), 1)
        self.assertGreater(app.requests[0][1], 4 * SR)
        self.assertEqual((latency["live_pieces"], latency["live_tail"]), (1, 4.0))

    def test_silent_tail_is_not_sent(self):
        app = Stub([])
        audio = np.concatenate([loud(10), np.zeros(SR, np.float32)])
        result = app._finish_live_pieces(FakeLive("Only piece.", 10 * SR), audio, "nemotron35", {})
        self.assertEqual(result["text"], "Only piece.")
        self.assertEqual(app.requests, [])

    def test_piece_error_falls_back_to_whole_recording(self):
        app = Stub([])
        result = app._finish_live_pieces(FakeLive("x", SR, RuntimeError("boom")), loud(3), "nemotron35", {})
        self.assertIsNone(result)

    def test_no_pieces_falls_back(self):
        self.assertIsNone(Stub([])._finish_live_pieces(FakeLive("", 0), loud(3), "nemotron35", {}))

    def test_tail_hallucination_dropped(self):
        app = Stub([{"text": "Thank you.", "time": 0.2}, {"text": "Thank you.", "time": 0.2}])
        result = app._finish_live_pieces(FakeLive("Real words.", 10 * SR), loud(12), "nemotron35", {})
        self.assertEqual(result["text"], "Real words.")

    def test_tail_error_is_reported_not_swallowed(self):
        app = Stub([{"error": "MLX died"}])
        result = app._finish_live_pieces(FakeLive("Real words.", 10 * SR), loud(12), "nemotron35", {})
        self.assertEqual(result, {"error": "MLX died"})

    def test_piece_errors_raise_for_fallback(self):
        app = Stub([{"error": "MLX died"}])
        with self.assertRaises(RuntimeError):
            app._transcribe_live_piece(loud(9), "nemotron35")

    def test_only_enabled_for_live_piece_models(self):
        app = Stub([])
        self.assertIsNone(app._start_live_pieces("fast"))
        app.live_pieces_enabled = False
        self.assertIsNone(app._start_live_pieces("nemotron35"))
        app.live_pieces_enabled = True
        live = app._start_live_pieces("nemotron35")
        try:
            self.assertIsNotNone(live)
        finally:
            live.finish()


if __name__ == "__main__":
    unittest.main()
