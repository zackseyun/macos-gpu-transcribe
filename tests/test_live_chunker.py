import threading
import time
import unittest

import numpy as np

import live_chunker as lc

SR = 16000


def speech(seconds, rng, level=0.08):
    t = np.arange(int(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
            + rng.normal(0, 0.01, len(t))).astype(np.float32)


def pause(seconds, rng, floor=0.01):
    # The MacBook mic at forced max input volume idles near 0.01 RMS.
    return rng.normal(0, floor, int(seconds * SR)).astype(np.float32)


class FindCutTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_cuts_in_middle_of_first_pause_after_minimum(self):
        audio = np.concatenate([speech(3, self.rng), pause(0.8, self.rng),   # too early
                                speech(6, self.rng), pause(0.8, self.rng),   # starts at 9.8 s
                                speech(4, self.rng)])
        cut = lc.find_cut(audio, 0, SR, min_chunk_seconds=8, pause_seconds=0.5)
        self.assertIsNotNone(cut)
        self.assertAlmostEqual(cut / SR, 9.8 + 0.4, delta=0.1)

    def test_short_pauses_are_not_cut(self):
        audio = np.concatenate([speech(9, self.rng), pause(0.3, self.rng), speech(4, self.rng)])
        self.assertIsNone(lc.find_cut(audio, 0, SR, min_chunk_seconds=8, pause_seconds=0.5))

    def test_respects_start_of_uncommitted_audio(self):
        audio = np.concatenate([speech(9, self.rng), pause(0.8, self.rng),
                                speech(5, self.rng), pause(0.8, self.rng), speech(3, self.rng)])
        first = lc.find_cut(audio, 0, SR, 8, 0.5)
        self.assertIsNone(lc.find_cut(audio, first, SR, 8, 0.5))  # next pause only 5.4 s later
        second = lc.find_cut(audio, first, SR, 5, 0.5)
        self.assertAlmostEqual(second / SR, 9.8 + 5 + 0.4, delta=0.1)

    def test_pause_still_in_progress_cuts_near_its_start(self):
        audio = np.concatenate([speech(9, self.rng), pause(2.0, self.rng)])
        cut = lc.find_cut(audio, 0, SR, 8, 0.5)
        self.assertAlmostEqual(cut / SR, 9 + 0.25, delta=0.1)

    def test_silence_only_is_never_a_piece(self):
        audio = pause(20, self.rng)
        self.assertIsNone(lc.find_cut(audio, 0, SR, 8, 0.5))

    def test_join_skips_empty_pieces(self):
        self.assertEqual(lc.join_texts([' One. ', '', 'Two.']), 'One. Two.')


class StitchTest(unittest.TestCase):
    def test_mid_sentence_pause_keeps_sentence_together(self):
        # Piece 1 ended at a pause and guessed a period; piece 2 re-heard the context.
        self.assertEqual(
            lc.stitch('I value the ability to operate.', 'Ability to operate the moving parts.'),
            'I value the ability to operate the moving parts.')

    def test_skips_a_garbled_first_word(self):
        self.assertEqual(
            lc.stitch('what the mainstream.', 'Ought what the mainstream consumer wants.'),
            'what the mainstream consumer wants.')

    def test_boundary_word_heard_differently(self):
        # Re-heard context turned "than" into "then"; still align on "per month".
        self.assertEqual(
            lc.stitch('more expensive per month than', 'Per month, then whatever we intended.'),
            'more expensive per month, then whatever we intended.')

    def test_match_far_from_boundary_is_rejected(self):
        self.assertIsNone(lc.stitch('alpha beta gamma delta epsilon zeta eta theta',
                                    'Alpha beta something else entirely.'))

    def test_no_overlap_returns_none(self):
        self.assertIsNone(lc.stitch('Completely different.', 'Nothing in common here.'))

    def test_continue_without_alignment_retranscribes_without_context(self):
        rng = np.random.default_rng(3)
        audio = np.concatenate([speech(9, rng), pause(0.8, rng), speech(4, rng)])
        calls = []
        def transcribe(a):
            calls.append(len(a))
            return 'unrelated words' if len(calls) == 1 else 'Second part.'
        text, stitched = lc.continue_transcript('First part.', audio, 9 * SR, len(audio), transcribe, SR)
        self.assertEqual((text, stitched), ('First part. Second part.', False))
        self.assertGreater(calls[0], calls[1])  # first request included context

    def test_context_starts_at_quiet_point(self):
        rng = np.random.default_rng(4)
        audio = np.concatenate([speech(6, rng), pause(0.4, rng), speech(3, rng)])
        begin = lc.context_start(audio, 8 * SR, SR, seconds=2.0, search_seconds=0.5)
        self.assertTrue(6.0 <= begin / SR <= 6.4)


class LivePiecesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.audio = np.concatenate([speech(9, rng), pause(0.8, rng), speech(3, rng)])

    def test_transcribes_piece_before_release(self):
        calls = []
        live = lc.LivePieces(lambda: self.audio, lambda a: calls.append(len(a)) or 'First piece.',
                             SR, 8, 0.5, poll_seconds=0.01).start()
        deadline = time.time() + 2
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        text, start, error = live.finish()
        self.assertIsNone(error)
        self.assertEqual(text, 'First piece.')
        self.assertAlmostEqual(start / SR, 9.4, delta=0.1)
        self.assertEqual(calls, [start])

    def test_error_requests_whole_recording_fallback(self):
        def boom(_):
            raise RuntimeError('worker died')
        live = lc.LivePieces(lambda: self.audio, boom, SR, 8, 0.5, poll_seconds=0.01).start()
        time.sleep(0.1)
        text, start, error = live.finish()
        self.assertIsInstance(error, RuntimeError)

    def test_finish_waits_for_in_flight_piece(self):
        started, release = threading.Event(), threading.Event()
        def slow(_):
            started.set()
            release.wait(2)
            return 'Done.'
        live = lc.LivePieces(lambda: self.audio, slow, SR, 8, 0.5, poll_seconds=0.01).start()
        self.assertTrue(started.wait(2))
        threading.Timer(0.1, release.set).start()
        text, start, error = live.finish()
        self.assertEqual((text, error), ('Done.', None))

    def test_no_pause_means_no_pieces(self):
        rng = np.random.default_rng(2)
        audio = speech(12, rng)
        live = lc.LivePieces(lambda: audio, lambda a: 'x', SR, 8, 0.5, poll_seconds=0.01).start()
        time.sleep(0.05)
        self.assertEqual(live.finish(), ('', 0, None))


if __name__ == '__main__':
    unittest.main()
