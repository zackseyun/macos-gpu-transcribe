import re
import signal
import time
import unittest
from unittest.mock import patch

import gpu_yield

UID = 501
TABLE = [
    (1, 0, "0", "/sbin/launchd"),
    (100, 1, "501", "/Applications/Python -u transcribe.py"),          # the app
    (101, 100, "501", "/Applications/Python mlx_next_asr.py --serve"),  # app child
    (200, 1, "501", "/opt/homebrew/Python -u translated_timing_worker.py --language SPN"),
    (201, 200, "501", "whisper-cli -m model.bin -l es"),
    (202, 200, "501", "caffeinate -i python -u translated_timing_worker.py"),
    (203, 201, "501", "ffmpeg -i tape.mp3"),
    (300, 1, "501", "/bin/zsh -c tail -f worker-SPN.log translated_timing_worker.py"),
    (400, 1, "0", "/usr/local/bin/whisper-cli -m other.bin"),          # another user
    (500, 1, "501", "/usr/local/bin/whisper-cli -m solo.bin"),
]
PATTERNS = [re.compile(r"^\S*[Pp]ython\S* (-u )?\S*translated_timing_worker\.py"),
            re.compile(r"^\S*whisper-cli ")]


class MatchingTreeTest(unittest.TestCase):
    def test_pauses_the_job_and_its_children_only(self):
        found = gpu_yield.matching_tree(PATTERNS, TABLE, own_pid=100, uid=UID)
        self.assertEqual(found, {200, 201, 202, 203, 500})

    def test_never_touches_the_app_tree(self):
        everything = [re.compile(".")]
        found = gpu_yield.matching_tree(everything, TABLE, own_pid=100, uid=UID)
        self.assertNotIn(100, found)
        self.assertNotIn(101, found)


class GpuYieldTest(unittest.TestCase):
    def setUp(self):
        self.signals = []
        patcher = patch.object(gpu_yield.os, "kill", side_effect=lambda pid, sig: self.signals.append((pid, sig)))
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in (("_process_table", lambda: TABLE), ):
            p = patch.object(gpu_yield, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(gpu_yield.os, "getpid", return_value=100)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(gpu_yield.os, "getuid", return_value=UID)
        p.start()
        self.addCleanup(p.stop)
        self.logs = []

    def make(self, **kwargs):
        return gpu_yield.GpuYield([PATTERNS[0].pattern, PATTERNS[1].pattern], log=self.logs.append, **kwargs)

    def test_pause_then_resume_continues_everything_it_stopped(self):
        y = self.make()
        y.pause()
        stopped = {pid for pid, sig in self.signals if sig == signal.SIGSTOP}
        self.assertEqual(stopped, {200, 201, 202, 203, 500})
        y.resume()
        continued = {pid for pid, sig in self.signals if sig == signal.SIGCONT}
        self.assertEqual(continued, stopped)
        self.assertIsNone(y._timer)

    def test_short_dictation_that_ends_first_never_pauses(self):
        y = self.make()
        with y._lock:
            y._wanted = True
        y.resume()            # dictation ended before the pause thread ran
        y._pause_if_wanted()  # the late pause thread
        self.assertEqual(self.signals, [])

    def test_time_limit_resumes_a_forgotten_pause(self):
        y = self.make(max_pause=0.05)
        y.pause()
        time.sleep(0.3)
        self.assertTrue(any(sig == signal.SIGCONT for _, sig in self.signals))
        self.assertTrue(any("time limit" in line for line in self.logs))

    def test_no_patterns_does_nothing(self):
        y = gpu_yield.GpuYield([], log=self.logs.append)
        y.pause()
        y.pause_in_background()
        y.resume_all()
        self.assertEqual(self.signals, [])

    def test_bad_pattern_is_skipped(self):
        y = gpu_yield.GpuYield(["(", PATTERNS[1].pattern], log=self.logs.append)
        self.assertEqual(len(y.patterns), 1)
        self.assertTrue(self.logs)


if __name__ == "__main__":
    unittest.main()
