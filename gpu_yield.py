"""Pause background GPU jobs while dictating, so Nemotron gets the GPU to itself.

On 2026-11-27 a sermon batch (a Python worker running Qwen on MPS plus two
Metal whisper-cli children) kept the GPU at 100%. An 8 s clip took ~6 s with
the batch running and 0.4-0.7 s with it paused. Background priority
(taskpolicy -b) still left 1-4 s, and pausing only the whisper-cli children
left 2.5 s, so the whole process tree is paused.

settings.json "pause_while_dictating" lists Python regexes matched against
each process's full command line (`ps -o command`). Anchor them to the
executable (for example "^\\S*whisper-cli ") so shells, editors or `tail`
commands that merely mention the job are never matched. Each matching
process and all of its descendants get SIGSTOP when Fn goes down and SIGCONT
when the dictation ends, after MAX_PAUSE_SECONDS at the latest, on app exit,
and on app start (so a crash mid-pause never leaves a job frozen).
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading

MAX_PAUSE_SECONDS = 180


def _process_table():
    out = subprocess.run(
        ["ps", "-Ao", "pid=,ppid=,uid=,command="],
        capture_output=True, text=True, timeout=3,
    ).stdout
    table = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            table.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return table


def matching_tree(patterns, table, own_pid, uid):
    """PIDs of this user's processes matching a pattern, plus their descendants,
    never including own_pid or anything under it."""
    children = {}
    for pid, ppid, _, _ in table:
        children.setdefault(ppid, []).append(pid)
    mine = {own_pid}
    stack = [own_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in mine:
                mine.add(child)
                stack.append(child)
    roots = [pid for pid, _, owner, command in table
             if owner == str(uid) and pid not in mine
             and any(p.search(command) for p in patterns)]
    found = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in found or pid in mine:
            continue
        found.add(pid)
        stack.extend(children.get(pid, []))
    return found


class GpuYield:
    def __init__(self, patterns, log=print, max_pause=MAX_PAUSE_SECONDS):
        self.patterns = []
        for pattern in patterns or []:
            try:
                self.patterns.append(re.compile(pattern))
            except (re.error, TypeError) as exc:
                log(f"pause_while_dictating: ignoring {pattern!r}: {exc}")
        self._log = log
        self._max_pause = max_pause
        self._lock = threading.Lock()
        self._paused = set()
        self._timer = None
        self._wanted = False

    def _matching(self):
        try:
            return matching_tree(self.patterns, _process_table(), os.getpid(), os.getuid())
        except (OSError, subprocess.SubprocessError) as exc:
            self._log(f"pause_while_dictating: process scan failed: {exc}")
            return set()

    @staticmethod
    def _signal(pids, sig):
        done = set()
        for pid in pids:
            try:
                os.kill(pid, sig)
                done.add(pid)
            except OSError:
                pass
        return done

    def pause(self):
        """Pause matching jobs now (blocking)."""
        with self._lock:
            self._wanted = True
        self._pause_if_wanted()

    def pause_in_background(self):
        """Fn went down: pause without delaying the recording start."""
        if not self.patterns:
            return
        with self._lock:
            self._wanted = True
        threading.Thread(target=self._pause_if_wanted, daemon=True).start()

    def _pause_if_wanted(self):
        if not self.patterns:
            return
        with self._lock:
            if not self._wanted:  # the dictation already ended
                return
            # Two passes: a parent caught mid-fork may start a child after the first scan.
            paused = self._signal(self._matching() - self._paused, signal.SIGSTOP)
            paused |= self._signal(self._matching() - self._paused - paused, signal.SIGSTOP)
            self._paused |= paused
            if self._paused and self._timer is None:
                self._timer = threading.Timer(self._max_pause, self.resume, kwargs={"reason": "time limit"})
                self._timer.daemon = True
                self._timer.start()
        if paused:
            self._log(f"Paused {len(paused)} background GPU process(es) while dictating")

    def resume(self, reason="dictation done"):
        with self._lock:
            self._wanted = False
            pids, self._paused = self._paused, set()
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        resumed = self._signal(pids, signal.SIGCONT)
        if resumed and reason != "dictation done":
            self._log(f"Resumed {len(resumed)} background GPU process(es) ({reason})")

    def resume_all(self):
        """Continue every matching process, in case an earlier run died mid-pause."""
        if self.patterns:
            self._signal(self._matching(), signal.SIGCONT)
