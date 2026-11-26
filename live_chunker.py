"""Find safe pause boundaries in a dictation that is still being recorded.

Long dictations can be transcribed piece by piece while the speaker keeps
talking, so releasing Fn leaves only the last piece to transcribe. Pieces are
cut only in the middle of a real pause, never mid-word, and only after enough
speech has accumulated that each piece keeps its sentence context.
"""
from __future__ import annotations

import difflib
import string
import threading
import time

import numpy as np

FRAME_SECONDS = 0.03


def frame_rms(audio, sample_rate, frame_seconds=FRAME_SECONDS):
    frame = max(1, int(frame_seconds * sample_rate))
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(audio[: count * frame], dtype=np.float32).reshape(count, frame)
    return np.sqrt(np.mean(frames * frames, axis=1))


def silence_threshold(rms, floor=0.003):
    """Adaptive: above the mic's noise floor, well below speech level.

    The MacBook mic at forced max input volume idles around 0.01 RMS while
    noise-gated sources sit near zero, so a fixed threshold fails one or the other.
    """
    if len(rms) == 0:
        return floor
    p5, p90 = np.percentile(rms, [5, 90])
    # Cap relative to speech so dense talking (few quiet frames) still has pauses.
    return float(min(max(floor, 1.5 * p5, 0.08 * p90), max(floor, 0.35 * p90)))


def find_cut(audio, start, sample_rate=16000, min_chunk_seconds=8.0,
             pause_seconds=0.5, min_active_seconds=1.0, threshold=None):
    """Return a sample index at the middle of the first qualifying pause, or None.

    A qualifying pause lasts at least ``pause_seconds`` and starts at least
    ``min_chunk_seconds`` after ``start``. A pause still in progress at the end
    of the buffer counts once it is long enough; it is cut at the point reached
    so far, so trailing silence goes to the next piece rather than this one.
    The piece must contain ``min_active_seconds`` of sound above the threshold.
    """
    rms = frame_rms(audio, sample_rate)
    if len(rms) == 0:
        return None
    if threshold is None:
        threshold = silence_threshold(rms)
    frame = max(1, int(FRAME_SECONDS * sample_rate))
    first = start // frame + (1 if start % frame else 0)
    earliest = first + int(round(min_chunk_seconds / FRAME_SECONDS))
    need = int(round(pause_seconds / FRAME_SECONDS))
    silent = rms < threshold
    run_start = None
    for index in range(first, len(rms) + 1):
        quiet = index < len(rms) and silent[index]
        if quiet:
            if run_start is None:
                run_start = index
            continue
        if run_start is None:
            continue
        length = index - run_start
        still_quiet = index == len(rms)
        if run_start >= earliest and length >= need:
            active = float(np.sum(~silent[first:run_start])) * FRAME_SECONDS
            if active >= min_active_seconds:
                middle = run_start + (need // 2 if still_quiet else length // 2)
                return int(middle * frame)
        run_start = None
    return None


def join_texts(texts):
    """Join piece transcripts with single spaces, skipping empty pieces."""
    return " ".join(t.strip() for t in texts if t and t.strip())


def _norm(word):
    return word.strip(string.punctuation + "\u201c\u201d\u2018\u2019").casefold()


def context_start(audio, start, sample_rate=16000, seconds=2.0, search_seconds=0.5):
    """Quietest point about ``seconds`` before ``start`` (avoids starting mid-word)."""
    if start <= 0:
        return 0
    lo = max(0, int(start - (seconds + search_seconds) * sample_rate))
    hi = max(0, int(start - (seconds - search_seconds) * sample_rate))
    rms = frame_rms(audio[lo:hi], sample_rate)
    if len(rms) == 0:
        return lo
    frame = max(1, int(FRAME_SECONDS * sample_rate))
    return lo + int(np.argmin(rms)) * frame


def stitch(previous, following, window=20, min_match=2, max_trailing=3, max_skip=6):
    """Join ``following`` (which starts with re-heard context) onto ``previous``.

    Finds the longest run of words (ignoring case and punctuation) shared by the
    end of ``previous`` and the start of ``following``, and switches transcripts
    in the middle of that run. Words near the old boundary then come from the
    transcription that heard audio on both sides of them, so a pause mid-sentence
    does not become ". The". Up to ``max_trailing`` words after the shared run in
    ``previous`` may differ (the boundary word is often heard differently).
    Returns None when no usable overlap is found.
    """
    p, q = previous.split(), following.split()
    pn, qn = [_norm(w) for w in p], [_norm(w) for w in q]
    p0 = max(0, len(p) - window)
    qn_window = qn[:window]
    matcher = difflib.SequenceMatcher(a=pn[p0:], b=qn_window, autojunk=False)
    match = matcher.find_longest_match(0, len(pn) - p0, 0, len(qn_window))
    if match.size < min_match:
        return None
    pi, qi, n = p0 + match.a, match.b, match.size
    if not all(pn[pi:pi + n]) or len(p) - (pi + n) > max_trailing or qi > max_skip:
        return None
    half = n // 2
    return " ".join(p[:pi + half] + q[qi + half:])


def continue_transcript(text, audio, start, end, transcribe, sample_rate=16000,
                        context_seconds=3.0):
    """Transcribe ``audio[start:end]`` and append it to ``text``.

    After the first piece, the request also re-hears ``context_seconds`` before
    ``start`` and is stitched in the overlap. If the overlap cannot be aligned,
    the piece is transcribed again without context and joined with a space.
    Returns ``(new_text, stitched)``.
    """
    if not text:
        return join_texts([transcribe(audio[start:end])]), False
    begin = context_start(audio, start, sample_rate, context_seconds)
    piece = transcribe(audio[begin:end])
    if begin < start:
        joined = stitch(text, piece or "")
        if joined is not None:
            return joined, True
        piece = transcribe(audio[start:end])
    return join_texts([text, piece]), False


class LivePieces:
    """Transcribe finished pieces of one recording in the background.

    ``get_audio()`` returns the recording so far as a 1-D float array.
    ``transcribe(audio)`` returns text or raises. Pieces run one at a time on a
    single thread, so the worker sees at most one request from here.
    ``finish()`` stops looking for pauses, waits for an in-flight piece, and
    returns ``(text, start_sample, error)``: everything before ``start_sample``
    is transcribed in ``text``. Completed pieces remain valid after a later piece fails.
    """

    def __init__(self, get_audio, transcribe, sample_rate=16000, min_chunk_seconds=8.0,
                 pause_seconds=0.5, poll_seconds=0.5, context_seconds=3.0, log=None):
        self._get_audio = get_audio
        self._transcribe = transcribe
        self._sample_rate = sample_rate
        self._min_chunk_seconds = min_chunk_seconds
        self._pause_seconds = pause_seconds
        self._poll_seconds = poll_seconds
        self.context_seconds = context_seconds
        self._log = log or (lambda message: None)
        self._stop = threading.Event()
        self.text = ""
        self.pieces = 0
        self.start_sample = 0
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self._poll_seconds):
            try:
                audio = self._get_audio()
                cut = find_cut(audio, self.start_sample, self._sample_rate,
                               self._min_chunk_seconds, self._pause_seconds)
                if cut is None or cut <= self.start_sample:
                    continue
                began = time.perf_counter()
                text, stitched = continue_transcript(
                    self.text, audio, self.start_sample, cut, self._transcribe,
                    self._sample_rate, self.context_seconds)
                elapsed = time.perf_counter() - began
                self.pieces += 1
                self._log(
                    f"Live piece {self.pieces}: "
                    f"{self.start_sample / self._sample_rate:.1f}-{cut / self._sample_rate:.1f}s "
                    f"→ {elapsed:.2f}s{' (stitched)' if stitched else ''}"
                )
                self.text, self.start_sample = text, cut
            except Exception as exc:
                self.error = exc
                self._log(f"Live piece failed; will retry remaining audio: {exc}")
                return

    def finish(self, timeout=None):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
        if self._thread.is_alive():
            self.error = self.error or TimeoutError("live piece still running")
        return self.text, self.start_sample, self.error
