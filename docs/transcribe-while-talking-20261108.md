# Transcribe While Talking - November 8, 2026

Prototype for Nemotron 3.5 dictation: transcribe finished pieces of a long
dictation in the background while Zack keeps talking, so releasing Fn waits
only for the last piece. Menu toggle **Transcribe While Talking (Nemotron)**,
setting `live_pieces_enabled`, off by default in code.

## How it works

- `live_chunker.find_cut` looks every 0.5 s for a pause of at least 0.5 s that
  starts at least 8 s after the previous cut. The silence threshold adapts per
  recording: the MacBook mic at forced max input idles near 0.01 RMS, while
  noise-gated sources sit near zero.
- Each piece after the first re-hears the 3 s before its start, beginning at the
  quietest frame. `stitch` finds the longest shared run of words near the
  boundary and switches transcripts in its middle. If no overlap aligns, the
  piece is transcribed again without context and joined with a space.
- Pieces run one at a time on one thread. At release the app waits for an
  in-flight piece, then transcribes the tail the same way. A tail with no usable
  volume is skipped. A failed piece or a tail timeout falls back to the ordinary
  whole-recording request. Worker pipe writes are now serialized, because warm
  signals and requests come from different threads.

## Accuracy

Test audio: Zack's 33-second dictation; the 93-second YC founder Zoom take
(`~/Movies/Cartha YC Founder Video/raw/zoom-last-take.mp4`); the 136-second
speedrun extra take; and three cuts of those. That is 1,077 words in total. No
reference transcripts exist, so pieces were compared with whole-clip Nemotron
output.

| Variant | Word differences vs whole | Release wait, replay sum |
|---|---:|---:|
| Plain join (no context) | 23 | 80 s → 23 s |
| 2 s context, exact-suffix stitch | 26 | 62 s → 20 s |
| 3 s context, longest-run stitch (shipped) | 23 (2.1%) | 67 s → 24 s |

Without context, every mid-sentence pause produced artifacts such as "ability
to operate. The the moving parts" and "instincts to Conduct". Stitching
removed these. Remaining word differences go both ways:

- Better in pieces: "Carthage" → "Cartha", "my start" → "my startup".
- Worse in pieces: "Alexa AI" → "Lex AI", "than" → "then" at a boundary.
- Neutral: "um"/"uh", "one thousand five hundred" → "fifteen hundred".

The remaining punctuation differences are mostly sentence breaks the whole-clip
output missed on 90+ second audio ("people and my start right" → "people.
And my startup right").

## Latency

The replay sums assume the measured transcription times and real-time speech.
Real-time runs fed audio at speaking speed through `NextASR` (Low Power Mode on,
heavy background load):

| Recording | Whole | Live pieces |
|---|---:|---:|
| 33 s dictation (run 1, 2 s context) | 4.5 s | 3.1 s (waited 2.4 s for a piece cut in the final pause) |
| 93 s Zoom take (run 1) | 27.7 s | 10.6 s |
| 33 s dictation (run 3, 3 s context, heavier load) | 9.0 s | 10.0 s (pieces ran slower than speech) |

Worst case: the in-flight remainder plus the tail is still a subset of the whole
recording. The penalty is therefore roughly the 3 s re-heard context plus
per-request overhead, about 1 s under Low Power Mode. Dictations under about 8.5 s
never cut, so they are unchanged.

## Ideas not done

- Smaller minimum pieces (6 s / 0.4 s) cut the replay wait further (80 s → 14 s)
  but dropped two words ("our", "I") at boundaries in one clip.
- Cancelling an in-flight piece at release would need cancellation in the MLX
  child process.
