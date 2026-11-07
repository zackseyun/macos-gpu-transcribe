# Local MLX dictation comparison - November 7, 2026

Decision: pin this M4 Max / 128 GB Mac's Fn dictation to **Nemotron 3.5
Transcribe, MLX 4-bit** (`nemotron35`). Hardware Auto defaults on other machines
are unchanged. The previous Qwen 0.6B engine remains selectable.

## Final comparison

| Metric | Nemotron 3.5 4-bit | Granite Speech 5.0 2B |
|---|---:|---:|
| Clean speech word error | 2.00% | 5.00% |
| Added-noise speech word error | 3.50% | 3.50% |
| Combined short clips | 11 / 400 words (2.75%) | 17 / 400 words (4.25%) |
| Short-clip median warm inference | 1.29 s | 0.94 s |
| Nine additional warm repetitions, median | 1.27 s | 0.88 s |
| 163.29-second recording, inference | 24.00 s | 20.55 s |
| Long recording word error | 3.00% | 3.00% |
| Model loading, excluding warmup | 6.54 s | 2.32 s |
| Benchmark peak MLX memory | 19.67 GiB | 14.25 GiB |

Nemotron's lower short-form error rate justifies its modest latency/memory cost
on this machine. The actual private-pipe integration smoke test transcribed a
5.855-second clip in 1.14 seconds after startup/warmup (4.36 seconds). Startup
loading and download times are excluded from warm inference measurements.
Timings varied substantially between passes with system state, so these are not
latency guarantees. The Fn path includes additional recording/paste overhead.

## Method and limitations

- `hf-internal-testing/librispeech_asr_dummy`, clean validation rows
  0, 2, 4, 7, 10, 14, 20, 25: eight public, single-speaker reading clips.
- Same eight clips with deterministic white noise at 15 dB SNR, seed 42.
- One 163.29-second concatenation (the clean set repeated twice with 0.4-second
  pauses) tests the long-recording chunk boundary.
- Identical local adapter, greedy decoding, and no vocabulary hints; Granite
  requests punctuation but does not request translation. Nemotron uses its
  standard ASR task without reasoning or speaker-attribution prompts.
- Case/punctuation normalized; `Mr.` and `mister`, and `M A` and `MA`, are
  equivalent. This is a small correlated test, not a general accuracy ranking
  or a measurement of Zack's spontaneous speech or Cartha-specific vocabulary.
- The first long test exposed case/punctuation-sensitive overlap duplication.
  The adapter now compares overlap words case-insensitively without edge
  punctuation. Both models were rerun with the same fix.

## Reproducibility and verification

Model revisions are pinned in `scripts/download_mlx_next.py`; MLX Audio source
is pinned in `scripts/install_mlx_next.sh`, with tested package versions in
`requirements-mlx-next.txt`. Both model downloads passed their published LFS
SHA-256 checks. The runtime is MLX 0.32.4 on macOS 26.5, Apple M4 Max.

Run `scripts/benchmark_mlx_next.py` using the isolated runtime as documented in
the README. The corpus input can also be a JSON manifest containing `index`,
`file` (16 kHz mono WAV path), and `text` per clean fixture.

Local receipts are under
`/Users/zackseyun/Documents/future 2/output/transcription-upgrade-20261107/`:
`comparison.json`, both `*-results.json` files, `adapter-smoke.json`,
`download-ranged.log`, `tests-final.log`, and `restart.log`.

Verification: 91 unit tests executed (90 passed, one skipped). Private-pipe
transcription and subprocess cleanup passed. After restart, launchd reported
running, the Fn event listener reported active, the app read back the pinned
Nemotron selection, and the transcription worker reported the model ready.
A physical Fn-to-paste action was not synthesized into the user's active app.

Rollback: select **Qwen3-ASR 0.6B (MLX)** in the menu. The exact previous local
settings were backed up as `settings-before-upgrade.json` in the receipts folder.
