# Cohere MLX + Swift migration experiment

## Revert checkpoint

Safe checkpoint before this experiment:

```bash
git checkout checkpoint/pre-cohere-mlx-swift-20260521-113301
```

The latest production-safe state before experimentation is also pushed to GitHub as that tag.

## What was tested

Audio fixture: `last_recording.wav` (~32.25 seconds, 16 kHz mono).

### Cohere MLX 8-bit

Model: `mlx-community/cohere-transcribe-03-2026-mlx-8bit`, subdirectory `mlx-int8/`.
Runtime: `mlx-speech`.

Result: good transcript, very fast.

- 5-run median: ~0.68s
- Effective speed: ~47x real-time
- Transcript quality on local fixture: good

### Cohere MLX 4-bit

Model: `beshkenadze/cohere-transcribe-03-2026-mlx-4bit`.
Runtime required in current Python env: `mlx-audio` with `strict=False`.

Result: fast but not usable in the current Python runtime.

- 3-run median: ~0.92s
- Effective speed: ~35x real-time
- Transcript quality on local fixture: multilingual gibberish

The 4-bit model card says it was validated against newer Swift/Python MLX runtimes, so this may be a runtime/checkpoint-layout mismatch rather than proof the model itself is bad.

## Implementation decision

Do not fully rewrite the macOS app shell in Swift yet. The safest migration path is worker-first:

1. Keep Python UI/audio/Fn capture stable.
2. Move model execution paths to MLX-native runtimes.
3. Add a language-neutral worker protocol later if/when a Swift worker is worth it.
4. Only replace the app shell after model parity, permission behavior, audio capture, and packaging are proven.

## Current app behavior

- `fast`: Qwen3-ASR 0.6B 4-bit via MLX remains default.
- `cohere`: Cohere Transcribe MLX 8-bit via `mlx-speech`.
- `cohere-pytorch`: legacy full 2B PyTorch/MPS Cohere fallback.
- `granite`: Granite/CrispASR path unchanged.

## Files changed

- `transcribe_worker.py`: adds Cohere MLX 8-bit loading/transcription path and keeps legacy PyTorch Cohere as `cohere-pytorch`.
- `transcribe.py`: menu labels/options for MLX Cohere plus legacy PyTorch fallback.
- `scripts/benchmark_cohere_mlx.py`: durable benchmark script for 8-bit and experimental 4-bit.
- `requirements.txt`: adds `mlx-speech` and current Hugging Face hub requirement.
- `tests/*`: updates model/worker expectations.
