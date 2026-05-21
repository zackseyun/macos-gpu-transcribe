# Qwen vs Cohere local ASR notes

Date: 2026-05-21

## Practical finding

For the Mac push-to-talk dictation path, the local Qwen3-ASR fast path is the default because it is materially faster than Cohere while remaining good enough for day-to-day dictation.

Current local app setup:

- Fast/default: Qwen3-ASR 0.6B via MLX/Metal
- Local checkpoint: `models/qwen3-asr-0.6b-4bit` (~509 MB, ignored by git)
- Fallback/comparison: Cohere Transcribe 2B via PyTorch/MPS

## Local latency evidence

From `/tmp/voice-transcribe.log` after switching the live app to Qwen fast:

- Qwen fast recent median: ~0.37s total, ~0.21s ASR worker
- Cohere recent median: ~0.78s total, ~0.57s ASR worker
- Direct same-recording warm check:
  - Qwen full model warm: ~0.53s
  - Qwen local 4-bit warm: ~0.21s

The improvement is not only model quality; it is deployment shape:

1. Qwen fast is smaller (0.6B vs Cohere 2B).
2. The Qwen checkpoint is locally quantized to 4-bit.
3. The Qwen runtime uses MLX/Metal on Apple Silicon.
4. The app preloads and keeps Qwen warm, while Cohere preload is off by default so it does not compete for memory/GPU priority.

## Product implication

For low-latency live or near-live transcription, prefer a smaller resident ASR lane first. Keep larger/higher-accuracy ASR as a fallback, audit, or archival-quality lane only when the product actually needs it.

This suggests the backend should be reviewed for the same split:

- Live captions / interaction loop: small resident ASR model, low latency, warm worker.
- Post-session archive / shareable transcript: accuracy-first model or cloud path if needed.
- Clip semantic enrichment: transcript-first can be cheap enough; benchmark actual service logs before assuming a larger model is necessary.

## Deterministic formatting fixes added

The formatter now canonicalizes common Qwen mishearings:

- `QIN` -> `Qwen`
- `Quan` -> `Qwen`
- `Quin` -> `Qwen`
- `Quinn` -> `Qwen`
- `Quinn three ASR` / `Quan three ASR` / `QIN three ASR` -> `Qwen3-ASR`
