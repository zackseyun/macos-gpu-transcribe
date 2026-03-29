"""Transcription worker subprocess — loads model once, transcribes on demand.

Supports two backends:
  - MLX (Qwen3-ASR 0.6B/1.7B) — Metal-native, fastest on Apple Silicon
  - PyTorch MPS (Cohere Transcribe 2B) — best accuracy, runs via Metal MPS backend

The worker keeps models loaded in GPU memory across transcriptions. MLX models share
a cache (6GB limit). Cohere uses PyTorch's separate MPS allocation.

IMPORTANT — Cohere requires float32 on MPS. float16 causes overflow in masked_fill
(-1e9 > float16 max of 65504). This doubles VRAM usage (~4GB) but is necessary.
"""
import os
import signal
import time


# Restart worker after this many transcriptions to reclaim any non-cache memory leaks
MAX_TRANSCRIPTIONS_BEFORE_RESTART = 50
# Metal cache limit — caps GPU buffer cache to 6GB (both models + headroom)
METAL_CACHE_LIMIT_BYTES = 6 * 1024 * 1024 * 1024
# Restart if MLX active memory exceeds this (catches leaks that clear_cache misses)
MAX_ACTIVE_MEMORY_BYTES = 8 * 1024 * 1024 * 1024

MODEL_IDS = {
    "fast": "Qwen/Qwen3-ASR-0.6B",
    "accurate": "Qwen/Qwen3-ASR-1.7B",
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
}

# Which backend each model mode uses
MODEL_BACKENDS = {
    "fast": "mlx",
    "accurate": "mlx",
    "cohere": "pytorch",
}


def _transcribe_mlx(wav_path, model_id, transcribe_fn, mx):
    """Transcribe using MLX backend (Qwen3-ASR)."""
    if transcribe_fn is None:
        print("Transcription worker: loading MLX transcribe function...", flush=True)
        from mlx_qwen3_asr import transcribe as _transcribe_fn
        transcribe_fn = _transcribe_fn
        print("Transcription worker: MLX ready", flush=True)

    raw = transcribe_fn(wav_path, model=model_id)

    # Clear MLX Metal cache after each transcription
    try:
        mx.clear_cache()
    except Exception:
        pass

    if isinstance(raw, dict):
        text = raw.get("text", "").strip()
    elif hasattr(raw, "text"):
        text = raw.text.strip()
    else:
        text = str(raw).strip()

    return text, transcribe_fn


def _transcribe_cohere(wav_path, model_id, cohere_model, cohere_processor):
    """Transcribe using PyTorch MPS backend (Cohere Transcribe).

    Uses float32 because float16 overflows on MPS masked_fill ops.
    Model stays loaded on MPS across calls — only first call pays load cost.
    """
    import torch

    if cohere_model is None:
        print("Transcription worker: loading Cohere model (2B, float32 on MPS)...", flush=True)
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        cohere_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        cohere_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float32
        )
        cohere_model.to("mps")
        cohere_model.eval()
        print("Transcription worker: Cohere ready on MPS", flush=True)

    with torch.no_grad():
        result = cohere_model.transcribe(
            processor=cohere_processor,
            audio_files=[wav_path],
            language="en",
        )

    # result is a list of strings (one per file)
    text = result[0].strip() if result else ""
    return text, cohere_model, cohere_processor


def run(request_pipe, result_pipe):
    """Receive requests via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"Transcription worker started (PID {os.getpid()})", flush=True)

    # Set Metal cache limit upfront — prevents unbounded GPU buffer accumulation
    try:
        import mlx.core as mx
        mx.set_cache_limit(METAL_CACHE_LIMIT_BYTES)
        print(f"Transcription worker: Metal cache limit set to {METAL_CACHE_LIMIT_BYTES / (1024**3):.0f}GB", flush=True)
    except Exception as e:
        mx = None
        print(f"Transcription worker: failed to set cache limit: {e}", flush=True)

    mlx_transcribe_fn = None
    cohere_model = None
    cohere_processor = None
    transcription_count = 0

    while True:
        try:
            request = request_pipe.recv()
        except (EOFError, OSError):
            break

        if request == "__quit__":
            break

        # Support both old format (string path) and new format (dict with model_mode)
        if isinstance(request, str):
            wav_path = request
            model_mode = "fast"
        else:
            wav_path = request["wav_path"]
            model_mode = request.get("model_mode", "fast")

        model_id = MODEL_IDS.get(model_mode, MODEL_IDS["fast"])
        backend = MODEL_BACKENDS.get(model_mode, "mlx")

        try:
            t0 = time.time()

            if backend == "mlx":
                text, mlx_transcribe_fn = _transcribe_mlx(
                    wav_path, model_id, mlx_transcribe_fn, mx
                )
            elif backend == "pytorch":
                text, cohere_model, cohere_processor = _transcribe_cohere(
                    wav_path, model_id, cohere_model, cohere_processor
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

            elapsed = time.time() - t0
            transcription_count += 1

            # Memory tracking (MLX only — PyTorch MPS doesn't expose similar APIs easily)
            active_mem = 0
            if mx is not None:
                try:
                    active_mem = mx.get_active_memory()
                    cache_mem = mx.get_cache_memory()
                    print(f"Transcription worker: #{transcription_count} [{model_mode}/{backend}], active={active_mem/(1024**2):.0f}MB, cache={cache_mem/(1024**2):.0f}MB", flush=True)
                except Exception:
                    print(f"Transcription worker: #{transcription_count} [{model_mode}/{backend}]", flush=True)
            else:
                print(f"Transcription worker: #{transcription_count} [{model_mode}/{backend}]", flush=True)

            result_pipe.send({"text": text, "time": elapsed, "error": None})

            # Restart if active memory is growing out of control or we've done enough transcriptions
            if active_mem > MAX_ACTIVE_MEMORY_BYTES or transcription_count >= MAX_TRANSCRIPTIONS_BEFORE_RESTART:
                reason = f"mem={active_mem/(1024**3):.1f}GB" if active_mem > MAX_ACTIVE_MEMORY_BYTES else f"count={transcription_count}"
                print(f"Transcription worker: requesting restart ({reason})", flush=True)
                result_pipe.send({"__restart__": True})
                break

        except Exception as e:
            result_pipe.send({"text": "", "time": 0, "error": str(e)})

    print("Transcription worker: exiting", flush=True)
