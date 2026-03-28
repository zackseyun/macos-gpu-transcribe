"""Transcription worker subprocess — loads model once, transcribes on demand."""
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
}


def run(request_pipe, result_pipe):
    """Receive requests via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Transcription worker started (PID {os.getpid()})", flush=True)

    # Set Metal cache limit upfront — prevents unbounded GPU buffer accumulation
    try:
        import mlx.core as mx
        mx.set_cache_limit(METAL_CACHE_LIMIT_BYTES)
        print(f"Transcription worker: Metal cache limit set to {METAL_CACHE_LIMIT_BYTES / (1024**3):.0f}GB", flush=True)
    except Exception as e:
        print(f"Transcription worker: failed to set cache limit: {e}", flush=True)

    transcribe_fn = None
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

        try:
            if transcribe_fn is None:
                print("Transcription worker: loading transcribe function...", flush=True)
                from mlx_qwen3_asr import transcribe as _transcribe_fn
                transcribe_fn = _transcribe_fn
                print("Transcription worker: ready", flush=True)

            t0 = time.time()
            raw = transcribe_fn(wav_path, model=model_id)
            elapsed = time.time() - t0

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

            transcription_count += 1

            # Use MLX's own memory tracking (current, not peak)
            try:
                active_mem = mx.get_active_memory()
                cache_mem = mx.get_cache_memory()
                print(f"Transcription worker: #{transcription_count} [{model_mode}], active={active_mem/(1024**2):.0f}MB, cache={cache_mem/(1024**2):.0f}MB", flush=True)
            except Exception:
                active_mem = 0
                print(f"Transcription worker: #{transcription_count} [{model_mode}]", flush=True)

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
