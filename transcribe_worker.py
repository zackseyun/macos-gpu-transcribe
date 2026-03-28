"""Transcription worker subprocess — loads model once, transcribes on demand."""
import os
import signal
import time


# Restart worker after this many transcriptions to reclaim any non-cache memory leaks
MAX_TRANSCRIPTIONS_BEFORE_RESTART = 50
# Metal cache limit — caps GPU buffer cache to 2GB (model weights ~1.2GB + headroom)
METAL_CACHE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
# Restart if MLX active memory exceeds this (catches leaks that clear_cache misses)
MAX_ACTIVE_MEMORY_BYTES = 4 * 1024 * 1024 * 1024


def run(request_pipe, result_pipe):
    """Receive wav paths via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Transcription worker started (PID {os.getpid()})", flush=True)

    # Set Metal cache limit upfront — prevents unbounded GPU buffer accumulation
    try:
        import mlx.core as mx
        mx.metal.set_cache_limit(METAL_CACHE_LIMIT_BYTES)
        print(f"Transcription worker: Metal cache limit set to {METAL_CACHE_LIMIT_BYTES / (1024**3):.0f}GB", flush=True)
    except Exception as e:
        print(f"Transcription worker: failed to set cache limit: {e}", flush=True)

    model = None
    transcription_count = 0

    while True:
        try:
            wav_path = request_pipe.recv()
        except (EOFError, OSError):
            break

        if wav_path == "__quit__":
            break

        try:
            if model is None:
                print("Transcription worker: loading model...", flush=True)
                from mlx_qwen3_asr import transcribe as _transcribe_fn
                model = _transcribe_fn
                print("Transcription worker: model ready", flush=True)

            t0 = time.time()
            raw = model(wav_path)
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
                total_mb = (active_mem + cache_mem) / (1024**2)
                print(f"Transcription worker: #{transcription_count}, active={active_mem/(1024**2):.0f}MB, cache={cache_mem/(1024**2):.0f}MB", flush=True)
            except Exception:
                active_mem = 0
                total_mb = 0
                print(f"Transcription worker: #{transcription_count}", flush=True)

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
