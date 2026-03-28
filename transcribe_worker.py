"""Transcription worker subprocess — loads model once, transcribes on demand."""
import os
import signal
import time


# Restart worker after this many transcriptions to reclaim leaked Metal memory
MAX_TRANSCRIPTIONS_BEFORE_RESTART = 20
# Also restart if memory exceeds this (bytes) — 4GB
MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024


def _get_memory_bytes():
    """Get current process RSS in bytes."""
    try:
        import resource
        # ru_maxrss is in bytes on macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0


def run(request_pipe, result_pipe):
    """Receive wav paths via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Transcription worker started (PID {os.getpid()})", flush=True)

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

            # Clear MLX Metal cache after each transcription to prevent memory leak
            try:
                import mlx.core as mx
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
            mem_bytes = _get_memory_bytes()
            mem_gb = mem_bytes / (1024**3)
            print(f"Transcription worker: #{transcription_count}, mem={mem_gb:.1f}GB", flush=True)

            result_pipe.send({"text": text, "time": elapsed, "error": None})

            # Signal parent to restart us if memory is too high or we've done too many
            if transcription_count >= MAX_TRANSCRIPTIONS_BEFORE_RESTART or mem_bytes > MAX_MEMORY_BYTES:
                print(f"Transcription worker: requesting restart (count={transcription_count}, mem={mem_gb:.1f}GB)", flush=True)
                result_pipe.send({"__restart__": True})
                break

        except Exception as e:
            result_pipe.send({"text": "", "time": 0, "error": str(e)})

    print("Transcription worker: exiting", flush=True)
