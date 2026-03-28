"""Transcription worker subprocess — loads model once, transcribes on demand."""
import os
import signal
import time


def run(request_pipe, result_pipe):
    """Receive wav paths via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Transcription worker started (PID {os.getpid()})", flush=True)

    model = None
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

            if isinstance(raw, dict):
                text = raw.get("text", "").strip()
            elif hasattr(raw, "text"):
                text = raw.text.strip()
            else:
                text = str(raw).strip()

            result_pipe.send({"text": text, "time": elapsed, "error": None})
        except Exception as e:
            result_pipe.send({"text": "", "time": 0, "error": str(e)})

    print("Transcription worker: exiting", flush=True)
