"""Isolated resident MLX Audio backend. Audio travels only over private pipes.

The GUI's existing Python/permissions/dependencies remain untouched. The child
exits on pipe EOF or parent death, including forced worker recycling.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
MODES = {"granite5": "Granite Speech 5.0 2B (MLX)",
         "nemotron35": "Nemotron 3.5 Transcribe 4-bit (MLX)"}
MODEL_DIRS = {"granite5": "granite5", "nemotron35": "nemotron"}


class NextASR:
    def __init__(self):
        self.process = None
        self.mode = None
        self.lock = threading.RLock()

    def stop(self):
        with self.lock:
            p, self.process = self.process, None
            self.mode = None
            if p is not None:
                if p.stdin:
                    p.stdin.close()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                if p.stdout:
                    p.stdout.close()

    def _request(self, payload, timeout=180):
        p = self.process
        try:
            p.stdin.write(json.dumps(payload) + "\n")
            p.stdin.flush()
            if not select.select([p.stdout], [], [], timeout)[0]:
                raise TimeoutError("MLX transcription timed out")
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("MLX transcription process exited")
            result = json.loads(line)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        except Exception:
            self.stop()
            raise

    def warm(self, mode):
        if mode not in MODES:
            raise ValueError("Unknown MLX model mode")
        with self.lock:
            if self.process is not None and self.process.poll() is None and self.mode == mode:
                return
            self.stop()
            python = ROOT / ".mlx-next/venv14/bin/python"
            if not python.is_file():
                raise RuntimeError("Install the isolated runtime with scripts/install_mlx_next.sh")
            env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            self.process = subprocess.Popen(
                [str(python), str(Path(__file__).resolve()), "--serve", mode],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                text=True, bufsize=1, env=env,
            )
            self.mode = mode
            self._request({"warm": True})

    def transcribe(self, audio, mode):
        from swift_asr import _wav_bytes
        with self.lock:
            self.warm(mode)
            return self._request({"wav": base64.b64encode(_wav_bytes(audio)).decode()})["text"]


def serve(mode):
    import contextlib
    import io
    parent = os.getppid()
    def watchdog():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from mlx_audio.stt import load
        model = None
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if model is None:
                    model = load(str(ROOT / ".mlx-next/models" / MODEL_DIRS[mode]))
                if request.get("warm"):
                    # Compile the same path used for speech, without recording.
                    audio = np.zeros(16000, dtype=np.float32)
                    generate(model, audio, mode, max_tokens=128)
                    response = {"ready": True}
                else:
                    audio, sr = sf.read(io.BytesIO(base64.b64decode(request["wav"])), dtype="float32")
                    if sr != 16000 or audio.ndim != 1 or not len(audio):
                        raise ValueError("Expected nonempty mono 16 kHz audio")
                    response = {"text": generate(model, audio, mode)}
                mx.clear_cache()
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(response) + "\n")
            protocol.flush()


def generate(model, audio, mode, max_tokens=None):
    """Bound long recordings with overlapping chunks; never silently truncate."""
    import mlx.core as mx
    # Granite recommends <=2 minutes. Use overlap to retain boundary words.
    limit, overlap = 110 * 16000, 2 * 16000
    texts = []
    for start in range(0, len(audio), limit - overlap):
        chunk = audio[start:start + limit]
        budget = max_tokens or max(128, min(8192, int(len(chunk) / 16000 * 12 + 64)))
        kwargs = {"max_tokens": budget, "temperature": 0.0}
        if mode == "granite5":
            kwargs.update(punctuate=True)
        result = model.generate(mx.array(chunk), **kwargs)
        if max_tokens is None and (getattr(result, "finish_reason", None) == "length"
                or getattr(result, "generation_tokens", 0) >= budget):
            raise RuntimeError("Transcription reached its token limit; refusing incomplete text")
        text = result.text.strip()
        if texts:
            # Only deduplicate exact words in the overlap, not repeated phrases elsewhere.
            import string
            previous, current = texts[-1].split(), text.split()
            norm = lambda words: [w.strip(string.punctuation).casefold() for w in words]
            for n in range(min(40, len(previous), len(current)), 0, -1):
                if norm(previous[-n:]) == norm(current[:n]):
                    text = " ".join(current[n:])
                    break
        texts.append(text)
        if start + limit >= len(audio):
            break
    return " ".join(t for t in texts if t)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--serve" or sys.argv[2] not in MODES:
        raise SystemExit("Usage: mlx_next_asr.py --serve granite5|nemotron35")
    serve(sys.argv[2])
