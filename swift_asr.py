"""Resident speech-swift Qwen3 MLX backend, with a local Python MLX fallback.

The supervisor's stdin is owned by the transcription worker. EOF (including
SIGKILL of that worker) shuts down the native server, avoiding orphan GPU models.
All requests stay on loopback; audio is never sent to a remote transcription API.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import wave


MODEL = "qwen3-asr-0.6b-mlx-int4"
DEFAULT_BINARY = Path(__file__).resolve().parent / ".swift-runtime/speech-v0.0.27/speech-server"


def _wav_bytes(audio):
    """Encode the app's mono 16 kHz float audio without writing recordings to disk."""
    if isinstance(audio, (str, bytes, os.PathLike)):
        return Path(audio).read_bytes()
    import numpy as np

    sample_rate = 16000
    if isinstance(audio, tuple):
        audio, sample_rate = audio
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes((samples * 32767).clip(-32768, 32767).astype("<i2").tobytes())
    return output.getvalue()


def _split_wav(data):
    """Split long clips only at silence; preserve every sample in order.

    Qwen Swift's >15-second decoding guard can distort repeated words. Keep
    chunks under 14 seconds, or use Python's full-clip path if there is no safe
    pause. Returning None requests that fallback without disabling Swift.
    """
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            return None
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    limit = 14 * 16000
    if len(samples) <= limit:
        return [data]
    frame = 320  # 20 ms
    framed = samples[:len(samples) // frame * frame].astype(np.float32).reshape(-1, frame)
    rms = np.sqrt(np.mean(framed ** 2, axis=1))
    threshold = max(3.0, float(np.percentile(rms, 95)) * 0.10)
    quiet = rms < threshold
    pauses = np.convolve(quiet.astype(int), np.ones(8, dtype=int), mode="valid") == 8
    boundaries = []
    start = 0
    while len(samples) - start > limit:
        candidates = np.flatnonzero(pauses) * frame + 4 * frame
        candidates = candidates[(candidates >= start + 6 * 16000) & (candidates <= start + limit)]
        if not len(candidates):
            return None
        end = int(candidates[-1])
        boundaries.append((start, end))
        start = end
    boundaries.append((start, len(samples)))
    chunks = []
    for start, end in boundaries:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(samples[start:end].tobytes())
        chunks.append(output.getvalue())
    return chunks


class SwiftASR:
    def __init__(self):
        self.binary = Path(os.getenv("VOICE_TRANSCRIBE_QWEN_SWIFT_BIN", str(DEFAULT_BINARY)))
        self.enabled = os.getenv("VOICE_TRANSCRIBE_QWEN_BACKEND", "swift").lower() != "python"
        self.process = None
        self.url = None
        self.lock = threading.Lock()
        # Ignore HTTP_PROXY for private loopback audio requests.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.last_backend = "python-mlx"

    def _json(self, path, payload=None, timeout=2):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with self.http.open(request, timeout=timeout) as response:
            return json.load(response)

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.stop()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--supervise", str(self.binary), str(port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Swift MLX server exited during startup")
            try:
                models = self._json("/v1/models")
                if any(row.get("id") == MODEL for row in models.get("data", [])):
                    print(f"ASR backend: Swift MLX / Metal, {MODEL} (resident)", flush=True)
                    return
                raise ValueError("Swift server does not advertise the required Qwen model")
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        raise TimeoutError("Swift MLX server startup timed out")

    def transcribe(self, audio, fallback, *, fast=True, context="", warm=False):
        # The pinned Swift server does not forward context hints to Qwen.
        # Keep the existing context-aware path when Screen Assist supplies them.
        with self.lock:
            chunks = _split_wav(_wav_bytes(audio)) if self.enabled and fast and not context else None
            if self.enabled and fast and not context and chunks is not None:
                try:
                    if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
                        raise FileNotFoundError("run scripts/install_swift_qwen.sh to install Swift MLX")
                    self._start()
                    texts = []
                    for chunk in chunks:
                        result = self._json("/transcribe", {
                            "model": MODEL,
                            "audio_base64": base64.b64encode(chunk).decode("ascii"),
                        }, timeout=60)
                        if result.get("model") != MODEL or not isinstance(result.get("text"), str):
                            raise ValueError("Invalid Swift ASR response or unexpected model")
                        texts.append(result["text"].strip())
                    self.last_backend = "swift-mlx-int4"
                    if not warm:
                        print(f"ASR backend: {self.last_backend}", flush=True)
                    return {"text": " ".join(text for text in texts if text), "model": MODEL}
                except Exception as exc:
                    # Disable until worker restart, so a broken native runtime
                    # cannot add another timeout to every subsequent recording.
                    self.stop()
                    self.enabled = False
                    print(f"Swift MLX unavailable: {exc}; using Python MLX until worker restart", flush=True)
            self.last_backend = "python-mlx"
            if not warm:
                reason = " (screen context)" if context else ""
                print(f"ASR backend: {self.last_backend}{reason}", flush=True)
            return fallback()

    def stop(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def _supervise(binary, port):
    import signal

    child = subprocess.Popen([binary, "--host", "127.0.0.1", "--port", port])

    def terminate(*_):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def interrupted(*_):
        terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while child.poll() is None:
            if select.select([sys.stdin.buffer], [], [], 0.5)[0]:
                if not os.read(sys.stdin.fileno(), 1):
                    break
    finally:
        terminate()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--supervise":
        _supervise(sys.argv[2], sys.argv[3])
