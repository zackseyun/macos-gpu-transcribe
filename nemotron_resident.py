"""Opt-in per-user Nemotron service. No network, microphone, or recording storage."""
import base64
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
LABEL = "com.zack.voice-transcribe.nemotron"
AGENT = Path.home() / "Library/LaunchAgents" / (LABEL + ".plist")
STATE = Path.home() / "Library/Caches" / LABEL
SOCKET = STATE / "asr.sock"
LIMIT = 128 * 1024 * 1024


def enabled():
    return AGENT.is_file()


def request(payload, wait=180):
    deadline = time.monotonic() + wait
    with socket.socket(socket.AF_UNIX) as sock:
        while True:
            try:
                sock.connect(str(SOCKET))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Nemotron resident service is not ready")
                time.sleep(0.1)
        sock.settimeout(max(1, deadline - time.monotonic()))
        sock.sendall(json.dumps(payload).encode() + b"\n")
        with sock.makefile("rb") as stream:
            reply = json.loads(stream.readline(LIMIT))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply


def transcribe(audio):
    from swift_asr import _wav_bytes
    return request({"wav": base64.b64encode(_wav_bytes(audio)).decode()})["text"]


def run():
    from mlx_next_asr import NextASR
    os.umask(0o077)
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    STATE.chmod(0o700)
    SOCKET.unlink(missing_ok=True)
    engine = NextASR()
    try:
        engine.warm("nemotron35")
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(SOCKET))
            listener.listen(16)
            listener.settimeout(2)
            print("Nemotron ready", flush=True)
            while True:
                if engine.process.poll() is not None:
                    raise SystemExit("Model exited; restarting resident service")
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(180)
                    try:
                        with conn.makefile("rb") as stream:
                            line = stream.readline(LIMIT + 1)
                        if not line:
                            continue
                        if len(line) > LIMIT:
                            raise ValueError("Recording too large")
                        payload = json.loads(line)
                        if engine.process.poll() is not None:
                            raise SystemExit("Model exited; restarting resident service")
                        if payload.get("warm") is True:
                            reply = {"ready": True, "pid": engine.process.pid}
                        else:
                            reply = engine._request({"wav": payload["wav"]})
                    except Exception as exc:
                        if engine.process is None or engine.process.poll() is not None:
                            raise SystemExit("Lost model; restarting resident service") from exc
                        reply = {"error": str(exc)}
                    try:
                        conn.sendall(json.dumps(reply).encode() + b"\n")
                    except OSError:
                        pass
    finally:
        SOCKET.unlink(missing_ok=True)
        engine.stop()


if __name__ == "__main__":
    run()
