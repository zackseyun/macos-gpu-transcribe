#!/usr/bin/env python3
"""Benchmark Cohere Transcribe MLX quantized variants against local audio.

This is intentionally standalone so we can test quantized Cohere without
changing the live Fn dictation path.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = REPO_DIR / "last_recording.wav"
DEFAULT_OUT = REPO_DIR / "benchmark_results" / "cohere_mlx_benchmark.json"

VARIANTS = {
    # AppAutomaton/mlx-community 8-bit is the format currently compatible with
    # mlx-speech and produced accurate transcripts in local tests.
    "cohere-mlx-8bit": {
        "repo_id": "mlx-community/cohere-transcribe-03-2026-mlx-8bit",
        "subdir": "mlx-int8",
        "runtime": "mlx_speech",
    },
    # beshkenadze 4-bit is smaller, but it uses the newer mlx-audio/Swift-style
    # checkpoint layout. The Swift runtime is the reliable path in local tests.
    "cohere-mlx-4bit-swift": {
        "repo_id": "beshkenadze/cohere-transcribe-03-2026-mlx-4bit",
        "subdir": ".",
        "runtime": "mlx_audio_swift",
    },
    "cohere-mlx-4bit-swift-server": {
        "repo_id": "beshkenadze/cohere-transcribe-03-2026-mlx-4bit",
        "subdir": ".",
        "runtime": "mlx_audio_swift_server",
    },
    # Kept only to prove the earlier failure mode: current Python mlx-audio can
    # load this checkpoint with strict=False, but that path produced gibberish.
    "cohere-mlx-4bit-python-experimental": {
        "repo_id": "beshkenadze/cohere-transcribe-03-2026-mlx-4bit",
        "subdir": ".",
        "runtime": "mlx_audio_python",
    },
}


def load_audio_16k(path: Path) -> tuple[np.ndarray, int, float]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        old_len = len(audio)
        new_len = int(round(old_len * 16000 / sr))
        audio = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            audio,
        ).astype(np.float32)
        sr = 16000
    return audio, sr, len(audio) / sr if sr else 0.0


def resolve_model_dir(variant: str) -> Path:
    meta = VARIANTS[variant]
    print(f"Downloading/resolving {variant}: {meta['repo_id']} ...", flush=True)
    path = snapshot_download(
        repo_id=meta["repo_id"],
        allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "*.md", "mlx-int8/*"],
    )
    return Path(path) / meta["subdir"]


def find_swift_stt_bin(configured: str | None = None) -> str | None:
    candidates = [
        configured,
        os.getenv("VOICE_TRANSCRIBE_SWIFT_STT_BIN"),
        str(REPO_DIR / ".swift-runtime" / "Release" / "mlx-audio-swift-stt"),
        shutil.which("mlx-audio-swift-stt"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def find_swift_stt_server_bin(configured: str | None = None) -> str | None:
    candidates = [
        configured,
        os.getenv("VOICE_TRANSCRIBE_SWIFT_STT_SERVER_BIN"),
        str(REPO_DIR / ".swift-runtime" / "Release" / "mlx-audio-swift-stt-server"),
        shutil.which("mlx-audio-swift-stt-server"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


class SwiftSttServer:
    def __init__(self, bin_path: str, repo_id: str):
        self.proc = subprocess.Popen(
            [bin_path, "--model", repo_id, "--language", "English"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        deadline = time.perf_counter() + 60
        while time.perf_counter() < deadline:
            payload = self._read_json(deadline)
            if payload.get("event") == "ready":
                return
        raise TimeoutError("Swift STT server did not become ready")

    def _reader(self):
        if not self.proc.stdout:
            return
        for line in self.proc.stdout:
            self.lines.put(line)

    def _read_json(self, deadline: float, expected_id: str | None = None) -> dict:
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Swift STT server exited {self.proc.returncode}")
            try:
                line = self.lines.get(timeout=min(0.5, max(0.05, deadline - time.perf_counter())))
            except queue.Empty:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"swift-server: {line[:160]}", flush=True)
                continue
            if expected_id is not None and payload.get("id") not in {expected_id, None}:
                continue
            return payload
        raise TimeoutError("Timed out waiting for Swift STT server response")

    def transcribe(self, audio_path: Path) -> dict:
        request_id = str(time.time_ns())
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"id": request_id, "audio": str(audio_path), "language": "English"}) + "\n")
        self.proc.stdin.flush()
        payload = self._read_json(time.perf_counter() + 300, expected_id=request_id)
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "Swift STT server failed")
        return payload

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_swift_server_variant(
    variant: str,
    audio_path: Path,
    runs: int,
    swift_stt_bin: str | None,
) -> dict:
    meta = VARIANTS[variant]
    server_bin = find_swift_stt_server_bin(swift_stt_bin)
    if not server_bin:
        raise RuntimeError(
            "mlx-audio-swift-stt-server not found. Run scripts/install_mlx_audio_swift.sh "
            "or pass --swift-stt-bin."
        )
    load_start = time.perf_counter()
    server = SwiftSttServer(server_bin, meta["repo_id"])
    load_seconds = time.perf_counter() - load_start
    timings: list[float] = []
    model_times: list[float] = []
    texts: list[str] = []
    peak_memory_gb: list[float] = []
    try:
        for idx in range(runs):
            start = time.perf_counter()
            payload = server.transcribe(audio_path)
            elapsed = time.perf_counter() - start
            text = str(payload.get("text", "")).strip()
            timings.append(elapsed)
            model_times.append(float(payload.get("totalTime", 0) or 0))
            texts.append(text)
            if payload.get("peakMemoryUsage") is not None:
                peak_memory_gb.append(float(payload["peakMemoryUsage"]))
            print(f"{variant} run {idx + 1}/{runs}: {elapsed:.3f}s resident :: {text[:160]}", flush=True)
    finally:
        server.close()

    return {
        "variant": variant,
        "repo_id": meta["repo_id"],
        "model_dir": "mlx-audio-swift cache",
        "runtime": meta["runtime"],
        "swift_stt_server_bin": server_bin,
        "load_seconds": load_seconds,
        "runs": timings,
        "model_total_times": model_times,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "median_model_total_seconds": statistics.median(model_times) if model_times else None,
        "peak_memory_gb": max(peak_memory_gb) if peak_memory_gb else None,
        "text": texts[-1] if texts else "",
    }


def run_swift_variant(
    variant: str,
    audio_path: Path,
    runs: int,
    swift_stt_bin: str | None,
) -> dict:
    if VARIANTS[variant]["runtime"] == "mlx_audio_swift_server":
        return run_swift_server_variant(variant, audio_path, runs, swift_stt_bin)

    meta = VARIANTS[variant]
    swift_bin = find_swift_stt_bin(swift_stt_bin)
    if not swift_bin:
        raise RuntimeError(
            "mlx-audio-swift-stt not found. Run scripts/install_mlx_audio_swift.sh "
            "or pass --swift-stt-bin."
        )

    timings: list[float] = []
    model_times: list[float] = []
    texts: list[str] = []
    peak_memory_gb: list[float] = []
    for idx in range(runs):
        output_stem = Path(tempfile.mktemp(prefix="cohere4-swift-", dir="/tmp"))
        cmd = [
            swift_bin,
            "--model",
            meta["repo_id"],
            "--audio",
            str(audio_path),
            "--output-path",
            str(output_stem),
            "--format",
            "json",
            "--language",
            "English",
        ]
        start = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - start
        output_json = output_stem.with_suffix(".json")
        payload = json.loads(output_json.read_text())
        output_json.unlink(missing_ok=True)
        text = str(payload.get("text", "")).strip()
        timings.append(elapsed)
        model_times.append(float(payload.get("total_time", 0) or 0))
        texts.append(text)
        if payload.get("peak_memory_usage") is not None:
            peak_memory_gb.append(float(payload["peak_memory_usage"]))
        print(f"{variant} run {idx + 1}/{runs}: {elapsed:.3f}s wall :: {text[:160]}", flush=True)

    return {
        "variant": variant,
        "repo_id": meta["repo_id"],
        "model_dir": "mlx-audio-swift cache",
        "runtime": meta["runtime"],
        "swift_stt_bin": swift_bin,
        "runs": timings,
        "model_total_times": model_times,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "median_model_total_seconds": statistics.median(model_times) if model_times else None,
        "peak_memory_gb": max(peak_memory_gb) if peak_memory_gb else None,
        "text": texts[-1] if texts else "",
    }


def run_variant(
    variant: str,
    audio_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    runs: int,
    swift_stt_bin: str | None = None,
    max_new_tokens: int | None = None,
    punctuation: bool = True,
) -> dict:
    if VARIANTS[variant]["runtime"] == "mlx_audio_swift":
        return run_swift_variant(variant, audio_path, runs, swift_stt_bin)
    if VARIANTS[variant]["runtime"] == "mlx_audio_swift_server":
        return run_swift_server_variant(variant, audio_path, runs, swift_stt_bin)

    import mlx.core as mx

    model_dir = resolve_model_dir(variant)
    runtime = VARIANTS[variant]["runtime"]
    load_start = time.perf_counter()
    if runtime == "mlx_speech":
        from mlx_speech.generation import CohereAsrModel

        model = CohereAsrModel.from_path(model_dir)
        kwargs = {"sample_rate": sample_rate, "language": "en", "punctuation": punctuation}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        transcribe = lambda: model.transcribe(audio, **kwargs)
    elif runtime == "mlx_audio_python":
        from mlx_audio.stt import utils as mlx_stt_utils

        model = mlx_stt_utils.load(str(model_dir), strict=False)
        transcribe = lambda: model.generate(audio, sample_rate=sample_rate, language="en")
    else:
        raise ValueError(f"unsupported runtime: {runtime}")
    mx.eval(mx.array([0]))
    load_seconds = time.perf_counter() - load_start

    timings: list[float] = []
    texts: list[str] = []
    for idx in range(runs):
        start = time.perf_counter()
        result = transcribe()
        mx.eval(mx.array([0]))
        elapsed = time.perf_counter() - start
        text = str(getattr(result, "text", result)).strip()
        timings.append(elapsed)
        texts.append(text)
        print(f"{variant} run {idx + 1}/{runs}: {elapsed:.3f}s :: {text[:160]}", flush=True)

    return {
        "variant": variant,
        "repo_id": VARIANTS[variant]["repo_id"],
        "model_dir": str(model_dir),
        "runtime": runtime,
        "max_new_tokens": max_new_tokens,
        "punctuation": punctuation,
        "load_seconds": load_seconds,
        "runs": timings,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "text": texts[-1] if texts else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant(s) to run. Defaults to 8-bit plus Swift 4-bit when the Swift CLI is installed.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--swift-stt-bin", help="Path to mlx-audio-swift-stt for the 4-bit Swift variant.")
    parser.add_argument("--max-new-tokens", type=int, help="Override Cohere MLX max_new_tokens for benchmark runs.")
    parser.add_argument("--no-punctuation", action="store_true", help="Benchmark Cohere MLX with punctuation disabled.")
    args = parser.parse_args()

    if args.variant:
        variants = args.variant
    else:
        variants = ["cohere-mlx-8bit"]
        if find_swift_stt_server_bin(args.swift_stt_bin):
            variants.append("cohere-mlx-4bit-swift-server")
        else:
            print(
            "Swift 4-bit server runtime not found; default benchmark will run 8-bit only. "
                "Run scripts/install_mlx_audio_swift.sh or pass --swift-stt-bin to include 4-bit.",
                flush=True,
            )
    audio, sample_rate, audio_seconds = load_audio_16k(args.audio)
    print(f"Audio: {args.audio} ({audio_seconds:.2f}s @ {sample_rate} Hz)", flush=True)

    results = {
        "audio": str(args.audio),
        "audio_seconds": audio_seconds,
        "runs_per_variant": args.runs,
        "variants": [],
    }
    for variant in variants:
        results["variants"].append(
            run_variant(
                variant,
                args.audio,
                audio,
                sample_rate,
                args.runs,
                swift_stt_bin=args.swift_stt_bin,
                max_new_tokens=args.max_new_tokens,
                punctuation=not args.no_punctuation,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
