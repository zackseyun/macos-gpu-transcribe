#!/usr/bin/env python3
"""Benchmark Cohere Transcribe MLX quantized variants against local audio.

This is intentionally standalone so we can test quantized Cohere without
changing the live Fn dictation path.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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
    # checkpoint layout. It loads with mlx-audio only with strict=False in the
    # current Python runtime, and local tests produced unusable multilingual
    # gibberish. Keep it benchmarkable so we can retest after runtime updates.
    "cohere-mlx-4bit-experimental": {
        "repo_id": "beshkenadze/cohere-transcribe-03-2026-mlx-4bit",
        "subdir": ".",
        "runtime": "mlx_audio",
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


def run_variant(variant: str, audio: np.ndarray, sample_rate: int, runs: int) -> dict:
    import mlx.core as mx

    model_dir = resolve_model_dir(variant)
    runtime = VARIANTS[variant]["runtime"]
    load_start = time.perf_counter()
    if runtime == "mlx_speech":
        from mlx_speech.generation import CohereAsrModel

        model = CohereAsrModel.from_path(model_dir)
        transcribe = lambda: model.transcribe(audio, sample_rate=sample_rate, language="en")
    elif runtime == "mlx_audio":
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
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="Variant(s) to run. Defaults to both.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    variants = args.variant or sorted(VARIANTS)
    audio, sample_rate, audio_seconds = load_audio_16k(args.audio)
    print(f"Audio: {args.audio} ({audio_seconds:.2f}s @ {sample_rate} Hz)", flush=True)

    results = {
        "audio": str(args.audio),
        "audio_seconds": audio_seconds,
        "runs_per_variant": args.runs,
        "variants": [],
    }
    for variant in variants:
        results["variants"].append(run_variant(variant, audio, sample_rate, args.runs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
