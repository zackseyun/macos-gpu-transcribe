"""Small, reproducible English accuracy/latency comparison; not a general ranking."""
import argparse
import io
import json
import pathlib
import re
import statistics
import sys
import time

import jiwer
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.stt import load

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlx_next_asr import generate


def normalize(text):
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    text = re.sub(r"\bmr\b", "mister", text)
    text = re.sub(r"\bm\s+a\b", "ma", text)
    return text.split()


def suite(corpus):
    if corpus.suffix == ".json":
        rows = json.loads(corpus.read_text())
    else:
        import pyarrow.parquet as pq
        source = pq.read_table(corpus).to_pylist()
        rows = [dict(source[i], index=i) for i in [0, 2, 4, 7, 10, 14, 20, 25]]
    cases = []
    rng = np.random.default_rng(42)
    for row in rows:
        index = row["index"]
        source = row.get("file") or io.BytesIO(row["audio"]["bytes"])
        audio, sample_rate = sf.read(source, dtype="float32")
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError("Fixtures must be mono 16 kHz")
        reference = row["text"]
        cases.append((f"clean-{index}", audio, reference))
        rms = np.sqrt(np.mean(audio * audio))
        noise = rng.normal(0, rms / 10 ** (15 / 20), audio.shape).astype(np.float32)
        cases.append((f"noise15db-{index}", audio + noise, reference))
    clean = [case for case in cases if case[0].startswith("clean")] * 2
    audio = np.concatenate([
        np.concatenate([case[1], np.zeros(6400, dtype=np.float32)]) for case in clean
    ])
    cases.append(("long", audio, " ".join(case[2] for case in clean)))
    return cases


def timed_generate(model, audio, mode):
    mx.synchronize()
    start = time.perf_counter()
    text = generate(model, audio, mode)
    mx.synchronize()
    return text, time.perf_counter() - start


def run(mode, corpus, out):
    out.mkdir(parents=True, exist_ok=True)
    cases = suite(corpus)
    model_dir = ROOT / ".mlx-next/models" / (
        "nemotron" if mode == "nemotron35" else "granite5"
    )
    start = time.perf_counter()
    model = load(str(model_dir))
    cold = time.perf_counter() - start
    print(f"loaded {mode} {cold:.2f}s", flush=True)
    timed_generate(model, cases[0][1], mode)
    results, repeats = [], []

    def save():
        payload = dict(mode=mode, cold_load_seconds=cold, results=results,
                       warm_repeats=repeats, peak_gb=mx.get_peak_memory() / 1024**3)
        (out / f"{mode}-results.json").write_text(json.dumps(payload, indent=2))

    for name, audio, reference in cases:
        text, elapsed = timed_generate(model, audio, mode)
        score = jiwer.process_words(
            " ".join(normalize(reference)), " ".join(normalize(text))
        )
        result = dict(
            name=name, duration=len(audio) / 16000, seconds=elapsed,
            reference=reference, text=text, wer=score.wer,
            errors=score.substitutions + score.deletions + score.insertions,
            words=len(normalize(reference)),
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        save()
    for index in [0, 2, 14]:
        name, audio, _ = cases[index]
        for repeat in range(3):
            text, elapsed = timed_generate(model, audio, mode)
            repeats.append(dict(name=name, repeat=repeat, seconds=elapsed, text=text))
    save()
    print("SUMMARY", mode,
          "WER", sum(r["errors"] for r in results) / sum(r["words"] for r in results),
          "median_s", statistics.median(r["seconds"] for r in results),
          "peak_GB", mx.get_peak_memory() / 1024**3, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["granite5", "nemotron35"])
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.corpus, args.out)
