#!/usr/bin/env python3
"""Probe live-state latency and GPU memory of the Cohere MLX 8-bit worker path.

Loads the same checkpoint the transcription worker uses and times the same
encode -> prefill -> greedy decode loop (mlx-speech internals), reporting
encoder vs decoder time, ms/token, and MLX active/cache/peak memory. Use it to
tell "the machine is throttled or contended" apart from "the app is slow":

    .venv/bin/python3 scripts/probe_cohere_mlx_latency.py            # instrumented
    .venv/bin/python3 scripts/probe_cohere_mlx_latency.py --simple   # model.transcribe() only
    .venv/bin/python3 scripts/probe_cohere_mlx_latency.py --wired    # with mx.set_wired_limit
    .venv/bin/python3 scripts/probe_cohere_mlx_latency.py --qos interactive

On an M4 Max with Low Power Mode off and an idle GPU, expect ~40x real-time.
2026-09-06 measurements: the same clip swung between 42x and 3x real-time within
minutes purely from Low Power Mode + Chrome/WindowServer GPU contention; thread
QoS made no difference, confirming the GPU (not CPU scheduling) was the limit.
"""
import argparse, ctypes, os, sys, time, wave
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]

QOS = {"interactive": 0x21, "initiated": 0x19, "default": 0x15, "utility": 0x11, "background": 0x09}

def set_qos(name):
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    fn = libc.pthread_set_qos_class_self_np
    fn.argtypes = [ctypes.c_uint, ctypes.c_int]; fn.restype = ctypes.c_int
    return fn(QOS[name], 0)

def load_wav(path):
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes()); ch = wf.getnchannels()
    a = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1).astype(np.float32)
    return a

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default=str(REPO_DIR / "last_recording.wav"))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--qos", default=None, choices=list(QOS))
    p.add_argument("--wired", action="store_true")
    p.add_argument("--warm-tokens", type=int, default=None)
    p.add_argument("--label", default="")
    p.add_argument("--simple", action="store_true", help="time model.transcribe() only (portable across mlx-speech versions)")
    a = p.parse_args()
    tag = f"[{a.label}] " if a.label else ""
    if a.qos:
        print(tag + f"qos={a.qos} rc={set_qos(a.qos)}")
    import mlx.core as mx
    mx.set_cache_limit(6 * 1024 ** 3)
    info = mx.device_info()
    print(tag + f"device_info={info}")
    if a.wired:
        lim = info["max_recommended_working_set_size"]
        old = mx.set_wired_limit(lim)
        print(tag + f"wired limit {old} -> {lim}")
    from huggingface_hub import snapshot_download
    from mlx_speech.generation import CohereAsrModel
    t = time.perf_counter()
    root = snapshot_download(repo_id="mlx-community/cohere-transcribe-03-2026-mlx-8bit",
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "mlx-int8/*"],
                             local_files_only=True)
    model = CohereAsrModel.from_path(os.path.join(root, "mlx-int8"))
    print(tag + f"load {time.perf_counter() - t:.2f}s")
    sil = np.zeros(8000, dtype=np.float32)
    kw = {"max_new_tokens": a.warm_tokens} if a.warm_tokens else {}
    for i in range(2):
        t = time.perf_counter()
        r = model.transcribe(sil, sample_rate=16000, language="en", **kw)
        print(tag + f"silence(0.5s) warm #{i}: {time.perf_counter() - t:.2f}s tokens={len(r.tokens)} text={r.text[:50]!r}")
    audio = load_wav(a.audio); dur = len(audio) / 16000
    if a.simple:
        import mlx_speech
        ver = getattr(mlx_speech, "__version__", "?")
        for i in range(a.runs):
            t0 = time.perf_counter(); r = model.transcribe(audio, sample_rate=16000, language="en"); total = time.perf_counter() - t0
            print(tag + f"simple run {i}: mlx_speech={ver} audio={dur:.1f}s total={total:.2f}s ({dur / total:.1f}x RT) tokens={len(r.tokens)}")
        print(tag + f"mem: active={mx.get_active_memory()/1024**3:.2f}GB cache={mx.get_cache_memory()/1024**3:.2f}GB peak={mx.get_peak_memory()/1024**3:.2f}GB")
        print(tag + "text: " + r.text[:160]); return
    fe = model.feature_extractor; eos = model.config.decoder.eos_token_id
    texts = []
    for i in range(a.runs):
        t0 = time.perf_counter()
        chunks = fe.process_audio(audio); t1 = time.perf_counter()
        ntok = 0; enc_t = 0.0; dec_t = 0.0; texts = []
        for features, mask in chunks:
            f = mx.array(features)[None]
            m = mx.array(mask, dtype=mx.bool_)[None] if mask is not None else None
            te = time.perf_counter(); enc, encm = model.model.encode(f, m); mx.eval(enc); enc_t += time.perf_counter() - te
            td = time.perf_counter()
            prompt = model.tokenizer.get_decoder_prompt_ids("en", True, itn=False)
            logits, skv, ckv = model.model.decode_step(mx.array([prompt], dtype=mx.int32), enc, encm,
                                                       self_kv_caches=None, cross_kv_caches=None, position_offset=0)
            mx.eval(logits)
            nt = int(logits[0, -1].argmax()); gen = [nt]; pos = len(prompt)
            for _ in range(447):
                if nt == eos: break
                logits, skv, ckv = model.model.decode_step(mx.array([[nt]], dtype=mx.int32), enc, encm,
                                                           self_kv_caches=skv, cross_kv_caches=ckv, position_offset=pos)
                mx.eval(logits)
                nt = int(logits[0, 0].argmax()); gen.append(nt); pos += 1
            dec_t += time.perf_counter() - td; ntok += len(gen)
            texts.append(model.tokenizer.decode(gen, skip_special_tokens=True))
        total = time.perf_counter() - t0
        print(tag + f"run {i}: audio={dur:.1f}s chunks={len(chunks)} total={total:.2f}s ({dur / total:.1f}x RT) "
              f"feat={t1 - t0:.2f}s enc={enc_t:.2f}s dec={dec_t:.2f}s tokens={ntok} "
              f"({ntok / dec_t if dec_t else 0:.0f} tok/s, {1000 * dec_t / max(ntok, 1):.1f} ms/tok)")
    print(tag + f"mem: active={mx.get_active_memory()/1024**3:.2f}GB cache={mx.get_cache_memory()/1024**3:.2f}GB peak={mx.get_peak_memory()/1024**3:.2f}GB")
    print(tag + "text: " + " ".join(texts)[:160])

if __name__ == "__main__":
    main()
