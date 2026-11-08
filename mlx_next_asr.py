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
        model = prefix = None
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if model is None:
                    model = load(str(ROOT / ".mlx-next/models" / MODEL_DIRS[mode]))
                    if mode == "nemotron35":
                        prefix = NemotronPrefixCache(model)
                if request.get("warm"):
                    # Compile the same path used for speech, without recording.
                    audio = np.zeros(16000, dtype=np.float32)
                    generate(model, audio, mode, max_tokens=128, prefix=prefix)
                    response = {"ready": True}
                else:
                    audio, sr = sf.read(io.BytesIO(base64.b64decode(request["wav"])), dtype="float32")
                    if sr != 16000 or audio.ndim != 1 or not len(audio):
                        raise ValueError("Expected nonempty mono 16 kHz audio")
                    response = {"text": generate(model, audio, mode, prefix=prefix)}
                mx.clear_cache()
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(response) + "\n")
            protocol.flush()


class NemotronPrefixCache:
    """Reuse the constant prompt prefix across Nemotron requests.

    Nemotron's ASR prompt puts the audio after 56 fixed tokens (empty system
    turn plus the transcription instruction). Their attention and Mamba state is
    computed once at load; each request prefills only its audio and the 8-token
    assistant suffix. Greedy output matched model.generate() exactly on every
    test clip, and it saves roughly 0.4-0.6 s per dictation in Low Power Mode.
    """

    def __init__(self, model):
        import mlx.core as mx
        from mlx_audio.lm.models.cache import KVCache
        from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import resolve_prompt
        self.model = model
        tag = model.config.audio_locator_tag
        prompt = resolve_prompt("asr", None, None, None, tag)
        # Same turns as Model._prepare_inputs (no system prompt, thinking off).
        turns = [
            "<|im_start|>system\n<|im_end|>\n",
            f"<|im_start|>user\n{prompt}<|im_end|>\n",
            "<|im_start|>assistant\n<think></think>",
        ]
        tokenizer = model._tokenizer
        ids = [t for turn in turns for t in tokenizer.encode(turn, add_special_tokens=False)]
        index = ids.index(tokenizer.convert_tokens_to_ids(tag))
        self.prefix_ids, self.suffix_ids = ids[:index], ids[index + 1:]
        embed = model.llm.backbone.embeddings
        self.suffix_embeddings = embed(mx.array(self.suffix_ids)[None])
        prefix = mx.array(self.prefix_ids)[None]
        cache = model.make_cache()
        model(prefix, cache=cache, input_embeddings=embed(prefix))
        mx.eval([c.state for c in cache], self.suffix_embeddings)
        self._saved = [("kv", c.state) if isinstance(c, KVCache) else ("ssm", list(c.state))
                       for c in cache]

    def _fresh_cache(self):
        from mlx_audio.lm.models.cache import KVCache
        from mlx_audio.stt.models.nemotron_transcribe.language_model import MambaCache
        caches = []
        for kind, state in self._saved:
            # The saved KV arrays are exactly prefix-sized, so the first update
            # reallocates and later in-place writes never touch the saved prefix.
            cache = KVCache() if kind == "kv" else MambaCache()
            cache.state = state if kind == "kv" else list(state)
            caches.append(cache)
        return caches

    def generate(self, audio, max_tokens):
        import types
        import mlx.core as mx
        from mlx_audio.lm.generate import StreamingDetokenizer, generate_step
        from mlx_audio.stt.utils import wired_limit
        model = self.model
        audio = mx.array(audio, dtype=mx.float32)
        if audio.ndim != 1 or audio.size < model.config.processor_config.hop_length:
            raise ValueError("Expected nonempty mono audio at the model sample rate")
        if not mx.all(mx.isfinite(audio)).item():
            raise ValueError("Audio contains non-finite samples")
        with wired_limit(model):
            audio_embeddings = model.perception(audio)
            embeddings = mx.concatenate(
                [audio_embeddings.astype(self.suffix_embeddings.dtype), self.suffix_embeddings],
                axis=1)[0]
            tokens = mx.array([model.config.pad_token_id] * audio_embeddings.shape[1]
                              + self.suffix_ids)
            decoder = StreamingDetokenizer(model._tokenizer, skip_special_tokens=False)
            count, finish_reason = 0, "length"
            for token, _ in generate_step(tokens, model, input_embeddings=embeddings,
                                          prompt_cache=self._fresh_cache(),
                                          max_tokens=max_tokens, prefill_step_size=512):
                if token in model.config.eos_token_id:
                    finish_reason = "stop"
                    break
                count += 1
                decoder.add_token(token)
            decoder.finalize()
        return types.SimpleNamespace(text=decoder.text.strip(), finish_reason=finish_reason,
                                     generation_tokens=count)


def generate(model, audio, mode, max_tokens=None, prefix=None):
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
        if prefix is not None:
            result = prefix.generate(chunk, budget)
        else:
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
