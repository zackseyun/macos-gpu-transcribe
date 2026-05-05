"""Transcription worker subprocess — loads model once, transcribes on demand.

Supports three model modes:
  - "granite": Granite Speech 4.1 NAR via CrispASR/GGUF (Metal if available)
  - "fast": Qwen3-ASR 0.6B via MLX (Metal)
  - "accurate": Qwen3-ASR 1.7B via MLX (Metal)
  - "cohere": Cohere Transcribe 2B via PyTorch (MPS)

Cold-start mitigation:
  MPS (Metal) evicts compiled shader state after ~30-60s of GPU idle, which
  makes the "first inference after idle" 10-25x slower than warm inference.
  To prevent this, the worker pre-warms the model right after load and runs a
  silent keep-warm inference every KEEP_WARM_INTERVAL seconds from a background
  thread. The keep-warm holds a lock so it never races with real transcriptions.
"""
import json
import os
import signal
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path


# Metal cache limit — caps GPU buffer cache to 6GB (both models + headroom)
METAL_CACHE_LIMIT_BYTES = 6 * 1024 * 1024 * 1024

QWEN3_MODEL_IDS = {
    "fast": "Qwen/Qwen3-ASR-0.6B",
    "accurate": "Qwen/Qwen3-ASR-1.7B",
}

COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
GRANITE_CRISPASR_BACKEND = os.getenv("VOICE_TRANSCRIBE_GRANITE_BACKEND", "granite-4.1-nar")
GRANITE_CRISPASR_MODEL = os.getenv("VOICE_TRANSCRIBE_GRANITE_MODEL", "auto")
GRANITE_CRISPASR_LANGUAGE = os.getenv("VOICE_TRANSCRIBE_GRANITE_LANGUAGE", "en")
MAX_SCREEN_CONTEXT_CHARS = 320

# Keep-warm cadence — must be SHORTER than MPS eviction window.
# Empirically macOS evicts Metal shader state after ~30s of GPU idle. We use 20s
# to stay comfortably under that. Sleep granularity is 2s so worst-case fire is ~22s.
KEEP_WARM_INTERVAL = float(os.getenv("VOICE_TRANSCRIBE_KEEP_WARM_INTERVAL", "20"))
KEEP_WARM_CHECK_INTERVAL = 2.0
KEEP_WARM_AUDIO_SECONDS = 0.5  # 0.5s of silence = trivially fast dummy inference

# After this many seconds of no real transcription, stop keep-warming. The next
# user action will trigger an on-demand warm via the __warm__ message instead.
# This prevents constant GPU heat during long idle periods.
KEEP_WARM_MAX_IDLE = float(os.getenv("VOICE_TRANSCRIBE_KEEP_WARM_MAX_IDLE", "300"))

# Skip on-demand warm if model was used within this many seconds (already hot).
ON_DEMAND_WARM_SKIP_THRESHOLD = 15.0


def _sanitize_screen_context(screen_context):
    if not screen_context:
        return ""
    compact = " ".join(str(screen_context).split()).strip()
    return compact[:MAX_SCREEN_CONTEXT_CHARS]


def _build_cohere_decoder_prompt_ids(processor, screen_context):
    import torch

    prompt_ids = processor.get_decoder_prompt_ids(language="en", punctuation=True)
    screen_context = _sanitize_screen_context(screen_context)
    if not screen_context:
        return torch.tensor([prompt_ids], dtype=torch.long)

    context_ids = processor.tokenizer.encode(screen_context, add_special_tokens=False)
    if not context_ids:
        return torch.tensor([prompt_ids], dtype=torch.long)

    transcript_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    try:
        transcript_idx = prompt_ids.index(transcript_token_id)
    except ValueError:
        transcript_idx = 2

    custom_prompt_ids = prompt_ids[:transcript_idx] + context_ids[:96] + prompt_ids[transcript_idx:]
    return torch.tensor([custom_prompt_ids], dtype=torch.long)


def _transcribe_cohere(wav_path, processor, model, screen_context=""):
    """Transcribe using Cohere via PyTorch MPS."""
    import torch
    from transformers.audio_utils import load_audio

    audio = load_audio(wav_path, sampling_rate=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
    custom_prompt_ids = _build_cohere_decoder_prompt_ids(processor, screen_context)
    batch_size = inputs["input_features"].shape[0]
    if batch_size > 1:
        custom_prompt_ids = custom_prompt_ids.repeat(batch_size, 1)
    inputs["decoder_input_ids"] = custom_prompt_ids
    inputs["decoder_attention_mask"] = torch.ones_like(custom_prompt_ids)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            no_repeat_ngram_size=4,  # prevent autoregressive loop degeneration
        )

    prompt_len = inputs["decoder_input_ids"].shape[1]
    if outputs.ndim == 2 and outputs.shape[1] >= prompt_len:
        if torch.equal(outputs[:, :prompt_len], inputs["decoder_input_ids"]):
            outputs = outputs[:, prompt_len:]

    result = processor.decode(outputs, skip_special_tokens=True)
    text = " ".join(result).strip() if isinstance(result, list) else result.strip()
    return text


def _write_silent_wav(path, seconds=KEEP_WARM_AUDIO_SECONDS, sample_rate=16000):
    """Write a short silent WAV used for pre-warm / keep-warm dummy inferences."""
    num_samples = int(seconds * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)


def _prewarm_cohere(processor, model, warm_wav_path):
    """Run a single silent inference to pay the MPS compile/load cost upfront."""
    t0 = time.time()
    try:
        _transcribe_cohere(warm_wav_path, processor, model, screen_context="")
    except Exception as exc:
        print(f"Transcription worker: Cohere pre-warm failed: {exc}", flush=True)
        return
    print(f"Transcription worker: Cohere pre-warm done ({time.time() - t0:.1f}s)", flush=True)


def _find_crispasr_bin():
    configured = os.getenv("VOICE_TRANSCRIBE_CRISPASR_BIN")
    candidates = [
        configured,
        str(Path(__file__).resolve().parent / ".crispasr" / "build" / "bin" / "crispasr"),
        shutil.which("crispasr"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _extract_crispasr_text(stdout):
    """Parse CrispASR JSON output; fall back to cleaned stdout if needed."""
    raw = (stdout or "").strip()
    if not raw:
        return ""

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
            segments = payload.get("transcription")
            if isinstance(segments, list):
                return " ".join(
                    str(segment.get("text", "")).strip()
                    for segment in segments
                    if isinstance(segment, dict) and segment.get("text")
                ).strip()
            for key in ("text", "transcript", "result"):
                if payload.get(key):
                    return str(payload[key]).strip()
        except Exception:
            pass

    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith(("[", "{", "}"))
    ]
    return " ".join(lines).strip()


def _transcribe_granite_crispasr(wav_path, screen_context=""):
    """Transcribe Granite Speech 4.1 NAR through CrispASR's Metal-capable GGUF runtime."""
    crispasr_bin = _find_crispasr_bin()
    if crispasr_bin is None:
        raise RuntimeError(
            "Granite Speech is selected, but CrispASR is not installed. "
            "Run ./install.sh or build CrispASR at .crispasr/build/bin/crispasr. "
            "The Hugging Face PyTorch path for Granite NAR requires CUDA + flash_attention_2, "
            "so the Mac app uses CrispASR/GGUF for local Granite inference."
        )

    if screen_context:
        print(
            "Transcription worker: Granite/CrispASR does not accept screen-context prompts yet; ignoring context.",
            flush=True,
        )

    cmd = [
        crispasr_bin,
        "--backend",
        GRANITE_CRISPASR_BACKEND,
        "-m",
        GRANITE_CRISPASR_MODEL,
        "-f",
        wav_path,
        "-l",
        GRANITE_CRISPASR_LANGUAGE,
        "--auto-download",
        "-np",
        "-oj",
    ]
    gpu_backend = os.getenv("VOICE_TRANSCRIBE_CRISPASR_GPU_BACKEND")
    if gpu_backend:
        cmd[1:1] = ["--gpu-backend", gpu_backend]

    env = os.environ.copy()
    env.setdefault("CRISPASR_GGUF_MMAP", "1")
    timeout = float(os.getenv("VOICE_TRANSCRIBE_CRISPASR_TIMEOUT_SECONDS", "900"))
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(f"CrispASR Granite failed ({proc.returncode}): {detail}")

    text = _extract_crispasr_text(proc.stdout)
    if not text:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"CrispASR Granite returned no transcript. Output: {detail[:1200]}")
    return text


def run(request_pipe, result_pipe):
    """Receive requests via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Transcription worker started (PID {os.getpid()})", flush=True)

    # Set Metal cache limit upfront — prevents unbounded GPU buffer accumulation
    mx = None
    try:
        import mlx.core as mx
        mx.set_cache_limit(METAL_CACHE_LIMIT_BYTES)
        print(f"Transcription worker: Metal cache limit set to {METAL_CACHE_LIMIT_BYTES / (1024**3):.0f}GB", flush=True)
    except Exception as e:
        print(f"Transcription worker: failed to set cache limit: {e}", flush=True)

    qwen3_transcribe_fn = None
    cohere_processor = None
    cohere_model = None

    # Persistent silent WAV for pre-warm + keep-warm (written once, reused forever)
    warm_wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".keep_warm_silent.wav"
    )
    try:
        _write_silent_wav(warm_wav_path)
    except Exception as exc:
        print(f"Transcription worker: failed to create keep-warm wav: {exc}", flush=True)
        warm_wav_path = None

    # Lock serializes keep-warm with real transcriptions so they never race on MPS.
    inference_lock = threading.Lock()
    last_real_inference_at = [time.time()]  # last real (non-warm) transcription
    last_any_inference_at = [time.time()]   # last any inference (real OR warm)

    def _do_warm(reason):
        """Run a silent dummy inference. Returns True if it actually ran."""
        if cohere_model is None or warm_wav_path is None:
            return False
        if not inference_lock.acquire(blocking=False):
            return False
        try:
            t0 = time.time()
            _transcribe_cohere(
                warm_wav_path, cohere_processor, cohere_model, screen_context=""
            )
            last_any_inference_at[0] = time.time()
            dt = time.time() - t0
            if dt > 1.5:
                print(
                    f"Transcription worker: {reason}-warm took {dt:.1f}s "
                    f"(MPS was cold — caught a stale state)",
                    flush=True,
                )
            return True
        except Exception as exc:
            print(f"Transcription worker: {reason}-warm failed: {exc}", flush=True)
            return False
        finally:
            inference_lock.release()

    def _keep_warm_loop():
        """Background thread: tight keep-warm loop that bounds itself to active sessions.

        Fires every KEEP_WARM_INTERVAL seconds while a real transcription has happened
        within KEEP_WARM_MAX_IDLE seconds. After that, it goes quiet — the next Fn press
        will trigger on-demand warming via the __warm__ message instead. This prevents
        wasted GPU/heat during long idle periods (overnight, walking away, etc.).
        """
        while True:
            try:
                time.sleep(KEEP_WARM_CHECK_INTERVAL)
                if cohere_model is None or warm_wav_path is None:
                    continue
                now = time.time()
                # Stop keep-warming if no real activity recently — let GPU sleep.
                if now - last_real_inference_at[0] > KEEP_WARM_MAX_IDLE:
                    continue
                # Don't warm if we just warmed/transcribed.
                if now - last_any_inference_at[0] < KEEP_WARM_INTERVAL:
                    continue
                _do_warm("keep")
            except Exception as exc:
                print(f"Transcription worker: keep-warm loop error: {exc}", flush=True)

    threading.Thread(target=_keep_warm_loop, daemon=True).start()

    while True:
        try:
            request = request_pipe.recv()
        except (EOFError, OSError):
            break

        if request == "__quit__":
            break

        # On-demand warm signal: main app sends this when user presses the Fn key
        # so by the time they release the key (1-15s later) MPS is already hot.
        # Run in a thread so the request loop returns to recv() immediately for
        # the upcoming real transcription request.
        if isinstance(request, dict) and request.get("__warm__"):
            if request.get("model_mode", "cohere") != "cohere":
                continue
            if cohere_model is None or warm_wav_path is None:
                continue
            now = time.time()
            if now - last_any_inference_at[0] < ON_DEMAND_WARM_SKIP_THRESHOLD:
                continue  # already hot, skip
            threading.Thread(
                target=_do_warm, args=("on-demand",), daemon=True
            ).start()
            continue

        # Support both old format (string path) and new format (dict with model_mode)
        if isinstance(request, str):
            wav_path = request
            model_mode = "fast"
            screen_context = ""
        else:
            wav_path = request["wav_path"]
            model_mode = request.get("model_mode", "fast")
            screen_context = _sanitize_screen_context(request.get("screen_context", ""))

        try:
            t0 = time.time()

            if model_mode == "granite":
                text = _transcribe_granite_crispasr(
                    wav_path,
                    screen_context=screen_context,
                )
            elif model_mode == "cohere":
                # Lazy-load Cohere model on first use
                if cohere_model is None:
                    load_t0 = time.time()
                    print("Transcription worker: loading Cohere Transcribe...", flush=True)
                    import torch
                    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
                    cohere_processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
                    cohere_model = CohereAsrForConditionalGeneration.from_pretrained(
                        COHERE_MODEL_ID, torch_dtype=torch.float32,
                    ).to("mps")
                    print(
                        f"Transcription worker: Cohere loaded on {cohere_model.device} "
                        f"({time.time() - load_t0:.1f}s)",
                        flush=True,
                    )
                    # Pre-warm so the first real inference isn't 8s+ of MPS compile
                    if warm_wav_path is not None:
                        _prewarm_cohere(cohere_processor, cohere_model, warm_wav_path)
                        last_any_inference_at[0] = time.time()

                with inference_lock:
                    text = _transcribe_cohere(
                        wav_path,
                        cohere_processor,
                        cohere_model,
                        screen_context=screen_context,
                    )
                    now = time.time()
                    last_real_inference_at[0] = now
                    last_any_inference_at[0] = now
            else:
                # Qwen3 via MLX
                model_id = QWEN3_MODEL_IDS.get(model_mode, QWEN3_MODEL_IDS["fast"])
                if qwen3_transcribe_fn is None:
                    print("Transcription worker: loading Qwen3 transcribe function...", flush=True)
                    from mlx_qwen3_asr import transcribe as _transcribe_fn
                    qwen3_transcribe_fn = _transcribe_fn
                    print("Transcription worker: Qwen3 ready", flush=True)

                raw = qwen3_transcribe_fn(
                    wav_path,
                    model=model_id,
                    context=screen_context,
                )

                # Clear MLX Metal cache after each transcription
                if mx is not None:
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass

                if isinstance(raw, dict):
                    text = raw.get("text", "").strip()
                elif hasattr(raw, "text"):
                    text = raw.text.strip()
                else:
                    text = str(raw).strip()

            elapsed = time.time() - t0
            context_note = f", screen_context={len(screen_context)} chars" if screen_context else ""
            print(f"Transcription worker: [{model_mode}] {elapsed:.1f}s{context_note}", flush=True)
            result_pipe.send({"text": text, "time": elapsed, "error": None})

        except Exception as e:
            result_pipe.send({"text": "", "time": 0, "error": str(e)})

    print("Transcription worker: exiting", flush=True)
