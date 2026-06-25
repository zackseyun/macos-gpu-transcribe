"""Transcription worker subprocess — loads model once, transcribes on demand.

Supports three model modes:
  - "fast": Qwen3-ASR 0.6B via MLX (Metal)
  - "accurate": Qwen3-ASR 1.7B via MLX (Metal)
  - "cohere": optional legacy Cohere Transcribe 2B via PyTorch (MPS)

Cold-start mitigation:
  MPS (Metal) evicts compiled shader state after ~30-60s of GPU idle, which
  makes the "first inference after idle" 10-25x slower than warm inference.
  To prevent this, the worker pre-warms the model right after load and runs a
  silent keep-warm inference every KEEP_WARM_INTERVAL seconds from a background
  thread. The keep-warm holds a lock so it never races with real transcriptions.
"""
import os
import signal
import sys
import threading
import time
import tempfile
import wave

import numpy as np


LOG_FILE = "/tmp/voice-transcribe.log"


def _log(message):
    print(message, flush=True)
    if getattr(sys.stdout, "name", None) == LOG_FILE:
        return
    try:
        with open(LOG_FILE, "a", buffering=1) as log:
            print(message, file=log, flush=True)
    except OSError:
        pass


# Metal cache limit — caps GPU buffer cache to 6GB (both models + headroom)
METAL_CACHE_LIMIT_BYTES = 6 * 1024 * 1024 * 1024

QWEN3_MODEL_IDS = {
    "fast": "Qwen/Qwen3-ASR-0.6B",
    "accurate": "Qwen/Qwen3-ASR-1.7B",
}

COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
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

STREAM_CHUNK_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_STREAM_CHUNK_SECONDS", "2.0"))
STREAM_MAX_CONTEXT_SECONDS = float(os.getenv("VOICE_TRANSCRIBE_STREAM_MAX_CONTEXT_SECONDS", "30"))
STREAM_FINALIZATION_MODE = os.getenv("VOICE_TRANSCRIBE_STREAM_FINALIZATION_MODE", "accuracy")
STREAM_ENDPOINTING_MODE = os.getenv("VOICE_TRANSCRIBE_STREAM_ENDPOINTING_MODE", "energy")
STREAM_TAIL_SILENCE_SECONDS = float(
    os.getenv("VOICE_TRANSCRIBE_STREAM_TAIL_SILENCE_SECONDS", "0.35")
)
STREAM_SAMPLE_RATE = 16000


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


def run(request_pipe, result_pipe):
    """Receive requests via request_pipe, send results via result_pipe."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _log(f"Transcription worker started (PID {os.getpid()})")

    # Set Metal cache limit upfront — prevents unbounded GPU buffer accumulation
    mx = None
    try:
        import mlx.core as mx
        mx.set_cache_limit(METAL_CACHE_LIMIT_BYTES)
        _log(f"Transcription worker: Metal cache limit set to {METAL_CACHE_LIMIT_BYTES / (1024**3):.0f}GB")
    except Exception as e:
        _log(f"Transcription worker: failed to set cache limit: {e}")

    qwen3_transcribe_fn = None
    qwen3_loaded_modes = set()
    qwen3_streaming_api = None
    stream_sessions = {}
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

    def _get_qwen3_transcribe_fn():
        nonlocal qwen3_transcribe_fn
        if qwen3_transcribe_fn is None:
            _log("Transcription worker: loading Qwen3 transcribe function...")
            from mlx_qwen3_asr import transcribe as _transcribe_fn
            qwen3_transcribe_fn = _transcribe_fn
            _log("Transcription worker: Qwen3 ready")
        return qwen3_transcribe_fn

    def _transcribe_qwen3(audio_input, model_mode, screen_context="", *, warm=False):
        model_id = QWEN3_MODEL_IDS.get(model_mode, QWEN3_MODEL_IDS["fast"])
        transcribe_fn = _get_qwen3_transcribe_fn()
        kwargs = {
            "model": model_id,
            "language": "en",
            "context": screen_context,
        }
        if warm:
            kwargs["max_new_tokens"] = 16
        return transcribe_fn(audio_input, **kwargs)

    def _get_qwen3_streaming_api():
        nonlocal qwen3_streaming_api
        if qwen3_streaming_api is None:
            from mlx_qwen3_asr.streaming import (
                feed_audio,
                finish_streaming,
                init_streaming,
                streaming_metrics,
            )

            qwen3_streaming_api = {
                "feed_audio": feed_audio,
                "finish_streaming": finish_streaming,
                "init_streaming": init_streaming,
                "streaming_metrics": streaming_metrics,
            }
            _log("Transcription worker: Qwen3 streaming ready")
        return qwen3_streaming_api

    def _do_warm(reason, model_mode="fast", allow_load=False):
        """Run a silent dummy inference. Returns True if it actually ran."""
        if stream_sessions:
            return False
        if warm_wav_path is None:
            return False
        if model_mode != "cohere" and model_mode not in qwen3_loaded_modes and not allow_load:
            return False
        if model_mode == "cohere" and cohere_model is None:
            return False
        if not inference_lock.acquire(blocking=False):
            return False
        try:
            t0 = time.time()
            if model_mode == "cohere":
                _transcribe_cohere(
                    warm_wav_path, cohere_processor, cohere_model, screen_context=""
                )
            else:
                _transcribe_qwen3(warm_wav_path, model_mode, warm=True)
                qwen3_loaded_modes.add(model_mode)
            last_any_inference_at[0] = time.time()
            dt = time.time() - t0
            if reason != "keep" or dt > 5.0:
                _log(
                    f"Transcription worker: [{model_mode}] {reason}-warm took {dt:.1f}s"
                )
            return True
        except Exception as exc:
            _log(f"Transcription worker: [{model_mode}] {reason}-warm failed: {exc}")
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
                if warm_wav_path is None:
                    continue
                now = time.time()
                # Stop keep-warming if no real activity recently — let GPU sleep.
                if now - last_real_inference_at[0] > KEEP_WARM_MAX_IDLE:
                    continue
                # Don't warm if we just warmed/transcribed.
                if now - last_any_inference_at[0] < KEEP_WARM_INTERVAL:
                    continue
                for warmed_mode in list(qwen3_loaded_modes):
                    _do_warm("keep", warmed_mode)
                if cohere_model is not None:
                    _do_warm("keep", "cohere")
            except Exception as exc:
                _log(f"Transcription worker: keep-warm loop error: {exc}")

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
            if warm_wav_path is None:
                continue
            now = time.time()
            warm_mode = request.get("model_mode", "fast")
            allow_load = bool(request.get("allow_load", False))
            if not allow_load and now - last_any_inference_at[0] < ON_DEMAND_WARM_SKIP_THRESHOLD:
                continue  # already hot, skip
            threading.Thread(
                target=_do_warm, args=("on-demand", warm_mode, allow_load), daemon=True
            ).start()
            continue

        if isinstance(request, dict) and request.get("__stream_start__"):
            stream_id = request.get("stream_id")
            if not stream_id:
                continue
            model_mode = request.get("model_mode", "fast")
            model_id = QWEN3_MODEL_IDS.get(model_mode, QWEN3_MODEL_IDS["fast"])
            screen_context = _sanitize_screen_context(request.get("screen_context", ""))
            try:
                api = _get_qwen3_streaming_api()
                state = api["init_streaming"](
                    model=model_id,
                    context=screen_context,
                    language="en",
                    chunk_size_sec=STREAM_CHUNK_SECONDS,
                    max_context_sec=STREAM_MAX_CONTEXT_SECONDS,
                    finalization_mode=STREAM_FINALIZATION_MODE,
                    endpointing_mode=STREAM_ENDPOINTING_MODE,
                )
                stream_sessions[stream_id] = {
                    "state": state,
                    "model_mode": model_mode,
                    "feed_time": 0.0,
                    "started_at": time.time(),
                    "error": None,
                }
                _log(
                    "Transcription worker: streaming started "
                    f"({stream_id}, mode={model_mode}, chunk={STREAM_CHUNK_SECONDS:.1f}s, "
                    f"finalize={STREAM_FINALIZATION_MODE}, endpoint={STREAM_ENDPOINTING_MODE})"
                )
            except Exception as exc:
                stream_sessions[stream_id] = {
                    "state": None,
                    "model_mode": model_mode,
                    "feed_time": 0.0,
                    "started_at": time.time(),
                    "error": str(exc),
                }
                _log(f"Transcription worker: streaming start failed ({stream_id}): {exc}")
            continue

        if isinstance(request, dict) and request.get("__stream_feed__"):
            stream_id = request.get("stream_id")
            entry = stream_sessions.get(stream_id)
            if not entry or entry.get("error"):
                continue
            try:
                api = _get_qwen3_streaming_api()
                audio_np = np.asarray(request.get("audio"), dtype=np.float32).flatten()
                if audio_np.size == 0:
                    continue
                with inference_lock:
                    state = entry["state"]
                    before_chunks = int(getattr(state, "chunk_id", 0))
                    t0 = time.time()
                    state = api["feed_audio"](audio_np, state)
                    dt = time.time() - t0
                    entry["state"] = state
                    entry["feed_time"] += dt
                    after_chunks = int(getattr(state, "chunk_id", 0))
                    qwen3_loaded_modes.add(entry["model_mode"])
                    now = time.time()
                    last_real_inference_at[0] = now
                    last_any_inference_at[0] = now
                if after_chunks > before_chunks:
                    _log(
                        "Transcription worker: streaming fed "
                        f"({stream_id}, chunks={after_chunks}, +{dt:.1f}s)"
                    )
            except Exception as exc:
                entry["error"] = str(exc)
                _log(f"Transcription worker: streaming feed failed ({stream_id}): {exc}")
            continue

        if isinstance(request, dict) and request.get("__stream_finish__"):
            stream_id = request.get("stream_id")
            entry = stream_sessions.pop(stream_id, None)
            if not entry:
                result_pipe.send({
                    "text": "",
                    "time": 0,
                    "error": "Streaming session missing; falling back is recommended.",
                    "streaming": True,
                })
                continue
            if entry.get("error"):
                result_pipe.send({
                    "text": "",
                    "time": 0,
                    "error": entry["error"],
                    "streaming": True,
                })
                continue
            try:
                api = _get_qwen3_streaming_api()
                with inference_lock:
                    if (
                        STREAM_TAIL_SILENCE_SECONDS > 0
                        and len(getattr(entry["state"], "buffer", ())) > 0
                    ):
                        tail_silence = np.zeros(
                            int(STREAM_SAMPLE_RATE * STREAM_TAIL_SILENCE_SECONDS),
                            dtype=np.float32,
                        )
                        pad_t0 = time.time()
                        entry["state"] = api["feed_audio"](tail_silence, entry["state"])
                        entry["feed_time"] += time.time() - pad_t0
                    t0 = time.time()
                    state = api["finish_streaming"](entry["state"])
                    elapsed = time.time() - t0
                    metrics = api["streaming_metrics"](state)
                    qwen3_loaded_modes.add(entry["model_mode"])
                    now = time.time()
                    last_real_inference_at[0] = now
                    last_any_inference_at[0] = now
                text = str(getattr(state, "text", "") or "").strip()
                _log(
                    "Transcription worker: streaming finished "
                    f"({stream_id}, finalize={elapsed:.1f}s, feed={entry['feed_time']:.1f}s, "
                    f"chunks={metrics.get('chunks_processed', 0)}, "
                    f"tail_pad={STREAM_TAIL_SILENCE_SECONDS:.2f}s)"
                )
                result_pipe.send({
                    "text": text,
                    "time": elapsed,
                    "error": None,
                    "streaming": True,
                    "streaming_feed_time": entry["feed_time"],
                    "streaming_chunks": metrics.get("chunks_processed", 0),
                    "streaming_metrics": metrics,
                })
            except Exception as exc:
                result_pipe.send({
                    "text": "",
                    "time": 0,
                    "error": str(exc),
                    "streaming": True,
                })
            continue

        if isinstance(request, dict) and request.get("__stream_cancel__"):
            stream_id = request.get("stream_id")
            if stream_id in stream_sessions:
                stream_sessions.pop(stream_id, None)
                _log(f"Transcription worker: streaming canceled ({stream_id})")
            continue

        # Support old string paths plus the newer direct-audio request shape.
        if isinstance(request, str):
            audio_input = request
            model_mode = "fast"
            screen_context = ""
        else:
            audio_input = request.get("audio", request.get("wav_path"))
            model_mode = request.get("model_mode", "fast")
            screen_context = _sanitize_screen_context(request.get("screen_context", ""))

        try:
            t0 = time.time()

            if model_mode == "cohere":
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
                    wav_path = audio_input
                    temp_wav_path = None
                    if not isinstance(audio_input, (str, bytes, os.PathLike)):
                        temp_wav_path = tempfile.mktemp(suffix=".wav")
                        audio_np = np.asarray(audio_input, dtype=np.float32).flatten()
                        with wave.open(temp_wav_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(
                                (audio_np * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                            )
                        wav_path = temp_wav_path

                    text = _transcribe_cohere(
                        wav_path,
                        cohere_processor,
                        cohere_model,
                        screen_context=screen_context,
                    )
                    if temp_wav_path and os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
                    now = time.time()
                    last_real_inference_at[0] = now
                    last_any_inference_at[0] = now
            else:
                # Qwen3 via MLX
                with inference_lock:
                    raw = _transcribe_qwen3(
                        audio_input,
                        model_mode,
                        screen_context=screen_context,
                    )
                    qwen3_loaded_modes.add(model_mode)
                    now = time.time()
                    last_real_inference_at[0] = now
                    last_any_inference_at[0] = now

                if isinstance(raw, dict):
                    text = raw.get("text", "").strip()
                elif hasattr(raw, "text"):
                    text = raw.text.strip()
                else:
                    text = str(raw).strip()

            elapsed = time.time() - t0
            context_note = f", screen_context={len(screen_context)} chars" if screen_context else ""
            _log(f"Transcription worker: [{model_mode}] {elapsed:.1f}s{context_note}")
            result_pipe.send({"text": text, "time": elapsed, "error": None})

        except Exception as e:
            result_pipe.send({"text": "", "time": 0, "error": str(e)})

    _log("Transcription worker: exiting")
