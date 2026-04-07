"""Transcription worker subprocess — loads model once, transcribes on demand.

Supports three model modes:
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
import os
import signal
import threading
import time
import wave


# Metal cache limit — caps GPU buffer cache to 6GB (both models + headroom)
METAL_CACHE_LIMIT_BYTES = 6 * 1024 * 1024 * 1024

QWEN3_MODEL_IDS = {
    "fast": "Qwen/Qwen3-ASR-0.6B",
    "accurate": "Qwen/Qwen3-ASR-1.7B",
}

COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MAX_SCREEN_CONTEXT_CHARS = 320

# Keep-warm cadence — must be short enough that MPS shader cache stays resident.
# macOS evicts Metal state after ~30-60s of GPU idle, so 45s is a safe ceiling.
KEEP_WARM_INTERVAL = float(os.getenv("VOICE_TRANSCRIBE_KEEP_WARM_INTERVAL", "45"))
KEEP_WARM_AUDIO_SECONDS = 0.5  # 0.5s of silence = trivially fast dummy inference


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
        outputs = model.generate(**inputs, max_new_tokens=512)

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
    last_inference_at = [time.time()]  # list = mutable for closure

    def _keep_warm_loop():
        """Background thread: run a silent dummy inference periodically to keep MPS hot."""
        while True:
            try:
                time.sleep(5)
                if cohere_model is None or warm_wav_path is None:
                    continue
                idle = time.time() - last_inference_at[0]
                if idle < KEEP_WARM_INTERVAL:
                    continue
                # Try to grab the lock without blocking real requests.
                if not inference_lock.acquire(blocking=False):
                    continue
                try:
                    t0 = time.time()
                    _transcribe_cohere(
                        warm_wav_path, cohere_processor, cohere_model, screen_context=""
                    )
                    last_inference_at[0] = time.time()
                    # Only log if it was slow (indicates we caught a cold-start)
                    dt = time.time() - t0
                    if dt > 1.5:
                        print(
                            f"Transcription worker: keep-warm took {dt:.1f}s "
                            f"(MPS was cold)",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"Transcription worker: keep-warm failed: {exc}", flush=True)
                finally:
                    inference_lock.release()
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

                with inference_lock:
                    text = _transcribe_cohere(
                        wav_path,
                        cohere_processor,
                        cohere_model,
                        screen_context=screen_context,
                    )
                    last_inference_at[0] = time.time()
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
