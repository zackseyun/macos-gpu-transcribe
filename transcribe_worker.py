"""Transcription worker subprocess — loads model once, transcribes on demand.

Supports three model modes:
  - "fast": Qwen3-ASR 0.6B via MLX (Metal)
  - "accurate": Qwen3-ASR 1.7B via MLX (Metal)
  - "cohere": Cohere Transcribe 2B via PyTorch (MPS)
"""
import os
import signal
import time


# Metal cache limit — caps GPU buffer cache to 6GB (both models + headroom)
METAL_CACHE_LIMIT_BYTES = 6 * 1024 * 1024 * 1024

QWEN3_MODEL_IDS = {
    "fast": "Qwen/Qwen3-ASR-0.6B",
    "accurate": "Qwen/Qwen3-ASR-1.7B",
}

COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MAX_SCREEN_CONTEXT_CHARS = 320


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
                    print("Transcription worker: loading Cohere Transcribe...", flush=True)
                    import torch
                    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
                    cohere_processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
                    cohere_model = CohereAsrForConditionalGeneration.from_pretrained(
                        COHERE_MODEL_ID, torch_dtype=torch.float32,
                    ).to("mps")
                    print(f"Transcription worker: Cohere loaded on {cohere_model.device}", flush=True)

                text = _transcribe_cohere(
                    wav_path,
                    cohere_processor,
                    cohere_model,
                    screen_context=screen_context,
                )
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
