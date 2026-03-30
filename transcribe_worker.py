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


def _transcribe_cohere(wav_path, processor, model):
    """Transcribe using Cohere via PyTorch MPS."""
    import torch
    from transformers.audio_utils import load_audio

    audio = load_audio(wav_path, sampling_rate=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

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
        else:
            wav_path = request["wav_path"]
            model_mode = request.get("model_mode", "fast")

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
                        COHERE_MODEL_ID, torch_dtype=torch.float32, device_map="auto",
                    )
                    print(f"Transcription worker: Cohere loaded on {cohere_model.device}", flush=True)

                text = _transcribe_cohere(wav_path, cohere_processor, cohere_model)
            else:
                # Qwen3 via MLX
                model_id = QWEN3_MODEL_IDS.get(model_mode, QWEN3_MODEL_IDS["fast"])
                if qwen3_transcribe_fn is None:
                    print("Transcription worker: loading Qwen3 transcribe function...", flush=True)
                    from mlx_qwen3_asr import transcribe as _transcribe_fn
                    qwen3_transcribe_fn = _transcribe_fn
                    print("Transcription worker: Qwen3 ready", flush=True)

                raw = qwen3_transcribe_fn(wav_path, model=model_id)

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
            print(f"Transcription worker: [{model_mode}] {elapsed:.1f}s", flush=True)
            result_pipe.send({"text": text, "time": elapsed, "error": None})

        except Exception as e:
            result_pipe.send({"text": "", "time": 0, "error": str(e)})

    print("Transcription worker: exiting", flush=True)
