"""Background screenshot glossary extraction using local Qwen 2.5 VL on MLX."""

from __future__ import annotations

import os
import re
import signal
import time
import json
from pathlib import Path


MODEL_ID = os.getenv(
    "VOICE_TRANSCRIBE_SCREEN_VLM_MODEL",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
)
MAX_OUTPUT_TOKENS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_VLM_MAX_TOKENS", "72"))
MAX_GLOSSARY_TERMS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_VLM_MAX_TERMS", "20"))
MAX_GLOSSARY_CHARS = int(os.getenv("VOICE_TRANSCRIBE_SCREEN_VLM_MAX_CHARS", "320"))

PROMPT = (
    "Read this screenshot and extract a short glossary of exact visible terms that would help speech transcription. "
    "Prefer app names, UI labels, filenames, URLs, code symbols, and proper nouns. "
    "Return only a comma-separated list of terms. No explanation."
)


def _prepare_model_path(model_id: str) -> str:
    """Download the model snapshot locally and patch known config quirks if needed."""
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(model_id)
    preprocessor_config_path = Path(local_path) / "preprocessor_config.json"
    if preprocessor_config_path.exists():
        data = json.loads(preprocessor_config_path.read_text())
        image_processor_type = data.get("image_processor_type")
        if image_processor_type == "Qwen2VLImageProcessor":
            data["image_processor_type"] = "Qwen2_5_VLImageProcessor"
            preprocessor_config_path.write_text(json.dumps(data, indent=2) + "\n")

    return local_path


def _load_qwen_vl(model_id: str):
    """Load Qwen 2.5 VL model + processor pieces without AutoProcessor."""
    from transformers import (
        AutoTokenizer,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
        Qwen2_5_VLProcessor,
    )
    from mlx_vlm.utils import StoppingCriteria
    from mlx_vlm.utils import load_config, load_model
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    model_path = Path(_prepare_model_path(model_id))
    config = load_config(model_path, trust_remote_code=True)
    model = load_model(model_path, lazy=False)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    video_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
    processor = Qwen2_5_VLProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=hf_tokenizer,
        chat_template=hf_tokenizer.chat_template or "",
    )

    base_tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id"))

    class CallableTokenizerProxy:
        def __init__(self, hf_tokenizer, base_tokenizer, eos_token_id):
            self._hf_tokenizer = hf_tokenizer
            self._base_tokenizer = base_tokenizer
            self.detokenizer = base_tokenizer.detokenizer
            self.stopping_criteria = StoppingCriteria(eos_token_id, hf_tokenizer)

        def __call__(self, *args, **kwargs):
            return self._hf_tokenizer(*args, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            return self._hf_tokenizer.apply_chat_template(*args, **kwargs)

        def __getattr__(self, name):
            if hasattr(self._base_tokenizer, name):
                return getattr(self._base_tokenizer, name)
            return getattr(self._hf_tokenizer, name)

    processor.tokenizer = CallableTokenizerProxy(
        hf_tokenizer=hf_tokenizer,
        base_tokenizer=base_tokenizer,
        eos_token_id=config.get("eos_token_id"),
    )

    class DetokenizerShim:
        def __init__(self, detokenizer):
            self._detokenizer = detokenizer

        def reset(self):
            return self._detokenizer.reset()

        def finalize(self):
            return self._detokenizer.finalize()

        def add_token(self, token, skip_special_token_ids=None):
            return self._detokenizer.add_token(token)

        @property
        def last_segment(self):
            return self._detokenizer.last_segment

    processor.detokenizer = DetokenizerShim(base_tokenizer.detokenizer)

    return model, processor, str(model_path)


def _split_terms(raw_text: str) -> list[str]:
    text = " ".join((raw_text or "").strip().split())
    if not text:
        return []

    text = re.sub(r"^[A-Za-z ]*:\s*", "", text)
    parts = re.split(r"[,|\n]+", text)

    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = part.strip(" -•\t\r\"'")
        cleaned = re.sub(r"^(\d+\.|\-)\s*", "", cleaned)
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= MAX_GLOSSARY_TERMS:
            break

    return normalized


def _normalize_glossary(raw_text: str) -> str:
    normalized = _split_terms(raw_text)

    glossary = ", ".join(normalized)
    if len(glossary) > MAX_GLOSSARY_CHARS:
        glossary = glossary[: MAX_GLOSSARY_CHARS - 1].rstrip() + "…"
    return glossary


def run(request_pipe, result_pipe):
    """Receive screenshot paths, return a compact glossary extracted by Qwen 2.5 VL."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Screen VLM worker started (PID {os.getpid()})", flush=True)

    model = None
    processor = None

    while True:
        try:
            request = request_pipe.recv()
        except (EOFError, OSError):
            break

        if request == "__quit__":
            break

        screenshot_path = request.get("screenshot_path")
        if not screenshot_path:
            result_pipe.send({"glossary": "", "raw_text": "", "time_ms": 0, "error": "missing screenshot_path"})
            continue

        try:
            started = time.time()

            if model is None or processor is None:
                print(f"Screen VLM worker: loading {MODEL_ID}...", flush=True)
                from mlx_vlm import generate

                model, processor, model_path = _load_qwen_vl(MODEL_ID)
                print(f"Screen VLM worker: loaded {MODEL_ID} from {model_path}", flush=True)

            from mlx_vlm import generate

            formatted_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{PROMPT}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            generation = generate(
                model,
                processor,
                formatted_prompt,
                image=screenshot_path,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                verbose=False,
            )

            raw_text = (generation.text or "").strip()
            terms = _split_terms(raw_text)
            glossary = ", ".join(terms)
            if len(glossary) > MAX_GLOSSARY_CHARS:
                glossary = glossary[: MAX_GLOSSARY_CHARS - 1].rstrip() + "…"
            elapsed_ms = int((time.time() - started) * 1000)
            result_pipe.send(
                {
                    "glossary": glossary,
                    "terms": terms,
                    "raw_text": raw_text,
                    "time_ms": elapsed_ms,
                    "error": None,
                    "model": MODEL_ID,
                    "generation_tokens": getattr(generation, "generation_tokens", None),
                    "peak_memory": getattr(generation, "peak_memory", None),
                }
            )
        except Exception as exc:
            result_pipe.send({"glossary": "", "raw_text": "", "time_ms": 0, "error": str(exc), "model": MODEL_ID})

    print("Screen VLM worker: exiting", flush=True)
