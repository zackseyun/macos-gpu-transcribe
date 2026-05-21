#!/usr/bin/env python3
"""Create a local 4-bit MLX Qwen3-ASR checkpoint for the fast dictation path."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils

from mlx_qwen3_asr.load_models import load_model, _resolve_path


COPY_SUFFIXES = {".json", ".txt", ".model"}


def copy_model_metadata(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.suffix in COPY_SUFFIXES:
            shutil.copy2(item, output / item.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output", default="models/qwen3-asr-0.6b-4bit")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    source = _resolve_path(args.model)

    print(f"Loading {args.model} from {source}")
    model, _config = load_model(args.model, dtype=mx.float16)

    print(f"Quantizing to {args.bits}-bit, group_size={args.group_size}")
    nn.quantize(model, bits=args.bits, group_size=args.group_size)
    mx.eval(model.parameters())

    print(f"Writing quantized checkpoint to {output}")
    copy_model_metadata(source, output)
    weights = dict(mlx_utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    (output / "quantization_config.json").write_text(
        json.dumps(
            {
                "bits": args.bits,
                "group_size": args.group_size,
                "source_model": args.model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
