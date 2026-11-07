#!/bin/bash
# Optional isolated MLX Audio runtime; never upgrades the GUI's Python packages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV=ecd43b9f5af55982f4e1bd810d7b60cbc3dfae23
mkdir -p "$ROOT/.mlx-next"
if [[ ! -d "$ROOT/.mlx-next/mlx-audio/.git" ]]; then
  git clone https://github.com/Blaizzy/mlx-audio.git "$ROOT/.mlx-next/mlx-audio"
fi
git -C "$ROOT/.mlx-next/mlx-audio" fetch origin "$REV"
git -C "$ROOT/.mlx-next/mlx-audio" checkout --detach "$REV"
python3.14 -m venv --system-site-packages "$ROOT/.mlx-next/venv14"
PY="$ROOT/.mlx-next/venv14/bin/python"
"$PY" -m pip install setuptools wheel
"$PY" -m pip install --no-build-isolation -r "$ROOT/requirements-mlx-next.txt" -e "$ROOT/.mlx-next/mlx-audio[stt]"
"$PY" "$ROOT/scripts/download_mlx_next.py"
printf '\nInstalled. Select Granite Speech 5.0 or Nemotron 3.5 in the app menu.\n'
