#!/usr/bin/env bash
# Install the pinned upstream ARM64 Swift MLX runtime, including its Metal library.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=v0.0.27
SHA256=2c2474bdd36d69fb07fa505b08b8798af58b61b572f754cea31c2e6b118e8a2d
[[ "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon is required.' >&2; exit 1; }
mkdir -p "$REPO_DIR/.swift-runtime"
STAGING="$(mktemp -d "$REPO_DIR/.swift-runtime/install.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
if [[ -n "${1:-}" ]]; then
  # A previously downloaded archive still passes the same pinned hash check.
  cp "$1" "$STAGING/runtime.tar.gz"
else
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
    "https://github.com/soniqo/speech-swift/releases/download/$VERSION/speech-macos-arm64.tar.gz" \
    -o "$STAGING/runtime.tar.gz"
fi
echo "$SHA256  $STAGING/runtime.tar.gz" | shasum -a 256 -c -
mkdir "$STAGING/runtime"
tar -xzf "$STAGING/runtime.tar.gz" -C "$STAGING/runtime"
"$STAGING/runtime/speech-server" --help >/dev/null
DEST="$REPO_DIR/.swift-runtime/speech-$VERSION"
mkdir -p "$DEST"
cp -R "$STAGING/runtime/." "$DEST/"
echo "Installed Swift MLX Qwen runtime: $DEST"
echo 'Restart Voice Transcribe. The first warmup downloads the Qwen 0.6B INT4 model.'
