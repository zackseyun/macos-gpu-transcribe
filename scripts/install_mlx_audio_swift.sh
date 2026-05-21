#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${VOICE_TRANSCRIBE_SWIFT_RUNTIME_DIR:-$REPO_DIR/.swift-runtime}"
SRC_DIR="$RUNTIME_DIR/mlx-audio-swift"
DERIVED_DATA="$RUNTIME_DIR/DerivedData"
RELEASE_DIR="$RUNTIME_DIR/Release"
MLX_AUDIO_SWIFT_REPO="${MLX_AUDIO_SWIFT_REPO:-https://github.com/Blaizzy/mlx-audio-swift.git}"
MLX_AUDIO_SWIFT_REF="${MLX_AUDIO_SWIFT_REF:-main}"

# Prefer stable Xcode when both stable and beta are installed; the beta on this
# machine may not have the Metal toolchain component installed by default.
if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

mkdir -p "$RUNTIME_DIR"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone "$MLX_AUDIO_SWIFT_REPO" "$SRC_DIR"
fi

git -C "$SRC_DIR" fetch --all --tags --prune
git -C "$SRC_DIR" checkout "$MLX_AUDIO_SWIFT_REF"

SERVER_SOURCE="$REPO_DIR/scripts/swift/mlx-audio-swift-stt-server-App.swift"
if [[ -f "$SERVER_SOURCE" ]]; then
  mkdir -p "$SRC_DIR/Sources/Tools/mlx-audio-swift-stt-server"
  cp "$SERVER_SOURCE" "$SRC_DIR/Sources/Tools/mlx-audio-swift-stt-server/App.swift"
  python3 - "$SRC_DIR/Package.swift" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

product = '''        .executable(
            name: "mlx-audio-swift-stt",
            targets: ["mlx-audio-swift-stt"],
        ),'''
if 'name: "mlx-audio-swift-stt-server"' not in s:
    s = s.replace(product, product + '''
        .executable(
            name: "mlx-audio-swift-stt-server",
            targets: ["mlx-audio-swift-stt-server"],
        ),''')

target = '''        .executableTarget(
            name: "mlx-audio-swift-stt",
            dependencies: ["MLXAudioCore", "MLXAudioSTT"],
            path: "Sources/Tools/mlx-audio-swift-stt",
            exclude: [
                "README.md",
            ]
        ),'''
if 'path: "Sources/Tools/mlx-audio-swift-stt-server"' not in s:
    s = s.replace(target, target + '''
        .executableTarget(
            name: "mlx-audio-swift-stt-server",
            dependencies: ["MLXAudioCore", "MLXAudioSTT"],
            path: "Sources/Tools/mlx-audio-swift-stt-server"
        ),''')

p.write_text(s)
PY
fi

# Xcode 26 can install the Metal toolchain as an on-demand asset. If it is
# already installed this returns quickly; if not, it prevents default.metallib
# build failures.
xcodebuild -downloadComponent MetalToolchain >/tmp/voice-transcribe-metal-toolchain.log 2>&1 || true

for scheme in mlx-audio-swift-stt mlx-audio-swift-stt-server; do
  xcodebuild build \
    -scheme "$scheme" \
    -destination 'platform=macOS' \
    -configuration Release \
    -derivedDataPath "$DERIVED_DATA"
done

PRODUCT_DIR="$DERIVED_DATA/Build/Products/Release"
mkdir -p "$RELEASE_DIR"
cp "$PRODUCT_DIR/mlx-audio-swift-stt" "$RELEASE_DIR/"
cp "$PRODUCT_DIR/mlx-audio-swift-stt-server" "$RELEASE_DIR/"
find "$PRODUCT_DIR" -maxdepth 1 -type d -name '*.bundle' -exec cp -R {} "$RELEASE_DIR/" \;

cat <<EOF
Installed mlx-audio-swift STT tools:
  $RELEASE_DIR/mlx-audio-swift-stt
  $RELEASE_DIR/mlx-audio-swift-stt-server

Use with:
  VOICE_TRANSCRIBE_SWIFT_STT_BIN='$RELEASE_DIR/mlx-audio-swift-stt'
  VOICE_TRANSCRIBE_SWIFT_STT_SERVER_BIN='$RELEASE_DIR/mlx-audio-swift-stt-server'
EOF
