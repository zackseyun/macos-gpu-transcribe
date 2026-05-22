#!/bin/bash
# Voice Transcribe — Install Wizard
# Supports Python 3.13 and 3.14 on Apple Silicon Macs.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "\n${BOLD}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════╗"
echo "║       Voice Transcribe Installer     ║"
echo "╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Check macOS + Apple Silicon ──────────────────────────────────────────
step "Checking system"
if [[ "$(uname)" != "Darwin" ]]; then
  fail "macOS required."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  fail "Apple Silicon (M1/M2/M3/M4) required."
fi
ok "macOS on Apple Silicon"

# ── 2. Find Python (3.14 preferred, 3.13 fallback) ──────────────────────────
step "Finding Python"
PYTHON_BIN=""
PYTHON_VER=""
for ver in 3.14 3.13; do
  bin="/opt/homebrew/bin/python${ver}"
  if [[ -x "$bin" ]]; then
    PYTHON_BIN="$bin"
    PYTHON_VER="$ver"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  warn "Python 3.13/3.14 not found via Homebrew."
  echo "Installing Python 3.13..."
  brew install python@3.13
  PYTHON_BIN="/opt/homebrew/bin/python3.13"
  PYTHON_VER="3.13"
fi
ok "Using Python $("$PYTHON_BIN" --version 2>&1 | awk '{print $2}') at $PYTHON_BIN"

# Derive Python.app path from version folder in Cellar
PYTHON_CELLAR_DIR=$(ls -d /opt/homebrew/Cellar/python@${PYTHON_VER}/*/ 2>/dev/null | head -1)
PYTHON_APP="${PYTHON_CELLAR_DIR}Frameworks/Python.framework/Versions/${PYTHON_VER}/Resources/Python.app/Contents/MacOS/Python"
if [[ ! -x "$PYTHON_APP" ]]; then
  fail "Could not find Python.app at: $PYTHON_APP"
fi
ok "Python.app: $PYTHON_APP"

# ── 3. Create virtual environment ───────────────────────────────────────────
step "Setting up virtual environment"
if [[ ! -d "$REPO_DIR/.venv" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$REPO_DIR/.venv"
  ok "Created .venv"
else
  ok ".venv already exists"
fi
PIP="$REPO_DIR/.venv/bin/pip"
VENV_PYTHON="$REPO_DIR/.venv/bin/python${PYTHON_VER}"
VENV_SITE="$REPO_DIR/.venv/lib/python${PYTHON_VER}/site-packages"

# ── 4. Install pip dependencies ──────────────────────────────────────────────
step "Installing dependencies"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$REPO_DIR/requirements.txt"
ok "Base dependencies installed"

"$PIP" install --quiet transformers torch librosa accelerate
ok "ML dependencies installed (transformers, torch, librosa, accelerate)"

# ── 5. Native Swift MLX runtime for Cohere 4-bit ───────────────────────────
step "Installing native Swift MLX runtime for Cohere 4-bit"
SWIFT_STT_SERVER="$REPO_DIR/.swift-runtime/Release/mlx-audio-swift-stt-server"
if [[ -x "$SWIFT_STT_SERVER" ]]; then
  ok "Swift STT server already built at $SWIFT_STT_SERVER"
else
  if ./scripts/install_mlx_audio_swift.sh; then
    ok "Swift STT server built at $SWIFT_STT_SERVER"
  else
    warn "Swift STT build failed. The menu bar app will still install, but Cohere Swift 4-bit will not work until you rerun scripts/install_mlx_audio_swift.sh."
  fi
fi

# ── 6. CrispASR runtime for Granite Speech ─────────────────────────────────
step "Installing CrispASR runtime for Granite Speech"
CRISP_DIR="$REPO_DIR/.crispasr"
CRISP_BIN="$CRISP_DIR/build/bin/crispasr"
if [[ -x "$CRISP_BIN" ]]; then
  ok "CrispASR already built at $CRISP_BIN"
else
  if ! command -v cmake >/dev/null 2>&1; then
    warn "CMake not found; installing with Homebrew..."
    brew install cmake
  fi

  if [[ ! -d "$CRISP_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/CrispStrobe/CrispASR.git "$CRISP_DIR" || {
      warn "Could not clone CrispASR. Granite Speech will not work until CrispASR is installed."
    }
  else
    git -C "$CRISP_DIR" pull --ff-only || warn "Could not update existing CrispASR checkout; using local copy."
  fi

  if [[ -d "$CRISP_DIR/.git" ]]; then
    if cmake -S "$CRISP_DIR" -B "$CRISP_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      && cmake --build "$CRISP_DIR/build" -j"$(sysctl -n hw.ncpu)" --target crispasr-cli; then
      ok "CrispASR built at $CRISP_BIN"
    else
      warn "CrispASR build failed. Switch the menu default back to Cohere until it is fixed."
    fi
  fi
fi

# ── 7. Disable Right Option key (system-wide HID remap) ─────────────────────
step "Disabling Right Option key (HID-level remap)"
# Right Option used to be a second hotkey, but it kept leaking stray special
# characters (®, ¥, etc.) into focused fields when held. Disabling it system-wide
# at the HID layer is the cleanest fix. The LaunchAgent re-applies on every login.
mkdir -p "$HOME/Library/LaunchAgents"
cp "$REPO_DIR/com.local.DisableRightOption.plist" "$HOME/Library/LaunchAgents/"
launchctl unload "$HOME/Library/LaunchAgents/com.local.DisableRightOption.plist" 2>/dev/null || true
launchctl load -w "$HOME/Library/LaunchAgents/com.local.DisableRightOption.plist"
ok "Right Option disabled (LaunchAgent loaded)"

# ── 8. HuggingFace login (Cohere model is gated) ────────────────────────────
step "HuggingFace authentication (required for Cohere Transcribe model)"
echo "The Cohere Transcribe model is gated."
echo "You need a HuggingFace account with access granted at:"
echo "  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026"
echo ""

# Check if already logged in
if "$VENV_PYTHON" -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
  ok "Already logged in to HuggingFace"
else
  read -rp "  Paste your HuggingFace token (hf_...): " HF_TOKEN
  if [[ -z "$HF_TOKEN" ]]; then
    warn "No token provided — skipping. Cohere model won't work until you log in."
    warn "Run: $VENV_PYTHON -c \"from huggingface_hub import login; login()\""
  else
    "$VENV_PYTHON" -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    ok "Logged in to HuggingFace"
  fi
fi

# ── 9. Update run.sh with correct paths ─────────────────────────────────────
step "Configuring run.sh"
cat > "$REPO_DIR/run.sh" << EOF
#!/bin/bash
# Launcher — uses Python.app (which has Accessibility + Input Monitoring permission).
# Do not pkill an existing instance here: transcribe.py has a lock, and killing
# the warm resident worker is exactly what makes the first Cohere dictation slow.
set -euo pipefail

REPO_DIR="${REPO_DIR}"
PYTHON_BIN="\${VOICE_TRANSCRIBE_PYTHON:-${PYTHON_APP}}"
VENV_SITE="${VENV_SITE}"
SCRIPT="\${REPO_DIR}/transcribe.py"

cd "\${REPO_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="\${VENV_SITE}:\${REPO_DIR}:\${PYTHONPATH:-}"
exec "\${PYTHON_BIN}" "\${SCRIPT}"
EOF
chmod +x "$REPO_DIR/run.sh"
ok "run.sh configured for this machine"

# ── 10. Set default model for this machine ──────────────────────────────────
step "Configuring local app settings"
"$VENV_PYTHON" - <<PY
import json
from pathlib import Path

p = Path("$REPO_DIR/settings.json")
data = {}
if p.exists():
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = {}
data.setdefault("screen_context_enabled", False)
data.setdefault("sound_effects_enabled", True)
data.setdefault("vocabulary", "")
data["default_model_mode"] = "cohere-swift-4bit"
p.write_text(json.dumps(data, indent=2) + "\\n")
PY
ok "Fn default set to Cohere Swift 4-bit in settings.json"

# ── 11. macOS permissions reminder ──────────────────────────────────────────
step "macOS permissions required (one-time)"
echo ""
echo -e "${BOLD}You must grant these permissions manually:${NC}"
echo ""
echo "  1. System Settings → Keyboard"
echo "     → Set \"Press 🌐 key to\" = Do Nothing"
echo ""
echo "  2. System Settings → Privacy & Security → Accessibility"
echo "     → Click + and add:"
echo "     ${PYTHON_APP}"
echo ""
echo "  3. System Settings → Privacy & Security → Input Monitoring"
echo "     → Click + and add the same Python.app"
echo ""
read -rp "Press Enter once you've granted those permissions..."

# ── 12. Install menu bar app LaunchAgent ────────────────────────────────────
step "Installing Voice Transcribe menu bar app"
VOICE_AGENT="$HOME/Library/LaunchAgents/com.zack.voice-transcribe.plist"
cat > "$VOICE_AGENT" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zack.voice-transcribe</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${REPO_DIR}/run.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>/tmp/voice-transcribe.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/voice-transcribe.log</string>
</dict>
</plist>
EOF
launchctl bootout "gui/$(id -u)" "$VOICE_AGENT" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$VOICE_AGENT" 2>/dev/null || launchctl load -w "$VOICE_AGENT"
launchctl kickstart -k "gui/$(id -u)/com.zack.voice-transcribe" 2>/dev/null || true
ok "Voice Transcribe installed as a login menu bar app"

# ── 13. Done ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}✓ Installation complete!${NC}"
echo ""
echo "The menu bar app is installed and should start automatically at login."
echo ""
echo "To restart it manually:"
echo "  launchctl kickstart -k gui/\$(id -u)/com.zack.voice-transcribe"
echo ""
echo "To launch directly:"
echo "  ${REPO_DIR}/run.sh"
echo ""
echo "Usage:"
echo "  Hold Fn  → record"
echo "  Release  → transcribes and pastes at cursor"
echo ""
