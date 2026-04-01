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

# ── 5. Install mlx-qwen3-asr system-wide ────────────────────────────────────
step "Installing mlx-qwen3-asr (Qwen3 model support)"
if "$PYTHON_BIN" -c "import mlx_qwen3_asr" 2>/dev/null; then
  ok "mlx-qwen3-asr already installed"
else
  "$PYTHON_BIN" -m pip install --break-system-packages --quiet mlx-qwen3-asr
  ok "mlx-qwen3-asr installed"
fi

# ── 6. HuggingFace login (Cohere model is gated) ────────────────────────────
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

# ── 7. Update run.sh with correct paths ─────────────────────────────────────
step "Configuring run.sh"
cat > "$REPO_DIR/run.sh" << EOF
#!/bin/bash
# Launcher — uses Python.app (which has Accessibility + Input Monitoring permission)
PYTHON_APP="${PYTHON_APP}"
VENV_SITE="${VENV_SITE}"
SCRIPT="${REPO_DIR}/transcribe.py"

# Kill any existing voice-transcribe processes
pkill -f "voice-transcribe/transcribe.py" 2>/dev/null
sleep 0.5

export PYTHONPATH="\${VENV_SITE}:${REPO_DIR}:\${PYTHONPATH}"
exec "\$PYTHON_APP" "\$SCRIPT"
EOF
chmod +x "$REPO_DIR/run.sh"
ok "run.sh configured for this machine"

# ── 8. macOS permissions reminder ───────────────────────────────────────────
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

# ── 9. Done ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}✓ Installation complete!${NC}"
echo ""
echo "To launch:"
echo "  ${REPO_DIR}/run.sh"
echo ""
echo "To run in background:"
echo "  nohup ${REPO_DIR}/run.sh > /tmp/voice-transcribe.log 2>&1 &"
echo ""
echo "Usage:"
echo "  Hold Fn          → record with Cohere 2B (most accurate)"
echo "  Hold Right Opt   → record with Qwen3 1.7B (faster)"
echo "  Release          → transcribes and pastes at cursor"
echo ""
