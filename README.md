# Voice Transcribe

A lightweight macOS menu bar app that transcribes speech to text using local AI models. **Hold a key** to record, **release** to transcribe — the text is automatically pasted at your cursor.

**No cloud APIs, no subscriptions, no data leaves your machine.** Runs entirely on-device using Apple Silicon GPU acceleration.

## How It Works

1. **Hold Fn** (or **Right Option**) → "Tink" sound plays, menu bar shows `🔴 1s`, `🔴 2s`... (recording)
2. **Speak**
3. **Release** → "Pop" sound plays, menu bar shows `⏳` (transcribing)
4. Text is pasted at your cursor via Cmd+V, menu bar returns to `🎙`

Click the 🎙 menu bar icon to see recent transcription history (last 20). Click any entry to copy it to clipboard.

## Models & Key Bindings

| Key | Model | Parameters | Framework | Throughput |
|-----|-------|-----------|-----------|------------|
| **Hold Fn** | [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 2B | PyTorch (MPS) | ~3-4x real-time |
| **Hold Right Option** | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | MLX (Metal) | ~5-6x real-time |

**Cohere Transcribe** is the default (Fn key) because it produces the most accurate transcriptions for English dictation, with better punctuation, capitalization, and fewer hallucinations on short clips.

### Model Comparison

All benchmarks on M4 Max, 16kHz mono audio, measured as **throughput** — the ratio of audio duration to processing time. A throughput of 6x means 10 seconds of audio is transcribed in ~1.7 seconds.

| Model | Params | Framework | Throughput | English Accuracy | Punctuation | Hallucination on Silence | Best For |
|-------|--------|-----------|------------|-----------------|-------------|------------------------|----------|
| Cohere Transcribe 2B | 2B | PyTorch MPS | ~3-4x | Highest | Excellent (natural caps + periods) | Rare | Daily dictation, long-form notes |
| Qwen3-ASR 1.7B | 1.7B | MLX Metal | ~5-6x | Very good | Good (sometimes misses periods) | Occasional | Multilingual, when speed matters |
| Qwen3-ASR 0.6B | 0.6B | MLX Metal | ~8-10x | Good | Basic | More frequent | Ultra-fast drafts (removed from key bindings) |

**Throughput explained:** Throughput is the key metric for real-time transcription. It measures how many seconds of audio the model can process per second of wall-clock time. For a hold-to-talk app, you want at minimum 1x (real-time) so transcription finishes before the user notices. At 3-4x (Cohere), a 30-second recording takes ~8 seconds to transcribe. At 8-10x (Qwen3 0.6B), the same recording takes ~3 seconds. Higher throughput = shorter wait after releasing the key.

**Why Cohere is the default despite lower throughput:** The accuracy difference is significant for dictation. Cohere handles homophones, technical terms, and sentence boundaries more reliably. The extra 1-2 seconds of processing time is worth it when the result doesn't need manual corrections. Qwen3 1.7B is kept on Right Option as an alternative for multilingual use or when faster turnaround is preferred.

### Post-Processing

Raw ASR output goes through `format_text.py` which normalizes:
- Number words → digits ("twenty five" → "25", "three hundred K" → "300K")
- Multipliers ("3 thousand" → "3,000", "5 million" → "5,000,000")
- Percentages ("25 percent" → "25%")
- Currency ("500 dollars" → "$500")
- First-letter capitalization

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13+ (Homebrew) |
| Primary ASR | [Cohere Transcribe 2B](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) via PyTorch (MPS) |
| Secondary ASR | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) via `mlx-qwen3-asr` (Metal) |
| Menu Bar | `rumps` (Ridiculously Uncomplicated macOS Python Statusbar apps) |
| Audio Recording | `sounddevice` (16kHz mono float32 PCM) |
| Key Monitoring | Quartz CGEvent tap (in a subprocess — see Architecture) |
| Clipboard/Paste | PyObjC (`AppKit.NSPasteboard` + `Quartz.CGEvent` Cmd+V simulation) |

## Architecture

```
┌─────────────────────┐      multiprocessing.Pipe       ┌──────────────────┐
│   Key Monitor       │ ─── "down:cohere" / "up" ─────▶ │   Main App       │
│   (subprocess)      │                                  │   (rumps menu    │
│                     │                                  │    bar app)      │
│  Quartz CGEvent tap │                                  │                  │
│  detects Fn / Right │      multiprocessing.Pipe        │  • Record audio  │
│  Option hold/release│                                  │  • Dispatch to   │
└─────────────────────┘                                  │    worker        │
                                                         └────────┬─────────┘
                                                                  │
                                                    multiprocessing.Pipe
                                                                  │
                                                         ┌────────▼─────────┐
                                                         │  Transcription   │
                                                         │  Worker          │
                                                         │  (subprocess)    │
                                                         │                  │
                                                         │  Cohere 2B (MPS) │
                                                         │  Qwen3 1.7B (MLX)│
                                                         │                  │
                                                         │  Models loaded   │
                                                         │  lazily, stay in │
                                                         │  GPU memory      │
                                                         └──────────────────┘
```

**Three processes, three reasons:**
1. **Key Monitor** — Quartz CGEvent tap and `rumps` both use macOS AppKit internally. Running them in the same process causes the event tap to silently stop receiving events. Separate process + pipe solves this.
2. **Transcription Worker** — Holds ML models in GPU memory permanently. Isolates model loading crashes and memory leaks from the UI process. Metal cache is capped at 6GB.
3. **Main App** — `rumps` menu bar UI + audio recording via `sounddevice`. Audio stream is opened once at startup and never stopped (see CoreAudio HALB_Mutex deadlock note in source).

**Hold-to-record (not toggle):** The key monitor uses Quartz low-level event taps to detect actual key press/release state. `down:cohere` or `down:accurate` fires on key-down, `up` fires on key-up. This gives natural push-to-talk behavior. macOS can disable event taps after sleep/wake — the monitor auto-recovers and a heartbeat watchdog in the main app restarts it if needed.

## Requirements

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Python 3.13 or 3.14** (Homebrew)
- **mlx-qwen3-asr** 0.3.1+ (system-wide install, for Qwen3 models)
- **transformers** 5.0+, **torch** (MPS), **librosa**, **accelerate** (for Cohere model, installed in venv)
- **HuggingFace account** with access to [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)

## Setup

### Automated (recommended)

```bash
git clone https://github.com/zackseyun/voice-transcribe
cd voice-transcribe
./install.sh
```

The installer will:
1. Detect or install Python 3.13/3.14 via Homebrew
2. Create a `.venv` with all dependencies
3. Install `mlx-qwen3-asr` system-wide for Qwen3 support
4. Prompt for your HuggingFace token to authenticate the Cohere model
5. Auto-configure `run.sh` for your machine
6. Walk you through the required macOS permissions

### Manual

```bash
# 1. Create venv
python3.13 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install transformers torch librosa accelerate

# 3. Install Qwen3 support system-wide
pip3.13 install --break-system-packages mlx-qwen3-asr

# 4. Authenticate HuggingFace (Cohere model is gated)
python -c "from huggingface_hub import login; login()"

# 5. Update run.sh paths for your machine (see run.sh comments)
```

### macOS permissions (one-time)

**System Settings → Keyboard:**
- "Press 🌐 key to" → **"Do Nothing"**

**System Settings → Privacy & Security → Accessibility:**
- Add `Python.app` — path printed by `install.sh`, or find it at:
  `/opt/homebrew/Cellar/python@3.13/<version>/Frameworks/Python.framework/Versions/3.13/Resources/Python.app`

**System Settings → Privacy & Security → Input Monitoring:**
- Add the same `Python.app`

**Microphone:** Will prompt automatically on first recording.

### Run

```bash
~/Desktop/voice-transcribe/run.sh
# or in background:
nohup ~/Desktop/voice-transcribe/run.sh > /tmp/voice-transcribe.log 2>&1 &
```

## Files

```
voice-transcribe/
├── transcribe.py          # Main app — menu bar UI, audio, paste logic
├── transcribe_worker.py   # Worker subprocess — model loading & inference
├── key_monitor.py         # Key monitor subprocess — Quartz CGEvent tap
├── format_text.py         # Post-processing — numbers, currency, percentages
├── install.sh             # Install wizard — sets up everything automatically
├── run.sh                 # Launcher (auto-configured by install.sh)
├── requirements.txt       # pip dependencies
├── history.json           # Auto-created transcription log (last 100 entries)
├── .venv/                 # Python venv (inherits system mlx-qwen3-asr)
└── README.md              # This file
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Fn key not detected | Toggle Input Monitoring OFF/ON for Python.app, restart app |
| No audio recording | Grant Microphone permission when prompted |
| Menu bar icon doesn't change | Expected on first run — the `rumps.Timer` needs the app run loop |
| Multiple menu bar icons | Kill all: `pkill -9 -f transcribe.py` then restart |
| Cohere model: 401 Unauthorized | Request access at huggingface.co/CohereLabs/cohere-transcribe-03-2026, then re-run `install.sh` |
| "Loading model..." takes a while | First run downloads models from HuggingFace (~2GB for Cohere, ~1.2GB for Qwen3 1.7B). Subsequent launches are instant. |
| Broken pipe error in log | Normal when quitting — the key monitor subprocess shuts down |
| Cohere model slow on first use | PyTorch MPS compilation is cached after first inference. Second transcription onward is faster. |

## Logs

```bash
tail -f /tmp/voice-transcribe.log
```
