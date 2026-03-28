# Voice Transcribe

A lightweight macOS menu bar app that transcribes speech to text using local AI. Tap the **Fn key** to start recording, tap again to stop — the transcription is automatically pasted at your cursor.

**No cloud APIs, no subscriptions, no data leaves your machine.** Runs entirely on-device using Apple Silicon Metal GPU acceleration.

## How It Works

1. **Tap Fn** → "Tink" sound plays, menu bar shows `🔴 1s`, `🔴 2s`... (recording)
2. **Speak**
3. **Tap Fn** → "Pop" sound plays, menu bar shows `⏳` (transcribing)
4. Text is pasted at your cursor via Cmd+V, menu bar returns to `🎙`

Click the 🎙 menu bar icon to see recent transcription history (last 20). Click any entry to copy it to clipboard.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| ASR Model | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via `mlx-qwen3-asr` |
| GPU Acceleration | Apple MLX framework (Metal) |
| Menu Bar | `rumps` (Ridiculously Uncomplicated macOS Python Statusbar apps) |
| Audio Recording | `sounddevice` (16kHz mono float32 PCM) |
| Key Monitoring | `pynput` (in a subprocess — see Architecture) |
| Clipboard/Paste | PyObjC (`AppKit.NSPasteboard` + `Quartz.CGEvent` Cmd+V simulation) |

## Architecture

```
┌─────────────────────┐         multiprocessing.Pipe         ┌──────────────────┐
│   Key Monitor       │ ──────────── "fn" ─────────────────▶ │   Main App       │
│   (subprocess)      │                                      │   (rumps menu    │
│                     │                                      │    bar app)      │
│  pynput.Listener    │                                      │                  │
│  detects Fn key     │                                      │  • Record audio  │
│  (keycode 63)       │                                      │  • Transcribe    │
│                     │                                      │  • Paste at      │
│  Debounces double   │                                      │    cursor        │
│  events (150ms)     │                                      │  • History mgmt  │
└─────────────────────┘                                      └──────────────────┘
```

**Why a subprocess?** `pynput` and `rumps` both use macOS Quartz/AppKit internally. When they run in the same process, their event taps conflict and the keyboard listener silently stops receiving events. Running `pynput` in a separate process with a `multiprocessing.Pipe` for IPC solves this completely.

**Why Fn key fires as "release" only?** macOS treats the Fn/Globe key as a modifier. `pynput` receives it only as `flagsChanged` events (reported as releases with keycode 63). Each physical press/release produces two release events. A 150ms debounce filters the duplicates, giving us clean toggle behavior.

**Thread-safe UI updates:** Menu bar title changes (`🎙` → `🔴` → `⏳`) are queued via `_set_title()` and applied on the main thread by a `rumps.Timer(0.1)` callback, since AppKit UI updates must happen on the main thread.

## Requirements

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Python 3.14** (Homebrew)
- **mlx-qwen3-asr** 0.3.1+ (system-wide install)

## Setup

### 1. Install dependencies

```bash
cd ~/voice-transcribe
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. macOS permissions (one-time)

**System Settings → Keyboard:**
- "Press 🌐 key to" → **"Do Nothing"**

**System Settings → Privacy & Security → Accessibility:**
- Add `Python.app` from: `/opt/homebrew/Cellar/python@3.14/3.14.1/Frameworks/Python.framework/Versions/3.14/Resources/Python.app`

**System Settings → Privacy & Security → Input Monitoring:**
- Add the same `Python.app`

**Microphone:** Will prompt automatically on first recording.

### 3. Run

```bash
~/voice-transcribe/transcribe.py
# or
nohup ~/voice-transcribe/.venv/bin/python3 ~/voice-transcribe/transcribe.py > /tmp/voice-transcribe.log 2>&1 &
```

## Files

```
~/voice-transcribe/
├── transcribe.py        # Main app (~350 lines)
├── requirements.txt     # pip dependencies
├── history.json         # Auto-created transcription log (last 100 entries)
├── .venv/               # Python venv (inherits system mlx-qwen3-asr)
└── README.md            # This file
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Fn key not detected | Toggle Input Monitoring OFF/ON for Python.app, restart app |
| No audio recording | Grant Microphone permission when prompted |
| Menu bar icon doesn't change | Expected on first run — the `rumps.Timer` needs the app run loop |
| Multiple menu bar icons | Kill all: `pkill -9 -f transcribe.py` then restart |
| "Loading model..." stuck | First run downloads ~600MB model from HuggingFace. Wait for it. |
| Broken pipe error in log | Normal when quitting — the key monitor subprocess shuts down |

## Logs

```bash
tail -f /tmp/voice-transcribe.log
```

## Performance

On M4 Max with Qwen3-ASR-0.6B:
- ~0.8s transcription for short clips
- ~6x real-time speed (10s audio → ~1.7s transcription)
- ~600MB model in memory after first load
