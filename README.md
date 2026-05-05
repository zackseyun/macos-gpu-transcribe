# macOS GPU Transcribe

A push-to-talk dictation app for Apple Silicon Macs. **Hold Fn** to record, **release** to transcribe — the text is pasted at your cursor. All transcription runs locally on the GPU via Metal / MLX / PyTorch-MPS. No cloud round-trips, no API keys at inference time, no speech leaves your machine.

![demo placeholder — hold Fn, see floating HUD near cursor, get text at cursor]

## Why this exists

Cloud dictation (Siri, Whisper API, Otter, etc.) has latency, privacy, and cost downsides. Apple Silicon is fast enough to run 2B-parameter ASR models in real time entirely on-device. This app wires that into a hold-to-talk hotkey with native Mac UI — menu bar icon, Dock icon, a cursor-tracking HUD, and a main settings window.

## How it works

1. **Hold Fn** anywhere in macOS.
2. A floating waveform HUD appears near your cursor showing the mic is live.
3. **Speak**.
4. **Release** the key. The HUD switches to `Transcribing…` (or `Loading model…` on a cold start).
5. The transcription is pasted at your cursor via ⌘V.

Everything runs on the GPU. The current default is Granite Speech 4.1 NAR so it is easy to compare against Cohere; switch back from the menu bar if Granite is not ready on your machine yet. Granite is lazy at launch, then stays resident in a local CrispASR server after the first warm/load so later dictations do not reload the 3GB GGUF.

## Interface

- **Floating HUD** — borderless, click-through, follows the cursor. Live waveform while recording; text label while transcribing or loading. Appears only during active use.
- **Menu bar icon** — classic rumps-style status indicator (`🎙` idle / `🔴 2s` recording / `⏳` processing). Clicking it gives you a history list (last 20, click to copy), toggles for sound effects / screen context, and a default-model selector.
- **Main window** — regular Mac window with current status, hotkey reminders, settings toggles, and a table of recent transcriptions. Opens automatically at launch; reopens when you click the Dock icon.
- **Keep-warm** — wake-from-sleep hook + App Nap opt-out + 15-min background ping means the GPU kernels stay hot, so the first press after opening your laptop is instant instead of a 7-second cold start.

## Model & key binding

| Key | Model | Parameters | Framework | Throughput (M4 Max) |
|-----|-------|-----------|-----------|---------------------|
| **Hold Fn** | Default selected in menu: [Granite Speech 4.1 NAR](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) or [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 2B | Granite via CrispASR/GGUF (Metal); Cohere via PyTorch (MPS) | TBD for Granite; Cohere ~3–4× real-time |

Granite Speech 4.1 NAR is the default for now so it can be tested quickly. The original Hugging Face PyTorch path requires CUDA + `flash_attention_2`, so this Mac app runs Granite through CrispASR's GGUF runtime instead. Granite resolves the model lazily on first Granite dictation, then keeps it loaded in a persistent local server. Cohere remains one click away in the menu and is also used as an automatic fallback if Granite returns low-content punctuation.

**Right Option is disabled** at the HID layer by a LaunchAgent that `install.sh` deploys (see [`com.local.DisableRightOption.plist`](com.local.DisableRightOption.plist)). It used to be a second hotkey, but it kept emitting stray special characters (®, ¥, etc.) into focused fields. Disabling it system-wide is the simplest fix.

### Throughput explained

Throughput = audio-duration ÷ wall-clock-transcribe-time. At 4× real-time, a 30-second clip transcribes in ~7.5s. The silence-RMS gate drops audio below the mic noise floor before ASR runs, which prevents the Whisper-family "Thank you." hallucination on near-silent clips.

### Post-processing

Raw ASR output runs through `format_text.py`, which normalizes:

- Number words → digits (`twenty five` → `25`, `three hundred K` → `300K`)
- Multipliers (`3 thousand` → `3,000`, `5 million` → `5,000,000`)
- Percentages (`25 percent` → `25%`)
- Currency (`500 dollars` → `$500`)
- First-letter capitalization

### Screen context (optional, off by default)

Opt-in feature that prefetches a screenshot of your frontmost window, runs local Vision OCR, and injects the resulting glossary into the ASR prompt. Helps with proper nouns and jargon. It proved too noisy for general dictation, so it ships disabled; toggle it from the menu bar or main window if you want to try it.

## Architecture

```
┌─────────────────────┐      multiprocessing.Pipe       ┌──────────────────┐
│   Key Monitor       │ ─── "down:default" / "up" ────▶ │   Main App       │
│   (subprocess)      │                                  │   (rumps menu    │
│                     │                                  │    bar app)      │
│  Quartz CGEvent tap │                                  │                  │
│  detects Fn hold/   │      multiprocessing.Pipe        │  • Record audio  │
│  release            │                                  │  • HUD + window  │
└─────────────────────┘                                  │  • Dispatch to   │
                                                         │    worker        │
                                                         └────────┬─────────┘
                                                                  │
                                                    multiprocessing.Pipe
                                                                  │
                                                         ┌────────▼─────────┐
                                                         │  Transcription   │
                                                         │  Worker          │
                                                         │  (subprocess)    │
                                                         │                  │
                                                         │  Granite/Cohere  │
                                                         │                  │
                                                         │  Model stays     │
                                                         │  resident in GPU │
                                                         │  memory          │
                                                         └──────────────────┘
```

**Three processes, three reasons:**

1. **Key Monitor** — Quartz CGEvent tap and `rumps` both use AppKit internally. Running them in the same process causes the tap to silently stop receiving events. Separate process + pipe solves this.
2. **Transcription Worker** — Holds the ML models in GPU memory permanently. Isolates model-loading crashes, compile warm-up, and Metal buffer leaks from the UI process. Auto-restarts after 50 transcriptions or 4 GB active memory to cap leak accumulation.
3. **Main App** — rumps menu bar UI, AppKit HUD + main window, audio recording via `sounddevice`. The audio stream opens once at startup and is never stopped — CoreAudio's `HALB_Mutex` deadlocks if you call `Pa_StopStream` while the callback is active. Recording is controlled by a boolean flag checked inside the callback.

**Hold-to-talk, not toggle.** The key monitor uses a low-level Quartz event tap for press/release fidelity — not a global hotkey. macOS disables event taps after sleep/wake; the monitor auto-recovers, and a heartbeat watchdog restarts it if it dies completely.

**Staying warm.** The worker pre-warms MLX/MPS on Fn key-down (so releasing finds a hot model), on a `NSWorkspaceDidWakeNotification` observer (so opening the lid re-warms the GPU kernels before your first press), and on a 15-minute periodic ping. The main process also calls `NSProcessInfo.beginActivityWithOptions_reason_` to opt out of App Nap so the worker isn't paged out during idle stretches.

## Tech stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13+ (Homebrew) |
| ASR | Granite Speech 4.1 NAR via CrispASR/GGUF; Cohere Transcribe 2B via PyTorch (MPS) |
| Menu bar | `rumps` |
| HUD + main window | AppKit via PyObjC (NSWindow, NSBezierPath, NSTimer) |
| Audio capture | `sounddevice` (16 kHz mono float32 PCM) |
| Key monitoring | Quartz CGEvent tap, subprocess-isolated |
| Clipboard/paste | `AppKit.NSPasteboard` + Quartz ⌘V event simulation |
| Screen OCR (optional) | Frontmost-window screenshot + `pyobjc-framework-Vision` text extraction |

## Requirements

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Python 3.13 or 3.14** (Homebrew)
- **CMake + Xcode command line tools** for the Granite/CrispASR runtime
- **HuggingFace account** with access to [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) if you switch back to Cohere

## Setup

### Automated

```bash
git clone https://github.com/zackseyun/macos-gpu-transcribe.git
cd macos-gpu-transcribe
./install.sh
```

The installer will:

1. Detect or install Python 3.13/3.14 via Homebrew.
2. Create a `.venv` with all dependencies.
3. Deploy `com.local.DisableRightOption.plist` as a user LaunchAgent that disables the Right Option key system-wide via `hidutil` (see [Right Option, why disabled](#model--key-binding)).
4. Build CrispASR locally for the Granite Speech runtime.
5. Prompt for your HuggingFace token (Cohere model is gated).
6. Auto-configure `run.sh` for your machine.
7. Walk you through the required macOS permissions.

### Manual

```bash
# 1. Create venv
python3.13 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install transformers torch librosa accelerate

# 3. Build CrispASR for Granite Speech
git clone --depth 1 https://github.com/CrispStrobe/CrispASR.git .crispasr
cmake -S .crispasr -B .crispasr/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build .crispasr/build -j"$(sysctl -n hw.ncpu)" --target crispasr-cli

# 4. Disable Right Option key system-wide (deploys LaunchAgent)
mkdir -p ~/Library/LaunchAgents
cp com.local.DisableRightOption.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.local.DisableRightOption.plist

# 5. Authenticate HuggingFace
python -c "from huggingface_hub import login; login()"

# 6. Update run.sh paths for your machine (see run.sh comments)
```

### macOS permissions (one-time)

- **System Settings → Keyboard → "Press 🌐 key to"**: set to **Do Nothing** (otherwise Fn triggers the globe menu)
- **Privacy & Security → Accessibility**: add the `Python.app` printed by `install.sh`
- **Privacy & Security → Input Monitoring**: add the same `Python.app`
- **Microphone**: prompts automatically on first recording
- **Screen Recording** (only needed if you turn on screen context): add the same `Python.app`

### Run

```bash
./run.sh
# or directly:
./.venv/bin/python3 ./transcribe.py
# background with logs:
nohup ./run.sh > /tmp/voice-transcribe.log 2>&1 &
```

### Environment variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `VOICE_TRANSCRIBE_SILENCE_RMS` | `0.008` | Silence gate threshold (raise if you get false transcriptions of ambient noise) |
| `VOICE_TRANSCRIBE_TIMEOUT_SECONDS` | `900` | Max wait for a transcription; long enough for first Granite model download |
| `VOICE_TRANSCRIBE_WARM_PING_SECONDS` | `900` | How often to ping warmable resident models |
| `VOICE_TRANSCRIBE_GRANITE_MODEL` | `auto` | CrispASR model argument for Granite; set to a `.gguf` path to avoid auto-download |
| `VOICE_TRANSCRIBE_GRANITE_LANGUAGE` | `en` | Spoken-language hint for Granite; avoids a separate language-detection model download |
| `VOICE_TRANSCRIBE_CRISPASR_BIN` | `.crispasr/build/bin/crispasr` | Override CrispASR binary path |
| `VOICE_TRANSCRIBE_GRANITE_USE_SERVER` | `1` | Keep Granite loaded in a persistent local CrispASR server after first use. Set to `0` to force the older one-shot CLI path |
| `VOICE_TRANSCRIBE_GRANITE_SERVER_PORT` | `8765` | Localhost port for the lazy Granite server |
| `VOICE_TRANSCRIBE_SCREEN_CONTEXT` | unset | Start with screen context enabled |
| `VOICE_TRANSCRIBE_RELEASE_DEBOUNCE_SECONDS` | `0.2` | Ignore duplicate release events within this window |

## Files

```
macos-gpu-transcribe/
├── transcribe.py          # Main app — menu bar UI, audio, paste, orchestration
├── transcribe_worker.py   # Worker subprocess — model loading & inference
├── key_monitor.py         # Key monitor subprocess — Quartz CGEvent tap
├── hud_overlay.py         # Floating cursor HUD (AppKit borderless window)
├── main_window.py         # Main app window (status, settings, history)
├── screen_context.py      # Optional frontmost-window screenshot + OCR
├── format_text.py         # Post-processing — numbers, currency, percentages
├── install.sh             # Install wizard
├── run.sh                 # Launcher
├── requirements.txt       # pip dependencies
├── .crispasr/             # Local CrispASR checkout/build (ignored)
├── com.local.DisableRightOption.plist  # User LaunchAgent — disables Right Option via hidutil
├── settings.json          # Per-user toggles (auto-managed)
├── history.json           # Transcription log (last 100)
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Fn key not detected | Toggle Input Monitoring OFF/ON for Python.app, restart the app |
| No audio recording | Grant Microphone permission when prompted |
| "Loading model…" takes a while on first Granite run | First use may download/load the Granite GGUF and start the local CrispASR server. Later runs should reuse the resident server; if they still look cold, check `/tmp/voice-transcribe.log` for worker restarts or server fallback |
| Granite says CrispASR is not installed | Run `./install.sh` or manually build `.crispasr/build/bin/crispasr`; switch the menu default to Cohere meanwhile |
| Granite returns only punctuation | The app now falls back to Cohere before deleting the audio. Failed recordings are preserved under `failed_recordings/`, and the most recent raw recording is always copied to `last_recording.wav` |
| First press after opening laptop is slow | Should be rare with the wake-hook; if it persists, check for "System woke from sleep → warming ASR model" in the log |
| Multiple menu bar icons | `pkill -9 -f transcribe.py` then restart |
| Cohere: 401 Unauthorized | Request access at `huggingface.co/CohereLabs/cohere-transcribe-03-2026`, re-run `install.sh` |
| Broken pipe error on quit | Expected — subprocesses shutting down |

## Logs

```bash
tail -f /tmp/voice-transcribe.log
```

## License

MIT. See [`LICENSE`](LICENSE) if present; otherwise do whatever you want with it — just don't blame the author if it sets your GPU on fire.
