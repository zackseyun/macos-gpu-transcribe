# Voice Transcribe Qwen

A local push-to-talk dictation app for Apple Silicon Macs.

Hold `Fn`, speak, release, and the app transcribes with Qwen3-ASR on your Mac's GPU and pastes the text at your cursor. No API key is needed for transcription, and speech does not leave your machine.

## Features

- Hold-to-talk with the `Fn` / Globe key
- Local Qwen3-ASR 0.6B transcription through MLX / Metal
- Automatic paste into the focused app
- Floating cursor HUD while recording and processing
- Menu bar history for recent transcriptions
- Model pre-warm and keep-warm to reduce cold-start delays
- Optional Screen Assist OCR for local window text context, off by default
- Experimental streaming code is present but disabled by default because full-clip transcription is more reliable for dictation

## Requirements

- Apple Silicon Mac: M1, M2, M3, or M4
- macOS 13 or newer
- Homebrew
- Python 3.13 or 3.14 from Homebrew
- A few GB of free disk space for model weights

The first transcription downloads `Qwen/Qwen3-ASR-0.6B` from Hugging Face through `mlx-qwen3-asr`.

## Quick Start

```bash
git clone https://github.com/zackseyun/macos-gpu-transcribe.git
cd macos-gpu-transcribe
./install.sh
./run.sh
```

The installer also creates:

```text
dist/Voice Transcribe Qwen.app
```

You can open that app bundle after install, but keep the cloned repo in place because the app wrapper points back to this checkout.

## macOS Permissions

macOS permissions are the fiddly part. The app runs through Homebrew's `Python.app`, so grant permissions to the Python.app path printed by `./install.sh`.

Required:

- System Settings -> Keyboard -> set "Press Globe key to" to "Do Nothing"
- Privacy & Security -> Accessibility -> add the printed `Python.app`
- Privacy & Security -> Input Monitoring -> add the same `Python.app`
- Microphone -> allow when macOS prompts on first recording

Optional:

- Privacy & Security -> Screen Recording -> add the same `Python.app` only if you enable Screen Assist

If the `Fn` key stops being detected after changing permissions, quit the app and run `./run.sh` again.

## Usage

1. Launch with `./run.sh` or open `dist/Voice Transcribe Qwen.app`.
2. Hold `Fn`.
3. Speak.
4. Release `Fn`.
5. The transcript is pasted into the active text field.

Logs are written to:

```bash
/tmp/voice-transcribe.log
```

Watch logs live:

```bash
tail -f /tmp/voice-transcribe.log
```

## Configuration

Local settings are stored in `settings.json`, which is ignored by git.

Useful environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `VOICE_TRANSCRIBE_PREWARM_ON_START` | `true` | Load and warm Qwen shortly after launch |
| `VOICE_TRANSCRIBE_KEEP_WARM_INTERVAL` | `20` | Seconds between keep-warm checks while active |
| `VOICE_TRANSCRIBE_KEEP_WARM_MAX_IDLE` | `300` | Stop keep-warming after this much idle time |
| `VOICE_TRANSCRIBE_SILENCE_RMS` | `0.008` | Drop near-silent recordings before ASR |
| `VOICE_TRANSCRIBE_STREAMING` | `false` | Enables experimental streaming mode |
| `VOICE_TRANSCRIBE_SCREEN_CONTEXT_MAX_CHARS` | `320` | Max OCR context injected when Screen Assist is on |

Example:

```bash
VOICE_TRANSCRIBE_PREWARM_ON_START=false ./run.sh
```

## Architecture

```text
Fn key press/release
        |
        v
key_monitor.py  --pipe-->  transcribe.py  --pipe-->  transcribe_worker.py
Quartz event tap            menu app, HUD, audio       Qwen3-ASR model process
```

Why separate processes:

- `key_monitor.py` owns the low-level Quartz event tap.
- `transcribe.py` owns the UI, audio stream, clipboard, paste, and app state.
- `transcribe_worker.py` keeps the ASR model resident and isolated from the UI process.

The audio stream is opened once at startup and kept open. Recording is controlled by a boolean checked inside the audio callback; this avoids CoreAudio deadlocks that can happen when repeatedly starting and stopping PortAudio streams.

## Files

```text
transcribe.py          Main app: menu bar, audio, paste, orchestration
transcribe_worker.py   Worker: Qwen3-ASR loading, warmup, inference
key_monitor.py         Fn key event tap subprocess
hud_overlay.py         Floating recording/processing HUD
main_window.py         Native status/settings window
screen_context.py      Optional frontmost-window OCR context
format_text.py         Transcript cleanup and formatting
install.sh             Dependency and setup helper
run.sh                 Portable launcher
package_app.sh         Builds dist/Voice Transcribe Qwen.app
requirements.txt       Python dependencies
```

## Troubleshooting

| Problem | Try |
| --- | --- |
| `Fn` does nothing | Re-grant Input Monitoring for Python.app, then restart the app |
| Text does not paste | Re-grant Accessibility for Python.app |
| No microphone input | Allow Microphone access when prompted, or remove/re-add Python.app in Privacy settings |
| First run is slow | The model is downloading or compiling MLX kernels; later runs should be faster |
| A recording is ignored | Check `/tmp/voice-transcribe.log`; the silence gate may have treated it as silence |
| Multiple menu icons | Run `pkill -f transcribe.py`, then relaunch |
| Streaming misses words | Leave `VOICE_TRANSCRIBE_STREAMING` off; full-clip Qwen is the reliable default |

## Notes

- The app is tuned for local dictation, not meeting transcription.
- Qwen3-ASR 0.6B is the default because it is fast and does not require gated model access.
- Cohere support remains in the worker for experiments, but the packaged app does not use it by default.
