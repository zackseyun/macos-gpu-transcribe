#!/Users/zackseyun/voice-transcribe/.venv/bin/python3
"""
Voice Transcribe — hold Fn or Right Option to record, release to transcribe & paste.

Menu bar app using Qwen3-ASR via MLX (Metal GPU accelerated).

Architecture:
  - Main process: rumps menu bar app + always-on audio stream
  - Key monitor subprocess: Quartz CGEvent tap for Fn/Right Option detection
  - Transcription worker subprocess: loads Qwen3-ASR model, transcribes on demand

IMPORTANT — CoreAudio threading constraints:
  The audio stream (sounddevice/PortAudio) is opened ONCE at startup and NEVER
  stopped or restarted. This is intentional. PortAudio's Pa_StopStream and
  Pa_AbortStream both call AudioDeviceStop, which tries to acquire CoreAudio's
  internal HALB_Mutex. If the audio callback is running (which it is every ~10ms),
  the callback thread already holds that mutex → DEADLOCK. This happens regardless
  of which thread calls stop/abort. The only safe approach is to never call them.
  Recording is controlled by toggling a boolean flag that the callback checks.
"""

import json
import multiprocessing
import os
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import rumps
import sounddevice as sd

# Subprocess target functions live in separate files (no rumps/sd imports)
import key_monitor
import transcribe_worker
from format_text import format_transcription

# -- Constants --
SAMPLE_RATE = 16000
CHANNELS = 1
HISTORY_FILE = Path(__file__).parent / "history.json"
MAX_HISTORY = 100
ICON_IDLE = "🎙"
ICON_RECORDING = "🔴"
ICON_PROCESSING = "⏳"
TRANSCRIBE_TIMEOUT = 30  # seconds — kill worker if it takes longer


class VoiceTranscribeApp(rumps.App):
    def __init__(self, key_pipe):
        super().__init__(ICON_IDLE, quit_button=None)

        self.key_pipe = key_pipe
        self._key_monitor = None  # set by caller after construction
        self.is_recording = False
        self.is_processing = False
        self.audio_buffer = []
        self.history = self._load_history()
        self._pending_title = None
        self._recording_start_time = None
        self._last_heartbeat = time.time()

        # Transcription worker subprocess (holds the ML model in GPU memory)
        self._tx_req_parent, self._tx_req_child = multiprocessing.Pipe()
        self._tx_res_parent, self._tx_res_child = multiprocessing.Pipe()
        self._tx_worker = None
        self._spawn_transcribe_worker()

        self._rebuild_menu()

        # Open always-on audio stream. See module docstring for why we never close it.
        self._open_audio_stream()

        # Poll pipe for key events in background thread
        threading.Thread(target=self._poll_pipe, daemon=True).start()

    # ── Key Monitor Management ──
    # The key monitor runs as a subprocess with a Quartz CGEvent tap.
    # macOS can disable event taps after sleep/wake — the monitor auto-recovers,
    # but if it dies completely, the heartbeat watchdog here restarts it.

    def _restart_key_monitor(self):
        """Kill and restart the key monitor subprocess."""
        if self._key_monitor and self._key_monitor.is_alive():
            self._key_monitor.kill()
            self._key_monitor.join(timeout=2)

        # If we were recording when monitor died, stop recording
        if self.is_recording:
            self._stop_recording()

        parent_conn, child_conn = multiprocessing.Pipe()
        self.key_pipe = parent_conn
        self._key_monitor = multiprocessing.Process(
            target=key_monitor.run, args=(child_conn,), daemon=True
        )
        self._key_monitor.start()
        self._last_heartbeat = time.time()
        print(f"Key monitor restarted (PID {self._key_monitor.pid})", flush=True)

    # ── Transcription Worker Management ──
    # The worker holds the Qwen3-ASR model in Metal GPU memory.
    # It auto-restarts after 50 transcriptions or 4GB active memory to prevent
    # MLX Metal buffer leaks from accumulating (see transcribe_worker.py).

    def _spawn_transcribe_worker(self):
        """Start (or restart) the transcription worker process."""
        if self._tx_worker and self._tx_worker.is_alive():
            self._tx_worker.kill()
            self._tx_worker.join(timeout=2)

        # Fresh pipes for clean state
        self._tx_req_parent, self._tx_req_child = multiprocessing.Pipe()
        self._tx_res_parent, self._tx_res_child = multiprocessing.Pipe()

        self._tx_worker = multiprocessing.Process(
            target=transcribe_worker.run,
            args=(self._tx_req_child, self._tx_res_child),
            daemon=True,
        )
        self._tx_worker.start()

    def _transcribe_via_worker(self, wav_path, model_mode="fast"):
        """Send wav to worker and wait for result with timeout. Returns dict or None."""
        request = {"wav_path": wav_path, "model_mode": model_mode}
        try:
            self._tx_req_parent.send(request)
        except (BrokenPipeError, OSError):
            print("Worker pipe broken, respawning...", flush=True)
            self._spawn_transcribe_worker()
            self._tx_req_parent.send(request)

        # Wait for result with timeout
        if self._tx_res_parent.poll(timeout=TRANSCRIBE_TIMEOUT):
            result = self._tx_res_parent.recv()

            # Worker may send a restart signal after the result
            if self._tx_res_parent.poll(timeout=0.1):
                extra = self._tx_res_parent.recv()
                if isinstance(extra, dict) and extra.get("__restart__"):
                    print("Worker requested restart (memory pressure), respawning...", flush=True)
                    self._spawn_transcribe_worker()

            # Handle case where the restart signal came instead of a result
            if isinstance(result, dict) and result.get("__restart__"):
                print("Worker requested restart (memory pressure), respawning...", flush=True)
                self._spawn_transcribe_worker()
                return None

            return result
        else:
            print(f"Transcription timed out after {TRANSCRIBE_TIMEOUT}s, killing worker...", flush=True)
            self._spawn_transcribe_worker()
            return None

    # ── UI Helpers ──

    def _set_title(self, icon):
        """Queue title change for main thread."""
        self._pending_title = icon

    @rumps.timer(0.1)
    def _tick(self, _):
        """Main-thread timer: apply pending title + update recording duration."""
        if self._pending_title is not None:
            self.title = self._pending_title
            self._pending_title = None
        elif self.is_recording and self._recording_start_time:
            elapsed = time.time() - self._recording_start_time
            self.title = f"🔴 {elapsed:.0f}s"

    def _play_sound(self, name):
        """Play a system sound (non-blocking)."""
        subprocess.Popen(
            ["afplay", f"/System/Library/Sounds/{name}.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # ── Pipe Polling ──
    # Runs in a background thread. Receives messages from the key monitor subprocess.
    # Messages: "heartbeat", "down:fast", "down:accurate", "up"

    def _poll_pipe(self):
        self._last_heartbeat = time.time()
        self._pending_model = None  # "fast" (0.6B) or "accurate" (1.7B)
        while True:
            try:
                # Use poll with timeout so we can detect a dead key monitor
                if self.key_pipe.poll(timeout=30):
                    msg = self.key_pipe.recv()
                else:
                    if time.time() - self._last_heartbeat > 30:
                        print("Key monitor appears dead (no heartbeat), restarting...", flush=True)
                        self._restart_key_monitor()
                    continue

                if msg == "heartbeat":
                    self._last_heartbeat = time.time()
                    continue
                elif msg.startswith("down:"):
                    if not self.is_recording and not self.is_processing:
                        mode = msg.split(":")[1]  # "fast", "accurate", or "cohere"
                        self._pending_model = mode
                        labels = {"fast": "Fn", "accurate": "Right Option", "cohere": "Fn"}
                        names = {"fast": "Qwen3 0.6B", "accurate": "Qwen3 1.7B", "cohere": "Cohere 2B"}
                        label = labels.get(mode, mode)
                        model_name = names.get(mode, mode)
                        print(f"{label} hold → recording ({model_name})", flush=True)
                        self._play_sound("Tink")
                        self._start_recording()
                elif msg == "up":
                    if self.is_recording:
                        elapsed = time.time() - self._recording_start_time if self._recording_start_time else 0
                        names = {"fast": "Qwen3 0.6B", "accurate": "Qwen3 1.7B", "cohere": "Cohere 2B"}
                        model_name = names.get(self._pending_model, self._pending_model)
                        print(f"Release → stop ({elapsed:.1f}s), transcribing with {model_name}...", flush=True)
                        self._play_sound("Pop")
                        self._stop_recording()
            except (EOFError, OSError):
                print("Key monitor pipe broken, restarting...", flush=True)
                self._restart_key_monitor()
                continue

    # ── Audio Stream (always-on) ──
    # DO NOT add stream.stop(), stream.abort(), or stream.close() anywhere.
    # See module docstring for the CoreAudio HALB_Mutex deadlock explanation.

    def _open_audio_stream(self):
        """Open a persistent audio input stream. Runs for the lifetime of the app.

        The callback always fires (~every 10ms). When is_recording is False, it
        discards the data (negligible CPU). When True, it appends to audio_buffer.
        """
        device_idx = None
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0 and 'MacBook' in dev['name']:
                    device_idx = i
                    break
            if device_idx is None:
                for i, dev in enumerate(sd.query_devices()):
                    if dev['max_input_channels'] > 0:
                        device_idx = i
                        break
        except Exception:
            pass

        dev_name = sd.query_devices(device_idx)['name'] if device_idx is not None else 'default'
        print(f"Audio stream opened on device {device_idx}: {dev_name}", flush=True)

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}", flush=True)
            if self.is_recording:
                self.audio_buffer.append(indata.copy())

        self._audio_stream = sd.InputStream(
            device=device_idx,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )
        self._audio_stream.start()

    # ── Recording ──
    # Start/stop are just flag toggles — no CoreAudio calls, no mutex, no deadlock.

    def _start_recording(self):
        if self.is_recording or self.is_processing:
            return
        self.audio_buffer = []
        self.is_recording = True
        self._recording_start_time = time.time()
        self._set_title(ICON_RECORDING)

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._recording_start_time = None
        self._set_title(ICON_PROCESSING)

        audio_data = self.audio_buffer
        self.audio_buffer = []

        if not audio_data:
            self._set_title(ICON_IDLE)
            return

        model_mode = self._pending_model or "fast"
        threading.Thread(
            target=self._transcribe_and_paste, args=(audio_data, model_mode), daemon=True
        ).start()

    # ── Transcription ──

    def _transcribe_and_paste(self, audio_data, model_mode="fast"):
        self.is_processing = True
        wav_path = None
        try:
            audio = np.concatenate(audio_data, axis=0).flatten()
            duration = len(audio) / SAMPLE_RATE
            peak = np.max(np.abs(audio))
            print(f"Audio: {duration:.1f}s, peak={peak:.4f} ({'SILENT' if peak < 0.001 else 'OK'})", flush=True)

            if duration < 0.3:
                print("Recording too short, skipping.", flush=True)
                rumps.notification("Voice Transcribe", "Too Short", "Recording was less than 0.3 seconds.")
                return

            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

            result = self._transcribe_via_worker(wav_path, model_mode)

            if result is None:
                print("Transcription timed out, skipping.", flush=True)
                rumps.notification("Voice Transcribe", "Timeout", "Transcription took too long. Try again.")
                return

            if result.get("error"):
                raise Exception(result["error"])

            text = result.get("text", "")
            transcribe_time = result.get("time", 0)

            if not text:
                print("No speech detected.", flush=True)
                rumps.notification("Voice Transcribe", "No Speech", "No speech was detected in the recording.")
                return

            text = format_transcription(text)
            print(f"Transcribed ({duration:.1f}s audio → {transcribe_time:.1f}s): {text}", flush=True)
            self._add_to_history(text)
            self._paste_text(text)

        except Exception as e:
            print(f"Transcription error: {e}", flush=True)
            rumps.notification("Voice Transcribe", "Error", str(e))
        finally:
            self.is_processing = False
            self._set_title(ICON_IDLE)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    # ── Paste ──

    def _paste_text(self, text):
        """Paste text at cursor without clobbering the user's clipboard.

        Saves the current clipboard, pastes the transcription via Cmd+V,
        then restores the original clipboard contents. The user's previous
        copy is preserved — transcription just gets inserted inline.
        """
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags,
            CGEventSourceCreate, kCGEventFlagMaskCommand,
        )
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()

        # Save current clipboard contents
        old_contents = pb.stringForType_(NSPasteboardTypeString)
        old_change_count = pb.changeCount()

        # Set clipboard to transcription and paste
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)

        time.sleep(0.05)

        source = CGEventSourceCreate(0)
        key_down = CGEventCreateKeyboardEvent(source, 9, True)
        key_up = CGEventCreateKeyboardEvent(source, 9, False)
        CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
        CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
        CGEventPost(0, key_down)
        CGEventPost(0, key_up)

        # Wait for paste to complete, then restore original clipboard
        time.sleep(0.15)
        if old_contents is not None:
            pb.clearContents()
            pb.setString_forType_(old_contents, NSPasteboardTypeString)

    # ── History ──

    def _load_history(self):
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_history(self):
        try:
            HISTORY_FILE.write_text(json.dumps(self.history, indent=2))
        except IOError as e:
            print(f"Failed to save history: {e}", flush=True)

    def _add_to_history(self, text):
        entry = {"text": text, "timestamp": datetime.now().isoformat()}
        self.history.insert(0, entry)
        self.history = self.history[:MAX_HISTORY]
        self._save_history()
        self._rebuild_menu()

    def _rebuild_menu(self):
        self.menu.clear()

        self.menu.add(rumps.MenuItem("Fn = Cohere 2B | Right Opt = Qwen3 1.7B", callback=None))
        self.menu.add(rumps.separator)

        if self.history:
            self.menu.add(rumps.MenuItem("Recent Transcriptions", callback=None))
            for entry in self.history[:20]:
                text = entry["text"]
                display = text[:80] + "…" if len(text) > 80 else text
                ts = entry.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        display = f"[{dt.strftime('%H:%M')}] {display}"
                    except ValueError:
                        pass
                item = rumps.MenuItem(display, callback=self._copy_history_item)
                item.representedObject = text
                self.menu.add(item)
            self.menu.add(rumps.separator)
            self.menu.add(rumps.MenuItem("Clear History", callback=self._clear_history))
        else:
            self.menu.add(rumps.MenuItem("No transcriptions yet", callback=None))

        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Quit", callback=rumps.quit_application))

    def _copy_history_item(self, sender):
        from AppKit import NSPasteboard, NSPasteboardTypeString
        text = getattr(sender, "representedObject", sender.title)
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)
        rumps.notification("Voice Transcribe", "Copied", text[:100])

    def _clear_history(self, _):
        self.history = []
        self._save_history()
        self._rebuild_menu()


if __name__ == "__main__":
    import atexit
    import signal

    # Hide from dock — menu bar only
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    parent_conn, child_conn = multiprocessing.Pipe()
    monitor = multiprocessing.Process(target=key_monitor.run, args=(child_conn,), daemon=True)
    monitor.start()

    app = VoiceTranscribeApp(parent_conn)
    app._key_monitor = monitor

    # Ensure all child processes are killed on exit (prevents orphan buildup).
    # Without this, each restart leaves zombie key_monitor/worker/tracker processes
    # reparented to launchd (PID 1) that accumulate indefinitely.
    def _cleanup():
        for proc in multiprocessing.active_children():
            proc.kill()
            proc.join(timeout=2)

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), os._exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), os._exit(0)))

    app.run()
