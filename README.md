# type4voices

Local voice typing for Windows, macOS and Linux. Hold a shortcut, speak, let go,
and the text appears at the cursor in whichever window has focus. Qwen3-ASR does
the transcription on your machine. Audio, text and telemetry stay there.

The default shortcut is the second mouse side button (Forward). You can record
any key, key combination or mouse click instead, from the tray menu.

Windows 11 is the platform this was built and tested on: Core Ultra 5 228V,
Arc 130V, 31.5 GB RAM, Python 3.12, CPU inference. The macOS and Linux paths use
the same code with the OS-specific parts in `platform_support.py`, but they have
not been run on real hardware yet. See [Platform notes](#platform-notes).

## What it does

- Holding the shortcut records. Releasing transcribes and pastes.
- The tray icon is a two-lamp traffic light: green while recording or
  transcribing, red while idle, both lamps dark while paused.
- A small panel shows the recording timer and then the recognised text. Click it
  for the history. It never takes keyboard focus, so the paste still lands in
  your window.
- The transcript stays on the clipboard after pasting. Left-click the traffic
  light to see the last five clipboard items and click one to copy it back.
- Spoken numbers, punctuation and filler words are cleaned up before pasting.
- On Windows the shortcut is swallowed before the focused app sees it, so the
  side button no longer navigates the browser.

## Install

You need Python 3.10 or newer. The environment and the model take about 6 GB
together, most of it the model.

### Windows

1. Install Python 3.12 and tick *Add python.exe to PATH*.
2. Double-click `setup.bat`. It creates `.venv`, installs the pinned
   dependencies and runs the environment check.
3. Download the model once (about 4.4 GB, needs internet):

   ```
   .venv\Scripts\python.exe download_model.py
   ```

4. Start it with `run_dictate.bat` for a console window,
   `run_dictate_background.vbs` for none, or tick *Start with Windows* in the
   tray menu.

### macOS

1. Install Python 3.12 from python.org. The system Python has no tkinter.
2. In Terminal, in this folder:

   ```sh
   chmod +x setup.sh run_dictate.sh run_dictate_background.sh run_dictate.command
   ./setup.sh
   .venv/bin/python download_model.py
   ```

3. Start it with `./run_dictate.sh`, or double-click `run_dictate.command`.
4. Allow Microphone, Accessibility and Input Monitoring when macOS asks, or add
   them under System Settings → Privacy & Security. Restart the app after
   changing Accessibility or Input Monitoring.

### Linux

1. Install Python 3.11+, tkinter, PortAudio and a clipboard tool:

   ```sh
   sudo apt install python3 python3-venv python3-tk libportaudio2 xclip
   ```

   Use `xsel` instead of `xclip` if you prefer it. Wayland sessions need
   `wl-clipboard` for the clipboard only.
2. Then:

   ```sh
   chmod +x setup.sh run_dictate.sh run_dictate_background.sh
   ./setup.sh
   .venv/bin/python download_model.py
   ./run_dictate.sh
   ```

3. Log in to an X11/Xorg session. Wayland does not let ordinary applications
   register global keyboard or mouse hooks, or a tray icon, so the shortcut,
   the recording trigger and the tray menu cannot work there. `dictate.py
   --doctor` reports which session it found.

The tray icon appears right away and stays red while the model loads. That is a
few seconds once Python and torch imports are warm, and around 45 s on the first
cold start, where importing torch and transformers dominates. You can dictate as
soon as the icon is red rather than dark.

## The model is not in this repository

The weights are not committed here. They are 4.4 GB, they have their own
license, and GitHub is the wrong place to store them.

`download_model.py` fetches `Qwen/Qwen3-ASR-1.7B` from Hugging Face once and
caches it at:

```
%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen3-ASR-1.7B   (Windows)
~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B               (macOS, Linux)
```

After that the app runs with `HF_HUB_OFFLINE=1` and never touches the network.
A smaller option is `Qwen/Qwen3-ASR-0.6B` (about 1.2 GB):

```
.venv\Scripts\python.exe download_model.py --model Qwen/Qwen3-ASR-0.6B
```

Then set `model_id` in `config.json` to the same string. The license for both
models is Apache-2.0, so commercial and offline use is allowed.

If you need to hand someone a folder that works with no download step, copy the
`models--Qwen--Qwen3-ASR-1.7B` directory into `model-cache\hub\` on that machine
and set `HF_HOME` to `model-cache` before starting the app. That is a manual
step today; there is no bundled installer.

## Pinning the icon to the taskbar (Windows)

A new tray icon hides behind the `^` overflow panel. Drag it onto the taskbar, or
turn it on under Settings → Personalization → Taskbar → Other system tray icons
→ Voice Dictation. Windows stores the choice per executable under
`HKCU\Control Panel\NotifyIconSettings` with `IsPromoted` set to 1.

## Use

| Action | Result |
|---|---|
| Hold the shortcut, speak, release | Text is pasted into the focused window |
| Tap shorter than `min_seconds` (1 s) | Ignored, to avoid pasting noise |
| Hold longer than `max_seconds` (120 s) | Stopped at the limit |
| Tray → Shortcut → Detect a new shortcut… | Press any key, combination or mouse click to set it |
| Tray → Shortcut → Side button 2 / 1 / Middle | Pick a mouse button directly |
| Tray → Pause / Resume | Releases the microphone |
| Tray → Copy last transcript | Copies it again without recording |
| Tray → Re-paste last transcript | Pastes it again elsewhere |
| Tray → Open Last Recording | Opens the last WAV |
| Tray → Open log file | Opens `dictate.log` |
| Tray → Transcript panel | Turns the mini panel on or off |
| Tray → Expand / collapse transcript | Same as clicking the panel |
| Tray → Clear transcript history | Forgets the shown transcripts |
| Left-click the tray icon | Shows clipboard history, last five copies |
| Click a clipboard item | Copies it back to the clipboard |
| Right-click the tray icon | Opens the full menu |
| Click the panel | Expands or collapses the history |
| Right-click the panel | Copies the whole history to the clipboard |

The panel sits above the taskbar at the bottom right and hides itself 8 s after
it goes idle (`overlay_hide_after`). While dictating it shows the elapsed time,
then the transcript. On an error it says so instead of going quiet.

The clipboard panel opens in the same spot, always expanded, and stays for 15 s.
`auto_clipboard` keeps each transcript on the clipboard after pasting. Turn it
off if you would rather get your previous clipboard contents back.

Latency on this laptop, CPU only: 2 s of speech reaches a paste in about 3 s,
4 s of speech in about 4.5 s.

### Choosing your own shortcut

Tray → Shortcut → Detect a new shortcut… opens the panel with "Waiting for a
key, combination or click…" and records what you do next:

- a mouse button: `Side button 1 (Back)`, `Side button 2 (Forward)`, `Middle click`
- a function or navigation key on its own: `F8`, `Pause`, `Scroll Lock`
- a combination: `Ctrl+Alt+J`, `Ctrl+Shift+Space`, or `Cmd+Shift+F9` on macOS

The shortcut is saved to `config.json` and applied without a restart. Esc, a
click on the panel, a right-click or the `✕` button cancels.

A bare letter or number is refused, because it would fire while you type
normally. Hold Ctrl, Alt or Cmd as well, or use a function key. On Windows,
keyboard input is suppressed while the recorder runs, so trying a shortcut does
not also trigger the app behind the panel.

From a terminal, without the panel:

```
.venv\Scripts\python.exe dictate.py --set-shortcut "ctrl+alt+j"
.venv\Scripts\python.exe dictate.py --set-shortcut "mouse:x2"     # reset
```

## Configuration

`config.json` is created on the first run. Editing it while the app runs is
safe; restart to apply.

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `Qwen/Qwen3-ASR-1.7B` | Model to load. `Qwen/Qwen3-ASR-0.6B` is faster if cached. |
| `language` | `Auto` | Pin to `English`, `Chinese`, etc. to stop language flipping on short clips. |
| `trigger_shortcut` | `mouse:x2` | The dictation shortcut: `mouse:x1`, `mouse:x2`, `mouse:middle`, `f8`, `ctrl+alt+j`. |
| `trigger` | `x2` | Legacy key, kept in sync for the mouse buttons so older builds still read it. |
| `min_seconds` | `1.0` | Clips shorter than this are discarded. |
| `max_seconds` | `120.0` | Hard cap on recording length. |
| `sample_rate` | `16000` | Capture rate. The model expects 16 kHz. |
| `device` | `null` | `null` means the default mic, or use an input device index. |
| `cleanup_enabled` | `true` | Runs `text_cleanup.postprocess_text()` before pasting. |
| `trim_silence` | `true` | Trims lead-in and tail silence before transcription. |
| `suppress_browser_nav` | `true` | Swallows the shortcut system-wide on Windows. `false` uses the pynput fallback, where the shortcut also reaches the app behind. |
| `keep_recording` | `true` | Keeps `last_recording.wav`. |
| `use_hotwords` | `true` | Biases transcription with `hotwords.txt`. |
| `paste_delay` | `0.10` | Seconds between the clipboard write and the paste chord (Ctrl+V, Cmd+V on macOS). |
| `clipboard_restore_delay` | `0.35` | Seconds before the previous clipboard is restored. |
| `tray_enabled` | `true` | Shows the tray icon. |
| `overlay_enabled` | `true` | Shows the mini transcript panel. |
| `overlay_hide_after` | `8.0` | Seconds the idle panel stays visible. |
| `overlay_max_entries` | `12` | How many transcripts the panel remembers. |
| `auto_clipboard` | `true` | Keeps the transcript on the clipboard after pasting. |
| `clipboard_history_size` | `5` | Clipboard items kept for the left-click popup, 1 to 20. |

### Hotwords

`hotwords.txt` holds one term per line, and `#` starts a comment. The terms are
passed to the model as context, so names, acronyms and jargon come back the way
you wrote them. The file is re-read when it changes.

## Platform notes

| | Windows | macOS | Linux (X11) |
|---|---|---|---|
| Shortcut backend | low-level hooks, event is swallowed | pynput listener, no suppression | pynput listener, no suppression |
| Paste chord | Ctrl+V | Cmd+V | Ctrl+V |
| Clipboard | `pyperclip` | `pyperclip` | `wl-copy`, `xclip` or `xsel` |
| Panel focus | `WS_EX_NOACTIVATE` window | non-activating `NSPanel` via PyObjC | Tk `dock` window |
| Autostart | `HKCU\…\Run` | LaunchAgent plist | `~/.config/autostart` |
| Single instance | named mutex | localhost socket | localhost socket |

macOS: grant Accessibility, Input Monitoring and Microphone, then restart the
app. Without Accessibility the shortcut is never seen. Without Input Monitoring
the recorder cannot detect keys. Handoff's Universal Clipboard can copy dictated
text to your other Apple devices, so turn Handoff off if that matters.

Linux: X11 only. Wayland blocks global hooks, synthetic keystrokes and tray icons
for regular applications. Install `xclip` or `xsel`, or `wl-clipboard` for
clipboard access on Wayland. The panel is a `dock` window, and some window
managers still let it take focus, so check that the paste lands in the right
window.

Suppression: only the Windows backend can swallow the shortcut before the
focused application sees it. On macOS and Linux the shortcut is observed, so if
`Ctrl+Alt+J` is already bound in the app you are typing into, both actions run.
Pick something unused or use a mouse button.

## Troubleshooting

### Tray icon stays gray, or says "Model failed - open log".
Open `dictate.log`. The usual cause is a model that is not cached. Run
`download_model.py`, then `dictate.py --doctor` to confirm.

### Nothing is pasted.
Run `dictate.py --doctor`. If the microphone check fails, look at the privacy
settings (Windows: Settings → Privacy → Microphone; macOS: Privacy & Security →
Microphone) and the physical mute switch. On Windows, a non-elevated app cannot
send keystrokes to an app running as administrator, so use tray → Restart as
Administrator for those. On macOS, allow Accessibility for the app that sends
Cmd+V.

### The shortcut still reaches the app, or the browser navigates.
Only Windows can swallow the event, so on macOS and Linux the app merely
observes it. On Windows this means the low-level hook failed to install; the log
says so, and the app falls back to pynput without suppression. Running as
administrator or closing the conflicting input tool usually fixes it.

### The recorder says input is not permitted, or never sees my keys.
Windows: the low-level hooks are blocked, often by antivirus or another input
tool. macOS: grant Accessibility and Input Monitoring, then restart. Linux: you
are probably in a Wayland session, so log in with X11.

### I cannot set a single letter as the shortcut.
That is deliberate, since a bare letter would fire while you type normally. Add
Ctrl, Alt or Cmd, or use a function key such as F8.

### Text is garbled or in the wrong language.
Set `language` in `config.json` or the tray menu instead of `Auto`, and add the
words you actually say to `hotwords.txt`.

### Two icons, or a double paste.
Only one instance can run. A second copy exits with a message. If you still see
duplicates, an older copy is running from another folder.

### It feels slow.
CPU inference runs at about 1 to 1.5 times realtime. Installing the XPU build of
torch enables the Arc GPU:

```
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/xpu
```

## Project layout

```
dictate.py                 app: tray, audio, queue, paste, CLI, config
trigger.py                 shortcut backend (Windows hooks / pynput)
capture.py                 "detect a new shortcut" recorder
shortcuts.py               parse, validate, display and match shortcuts
platform_support.py        per-OS paste chord, clipboard, panel, autostart
overlay.py                 mini panel: transcript, clipboard, shortcut recorder
text_cleanup.py            deterministic number/punctuation/filler cleanup
test_dictate.py            tests for settings, shortcuts, audio helpers
test_text_cleanup.py       tests for the cleanup pipeline (63 cases)
test_mic.py                manual microphone and model diagnostic
download_model.py          one-time model download (network allowed here)
landing.html               single-file page, open in any browser
config.json                created on first run
hotwords.txt               optional domain words
dictate.log                rotating log (1 MB, 3 files)
setup.bat / setup.sh       create .venv and install dependencies
run_dictate.bat            Windows foreground launcher
run_dictate_background.vbs Windows launcher, no console
run_dictate.sh             macOS/Linux foreground launcher
run_dictate_background.sh  macOS/Linux launcher, no terminal
run_dictate.command        macOS double-click launcher
suppress_xbutton2.ahk      optional legacy AutoHotkey suppresser
MODEL_NOTES.md             model choice, licensing, alternatives
```

### How a dictation flows

```
trigger.py (shortcut down)  -> start_recording()          [never blocks]
audio callback (16 kHz)     -> appends frames while recording
trigger.py (shortcut up)    -> stop_recording_and_process()
   trim silence -> duration guards -> push to job queue
worker thread               -> transcribe() -> postprocess_text() -> paste
clipboard watcher (0.6 s)   -> ClipboardHistory.add() -> left-click panel
```

One inference runs at a time, so quick dictations queue instead of competing for
the CPU. The paste leaves the transcript on the clipboard when `auto_clipboard`
is on, and the watcher adds it to the left-click history.

## Tests

```
.venv\Scripts\python.exe -m unittest -v test_dictate
.venv\Scripts\python.exe test_text_cleanup.py
.venv\Scripts\python.exe dictate.py --doctor
.venv\Scripts\python.exe dictate.py --doctor --full
```

On macOS and Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

## Licensing

Application code: use it as you like. The models are Apache-2.0. `MODEL_NOTES.md`
lists the models, their cache locations and the alternatives.
