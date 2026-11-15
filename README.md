# type4voices — local, offline voice typing

Hold a shortcut, speak, release. Your speech is transcribed **on this
computer** by Qwen3-ASR and pasted into whatever window has focus. No audio, no
text and no telemetry ever leaves the machine.

Runs on **Windows, macOS and Linux** from the same folder. Windows is the tested
platform; the macOS and Linux paths use the same code with platform-specific
pieces (paste chord, tray, panel, clipboard) in `platform_support.py`.

- **Trigger:** hold **XButton2** (Forward) by default — or let the app *detect
  any key, key combination or mouse click* and use that instead
- **Engine:** Qwen3-ASR-1.7B via `qwen-asr`, running locally on CPU
- **Feedback:** the tray icon is a two-lamp traffic light — **green** while the
  app is busy (recording or transcribing), **red** while it is idle, both lamps
  dark while paused. No other colours are used.
- **Transcript panel:** a small always-on-top strip shows the recording timer,
  then the text that was recognised. Click it for a short history. It never
  takes focus, so the paste still lands in your window.
- **Auto clipboard:** every transcript stays on the clipboard, ready to paste
  again somewhere else. **Left-click the traffic light** for the last 5 copies
  and click one to put it back on the clipboard.
- **Cleanup:** spoken numbers, punctuation and "um/uh" are tidied before paste
- **Never steals the shortcut:** on Windows the low-level hook swallows it, so
  browsers do not navigate back/forward while you dictate

Verified on: Windows 11, Intel Core Ultra 5 228V, Arc 130V, 31.5 GB RAM,
Python 3.12, CPU inference. macOS and Linux have not been run on real hardware
yet — see [Platform notes](#platform-notes).

---

## Install

### Windows

1. Install **Python 3.12** (tick *Add python.exe to PATH*).
2. Double-click **`setup.bat`**. It creates `.venv`, installs the pinned
   dependencies and runs the environment check.
3. Download the model once (needs ~4.4 GB of disk and an internet connection):

   ```
   .venv\Scripts\python.exe download_model.py
   ```

   The app itself runs with Hugging Face offline mode forced on, so the model
   must be cached before the first dictation.
4. Start it:
   - `run_dictate.bat` — foreground, with a console
   - `run_dictate_background.vbs` — no console window
   - or enable **Start with Windows** from the tray menu

### macOS

1. Install **Python 3.12** from python.org (the system Python lacks tkinter).
2. In Terminal, in this folder:

   ```sh
   chmod +x setup.sh run_dictate.sh run_dictate_background.sh run_dictate.command
   ./setup.sh
   .venv/bin/python download_model.py
   ```

3. Start it with `./run_dictate.sh`, or double-click `run_dictate.command`.
4. Grant permissions the first time macOS asks (or under **System Settings →
   Privacy & Security**):
   - **Microphone** — recording
   - **Accessibility** — global shortcut hooks
   - **Input Monitoring** — seeing the key or click you record

   The app must be restarted after granting Accessibility/Input Monitoring.

### Linux (X11)

1. Install Python 3.11+, tkinter and PortAudio:

   ```sh
   sudo apt install python3 python3-venv python3-tk libportaudio2 xclip
   ```

   (`xclip` or `xsel` for X11; `wl-clipboard` if you run Wayland.)
2. Then:

   ```sh
   chmod +x setup.sh run_dictate.sh run_dictate_background.sh
   ./setup.sh
   .venv/bin/python download_model.py
   ./run_dictate.sh
   ```

3. Log in with an **X11/Xorg** session. Wayland does not let an application
   register global keyboard/mouse hooks or a tray icon, so the shortcut,
   recording trigger and tray menu cannot work there. `dictate.py --doctor`
   says which session it found.

The tray icon appears immediately and stays red while the model loads. That
takes a few seconds once Python/torch imports are warm, and up to ~45 s on the
very first cold start (importing torch + transformers dominates). The app is
usable for dictation as soon as the icon is red rather than dark.

### Pinning the icon to the taskbar (Windows)

A new tray icon is hidden behind the `^` overflow panel by default. To keep it
always visible, either drag it out of that panel onto the taskbar, or use
Settings → Personalization → Taskbar → *Other system tray icons* →

**Voice Dictation** → On. Windows stores that choice per executable under
`HKCU\Control Panel\NotifyIconSettings` (`IsPromoted` = 1).

---

## Use

| Action | Result |
|---|---|
| Hold the **shortcut**, speak, release | Text is pasted into the focused window |
| Tap shorter than `min_seconds` (1 s) | Ignored — avoids pasting noise |
| Hold longer than `max_seconds` (120 s) | Auto-stopped at the limit |
| Tray → **Shortcut** → *Detect a new shortcut…* | Press any key, combination or mouse click to set it |
| Tray → **Shortcut** → Side button 2 / 1 / Middle | Pick a mouse button directly |
| Tray → Pause / Resume | Releases the microphone |
| Tray → Copy last transcript | Re-copy without recording again |
| Tray → Re-paste last transcript | Paste it again elsewhere |
| Tray → Open Last Recording | Inspect the last WAV |
| Tray → Open log file | See what happened |
| Tray → Transcript panel | Turn the mini panel on or off |
| Tray → Expand / collapse transcript | Same as clicking the panel |
| Tray → Clear transcript history | Forget the shown transcripts |
| **Left-click the tray icon** | Show clipboard history (last 5 copies) |
| Click a clipboard item | Copy it back to the clipboard |
| Right-click the tray icon | The full menu |
| Click the panel | Expand or collapse the history |
| Right-click the panel | Copy the whole history to the clipboard |

### Choosing your own shortcut

Tray → **Shortcut** → **Detect a new shortcut…** opens the panel with
*“Waiting for a key, combination or click…”* and records whatever you do next:

- a mouse button — `Side button 1 (Back)`, `Side button 2 (Forward)`, `Middle click`
- a function or navigation key on its own — `F8`, `Pause`, `Scroll Lock`, …
- a combination — `Ctrl+Alt+J`, `Ctrl+Shift+Space`, `Cmd+Shift+F9` on macOS

The new shortcut is saved to `config.json` and applied **without a restart**.
`Esc`, a click on the panel, a right-click, or the `✕` button cancels.

A bare letter or number is refused on purpose (it would hijack normal typing);
hold Ctrl/Alt/Cmd as well, or pick a function key. While recording on Windows,
keyboard input is swallowed so trying a shortcut does not also trigger the app
behind the panel.

From a terminal, the same thing without the panel:

```
.venv\Scripts\python.exe dictate.py --set-shortcut "ctrl+alt+j"
.venv\Scripts\python.exe dictate.py --set-shortcut "mouse:x2"     # reset
```

The panel sits above the taskbar at the bottom-right and hides itself 8 s after
it becomes idle (`overlay_hide_after`). While dictating it shows the elapsed
time and then the transcript; on an error it says so instead of going quiet.

The clipboard panel opens above the same spot, always expanded, and stays for
15 s. `auto_clipboard` keeps each transcript on the clipboard after pasting —
turn it off if you would rather have your previous clipboard contents restored.

Typical latency on this laptop (CPU): 2 s of speech → ~3 s until paste,
4 s of speech → ~4.5 s.

---

## Configuration

`config.json` is created on first run. Editing it while the app is running is
safe; restart to apply.

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `Qwen/Qwen3-ASR-1.7B` | Model to load. `Qwen/Qwen3-ASR-0.6B` is faster if cached. |
| `language` | `Auto` | Pin to `English`, `Chinese`, … to stop language flipping on short clips. |
| `trigger_shortcut` | `mouse:x2` | The dictation shortcut: `mouse:x1`, `mouse:x2`, `mouse:middle`, `f8`, `ctrl+alt+j`, … |
| `trigger` | `x2` | Legacy key, kept in sync for the mouse buttons (older builds read it). |
| `min_seconds` | `1.0` | Clips shorter than this are discarded. |
| `max_seconds` | `120.0` | Hard cap on recording length. |
| `sample_rate` | `16000` | Capture rate; 16 kHz is what the model expects. |
| `device` | `null` | `null` = default mic, or an input device index. |
| `cleanup_enabled` | `true` | Run `text_cleanup.postprocess_text()` before pasting. |
| `trim_silence` | `true` | Trim lead-in/tail silence before transcription. |
| `suppress_browser_nav` | `true` | Swallow the shortcut system-wide on Windows. `false` = pynput fallback (the shortcut also reaches the app behind). |
| `keep_recording` | `true` | Keep `last_recording.wav`. |
| `use_hotwords` | `true` | Bias transcription with `hotwords.txt`. |
| `paste_delay` | `0.10` | Seconds between clipboard write and the paste chord (Ctrl+V, Cmd+V on macOS). |
| `clipboard_restore_delay` | `0.35` | Seconds before the previous clipboard is restored. |
| `tray_enabled` | `true` | Show the tray icon. |
| `overlay_enabled` | `true` | Show the mini transcript panel. |
| `overlay_hide_after` | `8.0` | Seconds the idle panel stays visible. |
| `overlay_max_entries` | `12` | How many transcripts the panel remembers. |
| `auto_clipboard` | `true` | Keep the transcript on the clipboard after pasting. |
| `clipboard_history_size` | `5` | Clipboard items kept for the left-click popup (1–20). |

### Hotwords

`hotwords.txt` holds one term per line; `#` starts a comment. Terms are passed
to the model as context so names, acronyms and jargon come back spelled the way
you wrote them. The file is re-read automatically when it changes.

---

## Platform notes

| | Windows | macOS | Linux (X11) |
|---|---|---|---|
| Shortcut backend | low-level hooks, event is swallowed | pynput listener, no suppression | pynput listener, no suppression |
| Paste chord | Ctrl+V | Cmd+V | Ctrl+V |
| Clipboard | `pyperclip` | `pyperclip` | `wl-copy` / `xclip` / `xsel` |
| Panel focus | `WS_EX_NOACTIVATE` window | Non-activating `NSPanel` (PyObjC) | Tk `dock` window |
| Autostart | `HKCU\…\Run` | LaunchAgent plist | `~/.config/autostart` |
| Single instance | named mutex | localhost socket | localhost socket |

**macOS.** Grant Accessibility, Input Monitoring and Microphone, then restart
the app. Without Accessibility the shortcut is never seen; without Input
Monitoring the recorder cannot detect keys. Handoff's *Universal Clipboard* can
copy dictated text to your other Apple devices — turn Handoff off if that is a
concern.

**Linux.** X11 only: Wayland blocks global hooks, synthetic keystrokes and tray
icons for regular apps. Install `xclip` or `xsel` (or `wl-clipboard` for
clipboard access only). The panel is a `dock` window, which some window managers
still allow to take focus — check that the paste lands in the right window.

**Suppression.** Only the Windows backend can swallow the shortcut before the
focused application sees it. On macOS and Linux the shortcut is *observed*: if
`Ctrl+Alt+J` is already bound in the app you are typing into, both actions run.
Pick something unused, or use a mouse button.

---

**Tray icon stays gray / says "Model failed - open log".**
Open `dictate.log`. Usually the model is not cached: run `download_model.py`,
or `dictate.py --doctor` to confirm.

**Nothing is pasted.**
Run `dictate.py --doctor`. If the microphone check fails, check the system
privacy settings (Windows: Settings → Privacy → Microphone; macOS: Privacy &
Security → Microphone) and the physical mute switch. On Windows, if you are
dictating into an app running **as administrator** while Voice Dictation is
not, Windows blocks the synthetic Ctrl+V — use tray → *Restart as
Administrator*. On macOS, allow **Accessibility** for the app sending Cmd+V.

**The shortcut still reaches the app / the browser navigates.**
Only Windows can swallow the event; on macOS and Linux the app merely observes
it. On Windows, the low-level hook failed to install (see the log; it falls back
to pynput without suppression) — running as administrator, or closing the
conflicting input tool, usually fixes it.

**The recorder says input is not permitted, or never sees my keys.**
Windows: the low-level hooks are blocked (often by an antivirus/input tool).
macOS: grant **Accessibility** *and* **Input Monitoring**, then restart.
Linux: you are probably in a Wayland session — log in with X11.

**I cannot set a single letter as the shortcut.**
By design: a bare letter would fire while you type normally. Add Ctrl/Alt/Cmd,
or use a function key such as F8.

**Text is garbled or in the wrong language.**
Set `language` in `config.json` (or the tray menu) instead of `Auto`, and add
the words you actually say to `hotwords.txt`.

**Two icons / double paste.**
Only one instance can run; the second exits deliberately with a message. If
you still see duplicates, an old copy is running from another folder.

**It feels slow.**
CPU inference is ~1–1.5× realtime. Installing the XPU build of torch enables
the Arc GPU:

```
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/xpu
```

---

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
test_mic.py                manual microphone + model diagnostic
download_model.py          one-time model download (network allowed here)
landing.html               single-file marketing page (open in any browser)
config.json                created on first run
hotwords.txt               optional domain words
dictate.log                rotating log (1 MB × 3)
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

Only one inference runs at a time, so rapid dictations queue instead of
fighting over the CPU. Paste leaves the transcript on the clipboard
(`auto_clipboard`), and the watcher records it for the left-click history.

---

## Tests

```
.venv\Scripts\python.exe -m unittest -v test_dictate
.venv\Scripts\python.exe test_text_cleanup.py
.venv\Scripts\python.exe dictate.py --doctor
.venv\Scripts\python.exe dictate.py --doctor --full
```

On macOS/Linux use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

---

## Licensing

Application code: use as you like. Models are Apache-2.0 — see
`MODEL_NOTES.md` for the model list, cache locations and alternatives.
