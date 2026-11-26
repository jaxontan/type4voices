# type4voices

Voice typing for Windows, macOS and Linux. Hold a shortcut, speak, let go, and
the text appears at the cursor in whichever window has focus.

There are three speech engines. By default your clip goes to Google's Gemini 3.5
Transcribe Live over the public Live API with your own API key: documented,
cheap, and no 4.4 GB model in memory while you talk. You can switch to the same
speech-to-text stream Claude Code's `/voice` mode uses, on the Claude login you
already have (see the terms warning below), or to Qwen3-ASR on your own machine,
offline. Whichever heads the queue, the others can catch the clip if it fails:
the whole recording is buffered and replayed into the next engine, so a dead
socket costs latency, not the sentence. Either way a small local language model
then repairs weak or misheard English into what you meant, and that part never
leaves the machine. See [Where the speech goes](#where-the-speech-goes).

The default shortcut is the second mouse side button (Forward). You can record
any key, key combination or mouse click instead, from the tray menu.

Windows 11 is the platform this was built and tested on: Core Ultra 5 228V,
Arc 130V, 31.5 GB RAM, Python 3.12, inference on the Arc GPU (torch XPU build).
The macOS and Linux paths use
the same code with the OS-specific parts in `platform_support.py`, but they have
not been run on real hardware yet. See [Platform notes](#platform-notes).

## What it does

- Holding the shortcut records. Releasing transcribes and pastes.
- The tray icon is a two-lamp traffic light: green while recording or
  transcribing, red while idle, both lamps dark while paused.
- A small panel carries the same traffic light: two lamps, the state in small
  caps, the timer while recording, a live microphone level, and then the
  recognised text. Click it for the history. It never takes keyboard focus, so
  the paste still lands in your window.
- The transcript stays on the clipboard after pasting. Left-click the traffic
  light to see the last five clipboard items and click one to copy it back.
- Spoken numbers, punctuation and filler words are cleaned up before pasting.
- Speech is transcribed by Gemini (API key), by Claude on your subscription, or
  by Qwen3-ASR on this machine; one tray click picks the first engine, and the
  rest of the chain catches a failure. Your hotwords and the terms visible in
  the focused window travel with the audio as keyterms either way.
- Weak grammar and misheard words are rewritten into clean English by a local
  LLM ("I already tell him many time" → "I've already told him many times").
  See [Fixing weak English](#fixing-weak-english).
- On Windows the shortcut is swallowed before the focused app sees it, so the
  side button no longer navigates the browser.

## Where the speech goes

**Gemini (default).** The clip is streamed to Google's
`wss://generativelanguage.googleapis.com/ws/...BidiGenerateContent` Live
endpoint, authenticated with the API key in `gemini_api_key` (or
`GEMINI_API_KEY`). This is the documented, supported path: hotwords go in as
`customVocabulary`, `VERBATIM` mode keeps your wording, and the free tier covers
light use (about 0.005 USD per minute paid). If no key is configured, the chain
falls through to the next engine.

**Claude (opt-in, terms risk).** The clip is streamed to
`wss://api.anthropic.com/api/ws/speech_to_text/voice_stream`, authenticated with
the OAuth token Claude Code wrote to `~/.claude/.credentials.json`. That file is
the only place a credential is read from or written to. When the access token
has expired, the app exchanges the stored refresh token at the same endpoint and
with the same client id the installed Claude Code client uses, then writes the
rotated pair back — so your own client stays signed in too (a compare-and-swap
keeps its newer token if it refreshed first). Set
`"auto_refresh_claude_token": false` to stop that.

**This is not permitted by Anthropic's terms, and Anthropic enforces it.** The
endpoint is Claude Code's internal voice stream, not a documented API. Anthropic's
[Claude Code legal page](https://code.claude.com/docs/en/legal-and-compliance)
says OAuth authentication is intended *"exclusively"* for Claude Code and other
native Anthropic applications, that third parties may not *"route requests
through Free, Pro, Max, or Ultra plan credentials"*, and that developers may
not *"collect, store, or intermediate"* session tokens. This app does all three:
it reads the token Claude Code wrote, refreshes it, and writes the rotated pair
back. Anthropic clarified the ban in February 2026 and has suspended accounts
without warning. Keep this as a personal toggle, never ship it as the product's
default, and use Gemini or local for anything public. If the endpoint changes,
`anthropic_stt.py` holds the whole contract in one docstring.

When the login lapses, you are not sent to a terminal. At startup, and again if
a dictation fails on an expired token, the panel shows a **Claude login** card in
the corner — the same non-activating window, so nothing you are typing loses
focus. Clicking it runs `claude auth login` with no console window, which opens
the browser sign-in; the card watches `claude auth status` and turns into
"Claude login restored" when you finish, or offers another try if you don't.
Dismiss it with ✕ or a right-click and dictation keeps its error message until
you sign in.

**Local.** Qwen3-ASR on your machine, offline, no account involved. Pick it in
the tray, or set `"stt_engine": "local"` in `config.json`. The English fixer is
local in all three cases.

**The chain.** The engines are not mutually exclusive. `"stt_engine"` names the
first one, and `"fallback_to_local": true` (the default) appends the local model.
For a longer queue, set `"stt_engine_order"` explicitly, for example
`["gemini", "anthropic", "local"]` or `["anthropic", "local"]`. The whole clip is
kept until a transcript is final, so a provider that dies halfway is discarded
and the identical audio is replayed into the next one; the panel says which
engine caught it and the log keeps the list it tried. A provider that returns an
empty result *without* an error is treated as real silence, and stops the chain.
Choosing an engine from the tray resets the chain to that engine plus the local
fallback.

No engine is allowed to dead-end silently: a clip that Claude or Gemini cannot
transcribe — expired login, bad key, no network, endpoint changed — is
transcribed by the local model instead, and the panel says
*Claude login needed — used the local model*. If the login (or the Gemini key)
is already known to be missing at startup, the local model is warmed in the
background so that fallback is not a cold 45 s load. Set `fallback_to_local` to
`false` if you would rather get nothing than a local transcript.

## Install

You need Python 3.10 or newer. With the local speech engine the environment, the
speech model (4.4 GB) and the English fixer (3.4 GB) take about 10 GB together.
With Claude or Gemini as the engine you still want the English fixer, so budget
3.4 GB and skip the next paragraph's speech model.

The Python code is shared. Each system has its own folder with the launchers,
and all of them create `.venv` in this folder and run `dictate.py` from here.

### Windows

1. Install Python 3.12 and tick *Add python.exe to PATH*.
2. Double-click `windows\setup.bat`. It creates `.venv`, installs the pinned
   dependencies and runs the environment check.
3. Download the models once (about 8 GB, needs internet). `download_model.py`
   fetches the speech model, which only the local engine needs; the second line
   fetches the English fixer, which both engines use:

   ```
   .venv\Scripts\python.exe download_model.py
   .venv\Scripts\python.exe download_model.py --model Qwen/Qwen3-1.7B
   ```

   Without the fixer the app still works; it just pastes the transcript without
   the English fix. On an Intel Arc / Core Ultra laptop, also install the GPU
   build of torch (see [It feels slow](#it-feels-slow)).

4. Start it with `windows\run_dictate.bat` for a console window or
   `windows\run_dictate_background.vbs` for none. Ticking *Start with Windows*
   in the tray menu sets up autostart, so neither file is needed after that.

### macOS

1. Install Python 3.12 from python.org. The system Python has no tkinter.
2. In Terminal, from this folder:

   ```sh
   cd macos
   chmod +x setup.sh run_dictate.sh run_dictate_background.sh run_dictate.command
   ./setup.sh
   cd ..
   .venv/bin/python download_model.py
   ```

3. Start it with `./macos/run_dictate.sh`, or double-click
   `macos/run_dictate.command` in Finder.
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
   cd linux
   chmod +x setup.sh run_dictate.sh run_dictate_background.sh
   ./setup.sh
   cd ..
   .venv/bin/python download_model.py
   ./linux/run_dictate.sh
   ```

3. Log in to an X11/Xorg session. Wayland does not let ordinary applications
   register global keyboard or mouse hooks, or a tray icon, so the shortcut,
   the recording trigger and the tray menu cannot work there. `dictate.py
   --doctor` reports which session it found.

The tray icon appears right away and stays red while the model loads. That is a
few seconds once Python and torch imports are warm, and around 45 s on the first
cold start, where importing torch and transformers dominates. You can dictate as
soon as the icon is red rather than dark. With a cloud engine first in the chain
no speech model loads until a clip actually needs the local fallback, so the
light goes red as soon as the app is up; the English fixer still loads lazily on
the first transcript.

## The model is not in this repository

The weights are not committed here. They are 4.4 GB for the speech model and
3.4 GB for the English fixer, they have their own licenses, and GitHub is the
wrong place to store them.

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

Then set `model_id` in `config.json` to the same string. The English fixer,
`Qwen/Qwen3-1.7B`, is fetched the same way with `--model Qwen/Qwen3-1.7B`. The
license for all three is Apache-2.0, so commercial and offline use is allowed.

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
| Tap shorter than `min_seconds` (0.5 s) | Ignored, to avoid pasting noise |
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

Two checks run before anything is pasted. A clip shorter than `min_seconds`, or
one whose loudness hardly moves (room tone, a fan, a knock on the desk), is
dropped before the model sees it. A transcript that comes back with no letters
or digits is dropped as well, which catches the `????` the model sometimes
emits for noise. If the first pass comes back empty or unreadable, the app
retries once in the last language it transcribed correctly (English by
default), because short or accented clips often trip the language detector.
When `language` is pinned there is no other language to retry in, so the
retry uses detection instead and is dropped unless it comes back English.
When nothing can be pasted, the panel says so instead of staying silent.

Latency on this laptop with the Arc GPU: transcription takes about 1 s for a
few seconds of speech, and the English fixer adds about 1 s, so a sentence
reaches the paste about 2.5 s after you let go. On the CPU-only torch build the
transcription alone took 5 to 7 s.

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
| `stt_engine` | `gemini` | First speech engine: `gemini`, `anthropic` or `local`. The tray radio sets this. |
| `gemini_api_key` | `""` | API key for the Gemini engine. Empty falls back to `GEMINI_API_KEY`. |
| `stt_engine_order` | `[]` | Advanced: explicit provider chain, e.g. `["gemini", "local"]`. Empty derives from `stt_engine` + `fallback_to_local`. |
| `fallback_to_local` | `true` | Append the local model after a cloud engine, so a failed clip is transcribed instead of dropped. |
| `auto_refresh_claude_token` | `true` | Refresh the Claude OAuth token and write the rotated pair back. See the terms warning above. |
| `model_id` | `Qwen/Qwen3-ASR-1.7B` | Model to load. `Qwen/Qwen3-ASR-0.6B` is faster if cached. |
| `language` | `Auto` | Pin to `English`, `Chinese`, etc. to stop language flipping on short clips. |
| `accent` | `""` | A sentence about the speaker, sent with the hotwords. Set it for a strong accent. |
| `trigger_shortcut` | `mouse:x2` | The dictation shortcut: `mouse:x1`, `mouse:x2`, `mouse:middle`, `f8`, `ctrl+alt+j`. |
| `trigger` | `x2` | Legacy key, kept in sync for the mouse buttons so older builds still read it. |
| `min_seconds` | `0.5` | Clips shorter than this are discarded. |
| `max_seconds` | `120.0` | Hard cap on recording length. |
| `sample_rate` | `16000` | Capture rate. The model expects 16 kHz. |
| `device` | `null` | `null` means the default mic, or use an input device index. |
| `torch_device` | `auto` | `auto` uses an Arc (`xpu`), NVIDIA (`cuda`) or Apple (`mps`) GPU when torch sees one, else `cpu`. Falls back to CPU if the GPU load fails. |
| `cleanup_enabled` | `true` | Runs `text_cleanup.postprocess_text()` before pasting. |
| `rewrite_enabled` | `true` | Rewrites the transcript into clean English with a local LLM. Tray → Fix my English. |
| `rewrite_model` | `Qwen/Qwen3-1.7B` | The English-fixer model. Tray → English fixer model. See [the comparison](#model-comparison). |
| `rewrite_worker_python` | `""` | Python for transformers-5 models such as Gemma 4. Empty = `.venv-t5` next to `dictate.py`. |
| `rewrite_api_base` | `""` | Optional OpenAI-compatible server base URL (e.g. local `llama-server`, Ollama `http://127.0.0.1:11434/v1`, or a LAN GPU server). Falls back to the local model when unreachable. |
| `rewrite_api_model` | `""` | Optional model name override for `rewrite_api_base` (defaults to `rewrite_model`). |
| `rewrite_api_key` | `""` | Optional bearer token for `rewrite_api_base`. |
| `context_awareness` | `true` | Reads the active window title/app (<1 ms, local-only) to bias hotwords and adapt style per app; skips password/banking windows. Tray → App context awareness. |
| `second_hearing` | `true` | Transcribes each clip twice in one GPU call (with and without the accent prompt) and gives the fixer both. |
| `auto_gain` | `true` | Boosts quiet speech (capped at 20x) before transcription. |
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

`accent` in `config.json` is one sentence describing the speaker and goes into
that same context, for example *"The speaker has a Singapore and Malaysian
English accent and imperfect grammar; transcribe their words as spoken and do
not correct them."* It is the strongest knob for accented speech, because the
model otherwise tends to tidy the wording up into textbook English.

## Fixing weak English

The speech model writes what it hears: misheard words where pronunciation was
unclear, and your own grammar. A second, local language model reads the whole
transcript and rewrites it into the clear English you meant. Both run on the
GPU and nothing leaves the machine.

| Heard | Pasted |
|---|---|
| I already tell him many time but he still never fix the bug. | I've already told him many times, but he still hasn't fixed the bug. |
| I want to deploy the model on the GP you so the inference more fast. | I want to deploy the model on the GPU so that inference is faster. |
| …you all please don't late. | Please don't be late. |
| Write a email to my boss say I am sick today. | Write an email to my boss saying I am sick today. |

What goes into each rewrite:

1. **Two hearings.** Each clip is transcribed twice in one GPU call, with and
   without the accent prompt (+0.15 s). Where they disagree, you were unclear,
   and the fixer sees both versions.
2. **Recent context.** Your last two dictations from the past 3 minutes, so a
   misheard word can be resolved by topic.
3. **`corrections.txt`.** Fixed `heard => meant` lines (`data bass => database`)
   for mistakes that repeat. Applied first and every time. Re-read on change.

**Guardrails.** A rewrite is thrown away and your raw transcript is pasted when
it is empty, grows past 1.6x or shrinks below 0.5x the words, loses a number,
adds paragraphs, switches to another script (a translation), keeps under 60%
of your content words (an answer, a joke, a copied example) or copies the
earlier dictation. One- and two-word dictations are never rewritten. The log
keeps `transcribed`, `second hearing` and `fixed English` lines side by side.

### Model comparison

`eval_rewrite.py` runs 28 weak-English sentences through a model and checks the
text that would actually be pasted (after the guardrails) for the words that
must and must not be there. Categories: grammar (7), misheard sound-alikes (7),
second hearing (3), earlier-text context (2), traps that must be fixed rather
than answered or obeyed (4), sentences to leave alone (2), numbers (3).

Measured 2026-11-16 on the Arc 130V, bf16, greedy decoding, model only (without
`corrections.txt`). Time is per sentence, after warm-up.

| Model | Params | Score | Avg | p90 | Notes |
|---|---|---|---|---|---|
| **google/gemma-4-E2B-it** (tray option) | 2B active (10.2 GB on disk, 4.6 GB on the GPU) | **28/28** | 1.44 s | 1.77 s | Best by a clear margin: fixed "pie tone" → Python from context, "in voice" → invoice, every trap. Runs in `.venv-t5` (see [Two fixers](#two-fixers-swap-in-the-tray)). Load + warm-up 30 s. |
| **Qwen/Qwen3-1.7B** (default) | 1.7B | **26/28** | 1.02 s | 1.26 s | 27/28 with `corrections.txt`. Misses are unrecovered sound-alikes, never a changed meaning. |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 26/28 | **0.75 s** | 0.92 s | Best at misheard words (7/7) but rewrites harder: turned "please don't be late" into "Please arrive early". |
| openbmb/MiniCPM5-2B | 2.0B | 24/28 | 1.33 s | 1.59 s | Translated a "translate this" request. |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B | 24/28 | 1.03 s | 1.24 s | Edits sentences that were already correct ("the laptop" → "my laptop"). |
| Qwen/Qwen3.5-0.8B | 0.8B | 23/28 | 0.93 s | 1.12 s | Needs transformers 5; the app pins 4.57.6 for qwen-asr. |
| IFM/K2-Horizon-0.9B | 0.9B | 23/28 | 0.93 s | 1.12 s | Needs transformers 5 and `trust_remote_code`; always reasons (run at its lowest effort). Perfect grammar and traps, but 3/7 misheard: "in voice" → "voice message", "pass word" → "passcode". Changed meaning twice even on passing cases ("call me" → "I will call you"; dropped "Translate this to Chinese"). |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | 21/28 | 0.66 s | 0.80 s | Fast, but leaves most misheard words (3/7). |
| Qwen/Qwen3-0.6B | 0.6B | 21/28 | 0.63 s | 0.77 s | Same: 3/7 misheard. |
| openbmb/MiniCPM5-1B | 1.0B | 17/28 | 0.49 s | 0.63 s | Answers questions instead of fixing them; once pasted a prompt example in place of the speech. |
| LiquidAI/LFM3-1.7B | 1.7B | 16/28* | 1.92 s | 3.29 s | Needs transformers 5 and `trust_remote_code`. Half its rewrites ran 2-4x too long and were rejected; reasoning leaked into the output with thinking switched off. Its fast conv kernel is CUDA-only, so it falls back to plain PyTorch on Arc. |
| Qwen/Qwen3-4B-Instruct-2507 | 4.0B | 27/28 | 2.03 s | 2.42 s | Reference above 2B: one more case for twice the time. |

\* LFM3 was scored with an earlier, stricter checker (punctuation-sensitive);
its failures were mostly guardrail rejections, which the change does not affect.
Gemma 3 1B and Llama 3.2 1B were not tested: both need a Hugging Face login
and licence acceptance. LiquidAI/LFM2.5-8B-A1B has 1B
active parameters but 8B in total (16.9 GB), so it is not a sub-2B model in
memory.

The prompt and its examples were tuned while testing Qwen models, so the
comparison favours their habits somewhat. To compare on your own speech, add the
mistakes you see in `dictate.log` to `CASES` in `eval_rewrite.py`, download a
model with `download_model.py --model <id>`, and run:

```
.venv\Scripts\python.exe eval_rewrite.py --model Qwen/Qwen3-1.7B --model <id>
.venv\Scripts\python.exe eval_rewrite.py --model <id> --corrections   # with corrections.txt
```

Models needing transformers 5 run with `.venv-t5\Scripts\python.exe eval_rewrite.py`
instead. To use another model in the app, set `"rewrite_model": "<id>"` in
`config.json` and restart (add it to `WORKER_MODELS` in `dictate.py` if it needs
transformers 5).

### Two fixers, swap in the tray

Tray → **English fixer model** → *Qwen3-1.7B (fast, ~1.0 s)* or *Gemma 4 E2B
(best, ~1.4 s)*. The choice is saved to `config.json` (`rewrite_model`).

- **Qwen3-1.7B** runs inside the app.
- **Gemma 4 E2B** needs transformers 5, which breaks qwen-asr 0.0.6, so it runs
  in a second environment, `.venv-t5`, as a background process
  (`rewrite_worker.py`, log in `rewrite_worker.log`) that the app talks to
  over a pipe. Same prompt, same guardrails. Set it up once:

  ```
  py -3.12 -m venv .venv-t5
  .venv-t5\Scripts\python.exe -m pip install "torch==2.15.0+xpu" --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
  .venv-t5\Scripts\python.exe -m pip install -r requirements-t5.txt
  .venv\Scripts\python.exe download_model.py --model google/gemma-4-E2B-it
  ```

  Gemma 4 is 10.2 GB in bf16, which does not fit next to the speech model in
  the Arc 130V's shared 16 GB. 4.7 GB of it is a per-layer embedding table
  (one row lookup per token), so that stays in RAM, and the audio/vision
  encoders are dropped: 4.6 GB on the GPU, and faster than loading it whole.

Swapping:

- **To Gemma:** Qwen is freed first to make room. For the ~30 s Gemma takes to
  load, dictations get `corrections.txt` only. The tray shows "loading...".
- **To Qwen:** Gemma keeps fixing until Qwen is ready (~13 s), then its process
  is stopped.
- **If Gemma fails** (no `.venv-t5`, out of memory, the process crashes), the
  app logs why and loads Qwen3-1.7B instead. It does not retry Gemma until you
  pick it in the tray again.

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
Run `dictate.py --doctor`. If it says `compute cpu`, torch cannot see a GPU and
transcription runs at only about 1 to 1.5 times realtime. On an Intel Arc / Core
Ultra machine, install the XPU build of torch (about 6 to 7 times faster here):

```
winget install Microsoft.VCRedist.2015+.x64
.venv\Scripts\python.exe -m pip install "torch==2.15.0+xpu" --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
.venv\Scripts\python.exe dictate.py --doctor --full
```

The VC++ redistributable is required: the CPU torch wheel bundles
`vcruntime140_1.dll` but the XPU wheel does not, and without it `import torch`
fails with `WinError 126 ... shm.dll`.

### Quiet or mumbled speech comes out as something else.
A laptop array mic can record speech very quietly, and the model then fills the
gaps with its own guesses (it turned "...four. This is me and I'm in Singapore"
into "...five six seven eight nine ten"). `auto_gain` boosts quiet clips, and the
log shows `boosted quiet audio x6.4` when it does. Raising the microphone level
under Settings → System → Sound → Input still helps more than boosting after.

### The English fixer changed my meaning, or missed a word.
The log keeps both versions: `transcribed ...` is what the speech model heard,
`second hearing ...` the alternative, `fixed English ...` what was pasted. Add a
repeat mistake to `corrections.txt`, or add the sentence as a case in
`eval_rewrite.py` and re-run it before changing models. Tray → Fix my English
turns the fixer off.

## Project layout

The application code sits at the top level and is the same for every system.
The launchers are grouped by system, one folder each, and all of them run the
code from the folder above.

```
dictate.py                 app: tray, audio, queue, paste, CLI, config,
                           provider chain + audio replay
trigger.py                 shortcut backend (Windows hooks / pynput)
capture.py                 "detect a new shortcut" recorder
shortcuts.py               parse, validate, display and match shortcuts
platform_support.py        per-OS paste chord, clipboard, panel, autostart
overlay.py                 mini panel: transcript, clipboard, shortcut recorder
text_cleanup.py            deterministic number/punctuation/filler cleanup
context_awareness.py       active-window app category, dynamic hotwords & privacy guard
anthropic_stt.py           Claude voice stream (opt-in; see the terms warning)
gemini_stt.py              Gemini 3.5 Transcribe Live (documented Live API)
rewrite.py                 English fixer: prompt, guardrails, corrections, OpenAI backend
rewrite_worker.py          runs a transformers-5 fixer (Gemma 4) in .venv-t5
requirements-t5.txt        packages for .venv-t5
eval_rewrite.py            scores fixer models on 28 weak-English cases
test_anthropic_stt.py      tests for the Claude wire contract helpers
test_gemini_stt.py         tests for the Gemini setup/frame encoding
test_dictate.py            tests for settings, shortcuts, audio, engine chain
test_text_cleanup.py       tests for the cleanup pipeline (66 cases)
test_rewrite.py            tests for the fixer's guardrails and corrections
test_context_awareness.py  tests for window context, privacy guard & OpenAI rewriter
test_mic.py                manual microphone and model diagnostic
download_model.py          one-time model download (network allowed here)
landing_page/index.html    project landing page, open in any browser
legacy/cloudstream/        archived Electron prototype (source only, not built)
config.json                created on first run
hotwords.txt               optional domain words
corrections.txt            fixed "heard => meant" replacements
dictate.log                rotating log (1 MB, 3 files)
MODEL_NOTES.md             model choice, licensing, alternatives
```

```
windows/setup.bat              create .venv and install dependencies
windows/run_dictate.bat        foreground launcher, console window
windows/run_dictate_background.vbs  launcher with no console
windows/suppress_xbutton2.ahk  optional legacy AutoHotkey suppresser
macos/setup.sh                 create .venv and install dependencies
macos/run_dictate.sh           foreground launcher
macos/run_dictate_background.sh  launcher with no terminal
macos/run_dictate.command      double-click launcher for Finder
linux/setup.sh                 create .venv and install dependencies
linux/run_dictate.sh           foreground launcher
linux/run_dictate_background.sh  launcher with no terminal
```

### How a dictation flows

```
trigger.py (shortcut down)  -> start_recording()          [never blocks]
audio callback (16 kHz)     -> appends frames while recording
trigger.py (shortcut up)    -> stop_recording_and_process()
   trim silence -> duration guards -> speech gate -> auto gain -> job queue
worker thread               -> transcribe_with_engine() walks the chain
                               (gemini / anthropic / local), replaying the
                               same clip into the next provider on failure
                            -> postprocess_text() -> corrections.txt
                            -> rewrite (LLM + guardrails) -> paste
clipboard watcher (0.6 s)   -> ClipboardHistory.add() -> left-click panel
```

One inference runs at a time, so quick dictations queue instead of competing for
the CPU. The paste leaves the transcript on the clipboard when `auto_clipboard`
is on, and the watcher adds it to the left-click history.

## Tests

```
.venv\Scripts\python.exe -m unittest -v test_dictate test_rewrite test_context_awareness
.venv\Scripts\python.exe test_text_cleanup.py
.venv\Scripts\python.exe eval_rewrite.py --model Qwen/Qwen3-1.7B   # needs the model, ~1 min
.venv\Scripts\python.exe dictate.py --doctor
.venv\Scripts\python.exe dictate.py --doctor --full
```

On macOS and Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

## Licensing

Application code: use it as you like. The models are Apache-2.0. `MODEL_NOTES.md`
lists the models, their cache locations and the alternatives.
