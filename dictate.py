"""
Local Voice Dictation for Windows.

Hold a mouse side button, speak, release. The speech is transcribed locally by
Qwen3-ASR and pasted into whatever window has focus. Nothing leaves the PC.

Target: Windows laptop with Intel Core Ultra 228V / Arc 130V / 32 GB RAM.

Usage:
    python dictate.py              run in the foreground (console output)
    python dictate.py --doctor     check dependencies, microphone and model
    python dictate.py --doctor --full   also load the model and transcribe a clip
    pythonw dictate.py             run without a console (see dictate.log)

Configuration lives in config.json next to this file and is created on first
run. Optional hotwords (names, jargon, identifiers) live in hotwords.txt.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import logging.handlers
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import zlib
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageDraw
from pynput import keyboard, mouse

import context_awareness
import platform_support
import shortcuts
import rewrite
from text_cleanup import postprocess_text
from trigger import create_trigger

# The model is already cached locally. Force cache-only mode so startup and
# transcription never make network requests (including update checks). Assign
# directly instead of setdefault so an inherited "0" cannot re-enable network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Silence the one-off "temperature is not a valid generation flag" chatter.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Windows consoles often use a legacy code page. Never let a Chinese/accented
# transcription crash logging (and therefore skip the paste).
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------- constants --
APP_NAME = "Voice Dictation"
VERSION = "2.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
HOTWORDS_PATH = os.path.join(BASE_DIR, "hotwords.txt")
CORRECTIONS_PATH = os.path.join(BASE_DIR, "corrections.txt")
LOG_PATH = os.path.join(BASE_DIR, "dictate.log")
TMP_WAV = os.path.join(BASE_DIR, "last_recording.wav")
WORKER_PATH = os.path.join(BASE_DIR, "rewrite_worker.py")
WORKER_LOG_PATH = os.path.join(BASE_DIR, "rewrite_worker.log")

# English-fixer models offered in the tray: (model id, menu label).
REWRITE_CHOICES = (
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (fast, ~1.0 s)"),
    ("google/gemma-4-E2B-it", "Gemma 4 E2B (best, ~1.8 s)"),
)
# Models that need transformers 5, which breaks qwen-asr: they run in a
# separate process with rewrite_worker_python (rewrite_worker.py).
WORKER_MODELS = {"google/gemma-4-E2B-it", "Qwen/Qwen3.5-0.8B"}
MUTEX_NAME = "VoiceDictation_SingleInstance_v2"

DEFAULTS: dict[str, Any] = {
    "model_id": "Qwen/Qwen3-ASR-1.7B",
    # Where the model runs: "auto" picks an Intel Arc (xpu), NVIDIA (cuda) or
    # Apple (mps) GPU when torch can see one, else the CPU. A GPU is several
    # times faster; it needs a GPU build of torch (see requirements.txt).
    "torch_device": "auto",
    # "Auto" lets the model detect the language. Short or noisy clips flip to
    # the wrong language, so pinning "English" (or "Chinese") is safer if you
    # only ever dictate one language.
    "language": "Auto",
    # Optional sentence describing the speaker, sent to the model
    # together with the hotwords. Useful for a strong accent or
    # non-native grammar: it stops the model from tidying the words up.
    "accent": "",
    "trigger": "x2",                  # legacy: "x2" (Forward) or "x1" (Back)
    "trigger_shortcut": "",           # e.g. "mouse:x2", "ctrl+alt+j", "f8"
    "min_seconds": 0.5,               # ignore taps shorter than this
    "max_seconds": 120.0,             # auto-stop runaway recordings
    "sample_rate": 16000,             # ASR models expect 16 kHz mono
    "device": None,                   # None = default mic, or a device index
    "cleanup_enabled": True,          # tidy numbers / punctuation / fillers
    # Rewrite weak or misheard English into the clean English you meant,
    # with a local LLM on the same GPU. Your exact words stay in the log.
    "rewrite_enabled": True,
    # Transcribe each clip twice in one GPU call (with and without the accent
    # prompt) and give the English fixer both hearings.
    "second_hearing": True,
    # Swap in the tray ("English fixer model"). Scores from eval_rewrite.py
    # (28 weak-English cases, Arc 130V); full table in README.md:
    #   Qwen/Qwen3-1.7B        26/28  1.0s   <- default, runs in-process
    #   google/gemma-4-E2B-it  28/28  1.8s   runs in .venv-t5 (transformers 5)
    "rewrite_model": "Qwen/Qwen3-1.7B",
    # Python for models that need transformers 5 (see WORKER_MODELS).
    # Empty = .venv-t5 next to this script.
    "rewrite_worker_python": "",
    # Optional OpenAI-compatible server for the English fixer (e.g. local
    # llama-server Vulkan/Metal/CUDA GGUF, Ollama http://127.0.0.1:11434/v1,
    # or a LAN GPU server). When unreachable, falls back to the local model.
    "rewrite_api_base": "",
    "rewrite_api_key": "",
    "rewrite_api_model": "",
    # Inspect the focused window title/app (<1 ms, local-only) to extract
    # on-screen technical terms as dynamic hotwords and adapt style per app.
    "context_awareness": True,
    "trim_silence": True,             # trim lead-in/tail silence before ASR
    "auto_gain": True,                # boost quiet speech before ASR
    "suppress_browser_nav": True,     # swallow the shortcut system-wide (Windows)
    "keep_recording": True,           # keep last_recording.wav for inspection
    "use_hotwords": True,             # bias transcription with hotwords.txt
    "paste_delay": 0.10,              # delay between clipboard set and Ctrl+V
    "clipboard_restore_delay": 0.35,  # delay before restoring the clipboard
    "tray_enabled": True,
    # Small non-activating panel that shows what was transcribed. It never
    # takes focus, so the paste still lands in the window you were typing in.
    "overlay_enabled": True,
    "overlay_hide_after": 8.0,        # seconds before the collapsed panel hides
    "overlay_max_entries": 12,        # how many transcripts the panel keeps
    "auto_clipboard": True,           # keep the transcript on the clipboard
    "clipboard_history_size": 5,      # left-click the tray icon: last N copies
}

TRAY_LANGUAGES = (
    "Auto", "English", "Chinese", "Cantonese", "Japanese",
    "Korean", "Spanish", "French", "German", "Russian",
)

# Tray icon: a two-lamp traffic light. Green means the app is busy (recording
# or transcribing), red means it is idle. No other colours are used.
LAMP_RED = (236, 62, 54)
LAMP_GREEN = (52, 204, 96)
HOUSING = (34, 37, 42, 255)
RIM = (10, 11, 13, 255)

STATES = ("loading", "idle", "recording", "transcribing", "paused", "error")

# Which lamp is lit for each state. None = both lamps dark.
STATE_LAMPS = {
    "loading": "red",
    "idle": "red",
    "recording": "green",
    "transcribing": "green",
    "paused": None,
    "error": "red",
}

# ------------------------------------------------------------- Win32 basics --
ERROR_ALREADY_EXISTS = 183

if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateMutexW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_wchar_p,
    )
    _kernel32.CreateMutexW.restype = ctypes.c_void_p
    _kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
else:
    _kernel32 = None

_single_socket: socket.socket | None = None

# ----------------------------------------------------------------- logging --
log = logging.getLogger("dictate")


def setup_logging(verbose: bool = False, console: bool | None = None) -> None:
    """Log to a rotating file (so pythonw runs are diagnosable) and a console."""
    level = logging.DEBUG if verbose else logging.INFO
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        log.addHandler(file_handler)
    except Exception:
        pass

    if console is None:
        console = sys.stderr is not None
    if console and sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        stream_handler.setLevel(level)
        log.addHandler(stream_handler)

    if not log.handlers:
        log.addHandler(logging.NullHandler())


# ----------------------------------------------------------------- settings --
class Settings:
    """Small dict-backed settings object that persists to config.json."""

    def __init__(self, data: dict[str, Any], path: str = CONFIG_PATH):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "path", path)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def set(self, name: str, value: Any, persist: bool = True) -> None:
        self._data[name] = value
        if persist:
            self.save()

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except Exception as exc:
            log.warning("could not save %s: %s", self.path, exc)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


def _coerce(key: str, value: Any) -> Any:
    """Coerce a config value to the type of its default."""
    default = DEFAULTS[key]
    if default is None:
        return value
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, int):
            return int(value)
        return str(value)
    except (TypeError, ValueError):
        log.warning("invalid value for %s in config (%r); using default", key, value)
        return default


def load_settings(path: str = CONFIG_PATH, write_if_missing: bool = True) -> Settings:
    """Merge config.json over the defaults, coercing values to sane types."""
    data = dict(DEFAULTS)
    path = os.path.abspath(path)

    if os.path.exists(path):
        try:
            # utf-8-sig: Notepad and PowerShell can save a BOM, which plain
            # utf-8 json parsing rejects (and the app fell back to defaults).
            with open(path, "r", encoding="utf-8-sig") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if key in DEFAULTS:
                        data[key] = _coerce(key, value)
                    else:
                        log.warning("unknown config key %r ignored", key)
        except Exception as exc:
            log.error("could not read %s (%s); using defaults", path, exc)
    elif write_if_missing:
        log.info("no config found, writing defaults to %s", path)

    data["min_seconds"] = max(0.2, float(data["min_seconds"]))
    data["max_seconds"] = max(data["min_seconds"] + 1.0, float(data["max_seconds"]))
    data["sample_rate"] = int(data["sample_rate"])
    trigger = str(data["trigger"]).strip().lower()
    data["trigger"] = trigger if trigger in ("x1", "x2") else "x2"

    # Version 1 stored only the mouse button. Derive the new shortcut value
    # once, then keep both keys in sync so an older build still works.
    raw_shortcut = str(data.get("trigger_shortcut") or "").strip().lower()
    if not raw_shortcut:
        raw_shortcut = f"mouse:{data['trigger']}"
    try:
        shortcut = shortcuts.parse(raw_shortcut)
    except ValueError:
        log.warning("invalid trigger_shortcut %r; using mouse:x2", raw_shortcut)
        shortcut = shortcuts.parse("mouse:x2")
    data["trigger_shortcut"] = shortcut.canonical()
    if shortcut.is_mouse and shortcut.button in ("x1", "x2"):
        data["trigger"] = shortcut.button

    settings = Settings(data, path)
    if write_if_missing and (not os.path.exists(path) or data != DEFAULTS):
        # Only rewrite when something actually differs from what is on disk,
        # so a hand-edited file keeps its formatting until a value changes.
        try:
            on_disk = None
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8-sig") as handle:
                    on_disk = json.load(handle)
            if on_disk != data:
                settings.save()
        except Exception:
            pass
    return settings


# ------------------------------------------------------------ pure helpers --
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:)\]}]|%|°)")


def classify_duration(seconds: float, min_seconds: float, max_seconds: float) -> str:
    """Return "short", "ok" or "long" for a recording length."""
    if seconds < min_seconds:
        return "short"
    if seconds > max_seconds:
        return "long"
    return "ok"


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold_db: float = -35.0,
    frame_ms: float = 20.0,
    pad_ms: float = 200.0,
) -> np.ndarray:
    """Trim leading/trailing near-silence.

    The threshold is relative to the loudest 20 ms frame, which keeps quiet
    speech intact. If nothing stands out the audio is returned unchanged.
    """
    if audio is None or audio.size == 0:
        return audio

    samples = audio.reshape(-1).astype(np.float32, copy=False)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    frames = len(samples) // frame
    if frames < 2:
        return audio

    window = samples[: frames * frame].reshape(frames, frame)
    rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
    loudest = float(np.max(rms))
    if loudest <= 1e-6:
        return audio

    threshold = loudest * (10.0 ** (threshold_db / 20.0))
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0:
        return audio

    pad = int(sample_rate * pad_ms / 1000.0)
    start = max(0, int(above[0]) * frame - pad)
    end = min(len(samples), (int(above[-1]) + 1) * frame + pad)
    if end <= start:
        return audio
    return samples[start:end]


def speech_modulation_db(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: float = 20.0,
) -> float:
    """How far the frame loudness swings, in dB.

    Speech moves between syllables and the gaps between them, so a loud frame
    and a quiet frame can sit 15 dB or more apart. Room tone, a fan and mains
    hum stay within a couple of dB of themselves.
    """
    if audio is None or audio.size == 0:
        return 0.0
    samples = audio.reshape(-1).astype(np.float32, copy=False)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    frames = len(samples) // frame
    if frames < 2:
        return 0.0

    window = samples[: frames * frame].reshape(frames, frame)
    rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return float(np.percentile(db, 90.0) - np.percentile(db, 10.0))


def normalize_loudness(
    audio: np.ndarray,
    target_peak: float = 0.5,
    max_gain: float = 20.0,
) -> tuple[np.ndarray, float]:
    """Scale quiet speech up to a level the model hears reliably.

    A laptop array mic often records speech peaking near 0.05-0.1. At that
    level the model fills mumbled words with its own guesses (it turned
    "...four. This is me and I'm in Singapore" into "...five six seven eight
    nine ten"). The peak is taken at the 99.9th percentile so a single click
    cannot hold the gain down, and the gain is capped so room noise is not
    blown up into speech. Returns (audio, gain); loud audio is left alone.
    """
    if audio is None or audio.size == 0:
        return audio, 1.0
    peak = float(np.percentile(np.abs(audio), 99.9))
    if peak <= 1e-6 or peak >= target_peak:
        return audio, 1.0
    gain = min(max_gain, target_peak / peak)
    boosted = np.clip(audio * gain, -1.0, 1.0).astype(np.float32, copy=False)
    return boosted, gain


def looks_like_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_modulation_db: float = 3.0,
    min_peak: float = 1e-3,
) -> bool:
    """Reject a clip that is too flat to be speech.

    This is a cheap gate, not a detector. A held note would pass it, but
    nothing that reaches a dictation shortcut sounds like a held note, while
    the flat clips that used to produce `????` do not.
    """
    if audio is None or audio.size == 0:
        return False
    if float(np.max(np.abs(audio))) < min_peak:
        return False
    return speech_modulation_db(audio, sample_rate) >= min_modulation_db


def has_words(text: str) -> bool:
    """True when a transcript contains at least one letter or digit."""
    return any(char.isalnum() for char in (text or ""))


def is_pasteable(text: str) -> bool:
    """Reject the junk the model emits when it hears noise.

    Besides empty and symbol-only output (`????`), this drops transcripts with
    replacement characters, control bytes or lone surrogates, which show up as
    mojibake in the log and in whatever window receives the paste.
    """
    if not has_words(text):
        return False
    for char in text:
        code = ord(char)
        if code == 0xFFFD or 0xD800 <= code <= 0xDFFF:
            return False
        if code < 32 and char not in "\n\r\t":
            return False
    return True


def _words_only(text: str) -> str:
    return " ".join(
        "".join(c if c.isalnum() else " " for c in (text or "").lower()).split()
    )


def echoes_context(text: str, context: str) -> bool:
    """True when the model repeated a sentence of its own prompt.

    On near-silent audio the model sometimes "transcribes" the system prompt
    ("This is a dictation session."). Only whole prompt sentences count, so
    saying a single hotword that appears in the prompt is never dropped.
    """
    spoken = _words_only(text)
    if not spoken or not context:
        return False
    sentences = [_words_only(s) for s in re.split(r"(?<=[.!?])\s+", context)]
    return spoken == _words_only(context) or spoken in sentences


def worker_python(setting: str = "") -> str:
    """The Python that runs WORKER_MODELS (default: .venv-t5 here)."""
    if setting:
        return setting if os.path.isabs(setting) else os.path.join(BASE_DIR, setting)
    if platform_support.IS_WINDOWS:
        return os.path.join(BASE_DIR, ".venv-t5", "Scripts", "python.exe")
    return os.path.join(BASE_DIR, ".venv-t5", "bin", "python")


def resolve_torch_device(preference: str, torch_module=None) -> str:
    """Map the torch_device setting to a device torch can actually use."""
    pref = str(preference or "auto").strip().lower()
    if pref not in ("auto", "gpu"):
        return pref
    if torch_module is None:
        import torch as torch_module
    for name in ("xpu", "cuda", "mps"):
        backend = getattr(torch_module, name, None)
        if name == "mps":
            backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        try:
            if backend is not None and backend.is_available():
                return name
        except Exception:
            continue
    return "cpu"


def model_load_kwargs(device: str) -> dict[str, Any]:
    """from_pretrained() arguments for a device: bf16 on a GPU, fp32 on CPU."""
    if device == "cpu":
        return {}
    import torch

    return {"device_map": device, "dtype": torch.bfloat16}


def looks_like_english(detected: str) -> bool:
    """True when the detected language is English, or was not reported."""
    name = (detected or "").strip().lower()
    return not name or name.startswith("english")


def normalize_transcript(text: str) -> str:
    """Tidy spacing and invisible characters without changing the words.

    ASR output occasionally carries zero-width characters, doubled spaces or a
    space before a full stop. NFC leaves Chinese punctuation and accented
    letters alone, unlike NFKC.
    """
    if not text:
        return ""
    clean = unicodedata.normalize("NFC", str(text))
    clean = clean.translate(_INVISIBLE)
    clean = _WHITESPACE.sub(" ", clean).strip()
    clean = _SPACE_BEFORE_PUNCT.sub(r"\1", clean)
    return clean


def parse_hotwords(raw: str) -> list[str]:
    """Parse hotwords.txt: one term per line, '#' starts a comment."""
    terms: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        term = line.split("#", 1)[0].strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def compose_context(terms: list[str], accent: str = "") -> str:
    """Build the system prompt used to bias transcription.

    `accent` is an optional sentence about the speaker. Saying that the
    speaker has, for example, a Singapore or Malaysian accent stops the
    model from correcting their grammar and helps it keep local particles.
    """
    accent = (accent or "").strip()
    sentences = ["This is a dictation session."]
    if accent:
        sentences.append(accent)
    if terms:
        sentences.append(
            "Use these domain words and spellings exactly when they are "
            "spoken: " + ", ".join(terms) + "."
        )
    if len(sentences) == 1:
        return ""
    return " ".join(sentences)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s"


def should_restore_clipboard(
    auto_clipboard: bool, previous: str | None, current: str | None, text: str
) -> bool:
    """Decide whether to put the old clipboard back after a paste.

    With auto_clipboard the transcript stays on the clipboard on purpose, so it
    can be pasted somewhere else without dictating again.
    """
    return not auto_clipboard and previous is not None and current == text


class ClipboardHistory:
    """The last few texts that passed through the clipboard, newest first."""

    def __init__(self, limit: int = 5):
        self.limit = max(1, min(20, int(limit)))
        self._items: list[str] = []
        self._lock = threading.Lock()

    def add(self, text: str) -> bool:
        """Record `text`; returns True when the history actually changed."""
        text = (text or "").strip()
        if not text:
            return False
        with self._lock:
            if self._items and self._items[0] == text:
                return False
            if text in self._items:
                self._items.remove(text)
            self._items.insert(0, text)
            del self._items[self.limit:]
        return True

    def items(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# ----------------------------------------------------------- autostart ------
def _pythonw_path() -> str:
    return platform_support.python_executable()


def autostart_command() -> str:
    """Command line used by the Windows Run key / .desktop / LaunchAgent."""
    return subprocess.list2cmdline(
        platform_support.autostart_command(os.path.join(BASE_DIR, "dictate.py"))
    )


def autostart_enabled() -> bool:
    try:
        return platform_support.autostart_enabled()
    except Exception as exc:
        log.debug("could not read autostart state: %s", exc)
        return False


def set_autostart(enabled: bool) -> bool:
    ok = platform_support.set_autostart(enabled)
    if ok:
        log.info(
            "autostart %s: %s",
            "enabled" if enabled else "disabled",
            autostart_command(),
        )
    else:
        log.error("could not update autostart")
    return ok


def acquire_single_instance(name: str = MUTEX_NAME):
    """Claim the single-instance lock; False if another copy already has it."""
    global _single_socket

    if sys.platform == "win32":
        handle = _kernel32.CreateMutexW(None, False, name)
        last_error = ctypes.get_last_error()
        if not handle:
            log.error("CreateMutexW failed (WinError %s)", last_error)
            return None
        if last_error == ERROR_ALREADY_EXISTS:
            _kernel32.CloseHandle(handle)
            return False
        return handle

    # A bound localhost socket is released automatically when the process
    # exits, which gives the same protection without platform APIs.
    port = 47000 + (zlib.crc32(os.path.abspath(BASE_DIR).encode()) % 1000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
    except OSError:
        sock.close()
        return False
    _single_socket = sock
    return sock


def release_single_instance(owner) -> None:
    global _single_socket
    if not owner:
        return
    try:
        if sys.platform == "win32":
            _kernel32.CloseHandle(owner)
        else:
            owner.close()
    except Exception:
        pass
    _single_socket = None


# ---------------------------------------------------------------- tray ------
def _dim(color: tuple[int, int, int], factor: float = 0.34) -> tuple[int, int, int]:
    return tuple(int(channel * factor) for channel in color)


def create_tray_image(state: str, size: int = 64) -> Image.Image:
    """Draw a two-lamp traffic light: green while busy, red while idle.

    Rendered at 4x and downsampled so the lamps stay smooth at the 16 px
    notification-area size Windows actually uses.
    """
    supersample = 4
    canvas = size * supersample
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    scale = canvas / 64.0

    def px(value: float) -> float:
        return value * scale

    active = STATE_LAMPS.get(state, "red")

    # Housing
    draw.rounded_rectangle(
        [px(13), px(0), px(51), px(64)],
        radius=px(11),
        fill=HOUSING,
        outline=RIM,
        width=max(1, int(px(2))),
    )

    for name, center_y, color in (
        ("red", 16.5, LAMP_RED),
        ("green", 47.5, LAMP_GREEN),
    ):
        center_x = px(32)
        center_y_px = px(center_y)
        radius = px(11.5)
        lit = active == name
        body = color if lit else _dim(color)

        # Socket rim, then the lens.
        draw.ellipse(
            [center_x - radius, center_y_px - radius,
             center_x + radius, center_y_px + radius],
            fill=RIM,
        )
        inner = radius - px(1.6)
        draw.ellipse(
            [center_x - inner, center_y_px - inner,
             center_x + inner, center_y_px + inner],
            fill=body,
        )
        if lit:
            # Small specular highlight so the lit lamp reads as glass.
            glow = inner * 0.34
            offset = inner * 0.5
            draw.ellipse(
                [center_x - offset - glow, center_y_px - offset - glow,
                 center_x - offset + glow, center_y_px - offset + glow],
                fill=(255, 255, 255, 115),
            )

    return img.resize((size, size), Image.LANCZOS)


class DictationTray:
    def __init__(self, app: "VoiceDictationApp"):
        self.app = app
        self._icon = None
        self._current_state = "loading"

    def _shortcut_action(self, value: str):
        def handler(_icon=None, _item=None):
            self.app.set_shortcut(value)

        return handler

    def _capture_shortcut(self, icon=None, item=None):
        self.app.begin_shortcut_capture()

    def _language_action(self, language: str):
        # pystray only accepts callbacks with zero, one or two parameters, so
        # the language has to be captured by a factory rather than a default
        # argument (a default raises the parameter count).
        def handler(_icon=None, _item=None):
            self.app.set_language(language)

        return handler

    def _language_checked(self, language: str):
        def handler(_item=None):
            return self.app.settings.language == language

        return handler

    def build_menu(self):
        import pystray

        language_items = [
            pystray.MenuItem(
                lang,
                self._language_action(lang),
                checked=self._language_checked(lang),
                radio=True,
            )
            for lang in TRAY_LANGUAGES
        ]

        items = [
            pystray.MenuItem(
                lambda _item: f"● {APP_NAME} ({self._current_state.upper()})",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda _item: f"Model: {str(self.app.settings.model_id).split('/')[-1]}"
                + (f" ({self.app.device.upper()})" if self.app.device else ""),
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda _item: "Resume Dictation" if self.app.paused else "Pause Dictation",
                self._toggle_pause,
                checked=lambda _item: self.app.paused,
            ),
            pystray.MenuItem(
                lambda _item: f"Shortcut: {self.app.shortcut.display()}",
                pystray.Menu(
                    pystray.MenuItem(
                        "Detect a new shortcut…",
                        self._capture_shortcut,
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem(
                        lambda _item: (
                            f"Current: {self.app.shortcut.display()}"
                        ),
                        None,
                        enabled=False,
                    ),
                    pystray.MenuItem(
                        "Side button 2 (Forward)",
                        self._shortcut_action("mouse:x2"),
                        checked=lambda _item: self.app.shortcut.canonical() == "mouse:x2",
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "Side button 1 (Back)",
                        self._shortcut_action("mouse:x1"),
                        checked=lambda _item: self.app.shortcut.canonical() == "mouse:x1",
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "Middle click",
                        self._shortcut_action("mouse:middle"),
                        checked=lambda _item: self.app.shortcut.canonical() == "mouse:middle",
                        radio=True,
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem(
                        "Reset to Side button 2",
                        self._shortcut_action("mouse:x2"),
                    ),
                ),
            ),
            pystray.MenuItem("Dictation language", pystray.Menu(*language_items)),
            pystray.MenuItem(
                "Clean up dictated text",
                self._toggle_cleanup,
                checked=lambda _item: self.app.cleanup_enabled,
            ),
            pystray.MenuItem(
                "Fix my English (local AI)",
                self._toggle_rewrite,
                checked=lambda _item: bool(self.app.settings.rewrite_enabled),
            ),
            pystray.MenuItem(
                "App context awareness",
                self._toggle_context_awareness,
                checked=lambda _item: bool(getattr(self.app.settings, "context_awareness", True)),
            ),
            pystray.MenuItem("English fixer model", pystray.Menu(*self._rewrite_model_items())),
            pystray.MenuItem(
                "Transcript panel",
                self._toggle_overlay,
                checked=lambda _item: bool(self.app.settings.overlay_enabled),
            ),
            pystray.MenuItem(
                "Expand / collapse transcript",
                self._expand_overlay,
                enabled=lambda _item: self.app.overlay is not None,
            ),
            pystray.MenuItem(
                "Clear transcript history",
                self._clear_overlay,
                enabled=lambda _item: self.app.overlay is not None,
            ),
            pystray.MenuItem(
                "Clipboard history (left-click the icon)",
                self._show_clipboard,
                default=True,
            ),
            pystray.MenuItem("Clear clipboard history", self._clear_clipboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Copy last transcript",
                self._copy_last,
                enabled=lambda _item: bool(self.app.last_text),
            ),
            pystray.MenuItem(
                "Re-paste last transcript",
                self._repaste_last,
                enabled=lambda _item: bool(self.app.last_text),
            ),
            pystray.MenuItem("Open Last Recording (WAV)", self._open_last_wav),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Reload model", self._reload_model),
            pystray.MenuItem("Open log file", self._open_log),
            pystray.MenuItem(
                platform_support.autostart_label(),
                self._toggle_autostart,
                checked=lambda _item: autostart_enabled(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Restart as Administrator",
                self._restart_admin,
                visible=lambda _item: (
                    platform_support.IS_WINDOWS and not self.app.is_elevated()
                ),
            ),
            pystray.MenuItem(f"Exit {APP_NAME}", self._on_exit),
        ]
        return pystray.Menu(*items)

    def start(self):
        if not self.app.settings.tray_enabled:
            log.info("tray disabled by config")
            return
        try:
            import pystray

            self._icon = pystray.Icon(
                "VoiceDictation",
                create_tray_image("loading"),
                f"{APP_NAME} - loading model...",
                self.build_menu(),
            )
            self._icon.run_detached()
            log.info("tray icon started")
            if platform_support.IS_WINDOWS:
                def _auto_promote():
                    try:
                        time.sleep(1.5)
                        platform_support.promote_tray_icon()
                    except Exception:
                        pass
                threading.Thread(target=_auto_promote, daemon=True, name="promote-tray").start()
        except Exception:
            log.exception("could not initialise the system tray icon")
            self._icon = None

    def set_state(self, state: str, tooltip_suffix: str = ""):
        self._current_state = state
        # Keep the transcript panel in step with every state change, including
        # the ones that talk to the tray directly (model loading and errors).
        self.app._sync_overlay(state)
        if not self._icon:
            return
        try:
            self._icon.icon = create_tray_image(state)
            tip = f"{APP_NAME}: {state.capitalize()}"
            if tooltip_suffix:
                tip += f" ({tooltip_suffix})"
            self._icon.title = tip
            self._icon.update_menu()
        except Exception:
            # The icon may already be stopping during shutdown.
            pass

    def _toggle_pause(self, icon=None, item=None):
        self.app.toggle_pause()

    def _toggle_cleanup(self, icon=None, item=None):
        enabled = not self.app.cleanup_enabled
        self.app.set_cleanup(enabled)

    def _toggle_rewrite(self, icon=None, item=None):
        self.app.set_rewrite(not bool(self.app.settings.rewrite_enabled))

    def _toggle_context_awareness(self, icon=None, item=None):
        self.app.set_context_awareness(
            not bool(getattr(self.app.settings, "context_awareness", True))
        )

    def _rewrite_model_items(self) -> list:
        import pystray

        choices = list(REWRITE_CHOICES)
        current = self.app.settings.rewrite_model
        if current not in {m for m, _ in choices}:
            choices.append((current, current.split("/")[-1]))

        def pick(model_id):
            return lambda icon=None, item=None: self.app.set_rewrite_model(model_id)

        def label(model_id, text):
            # Shown while the chosen model is still loading (Gemma: ~2 min).
            return lambda _item: (
                text + "  - loading..."
                if self.app._rewriter_loading == model_id else text
            )

        return [
            pystray.MenuItem(
                label(model_id, text),
                pick(model_id),
                checked=lambda _item, m=model_id: self.app.settings.rewrite_model == m,
                radio=True,
            )
            for model_id, text in choices
        ]

    def _toggle_overlay(self, icon=None, item=None):
        self.app.set_overlay_enabled(not self.app.settings.overlay_enabled)

    def _expand_overlay(self, icon=None, item=None):
        if self.app.overlay is not None:
            self.app.overlay.toggle()

    def _clear_overlay(self, icon=None, item=None):
        if self.app.overlay is not None:
            self.app.overlay.clear_history()

    def _show_clipboard(self, icon=None, item=None):
        self.app.show_clipboard_history()

    def _clear_clipboard(self, icon=None, item=None):
        self.app.clear_clipboard_history()

    def _toggle_autostart(self, icon=None, item=None):
        set_autostart(not autostart_enabled())
        if self._icon:
            try:
                self._icon.update_menu()
            except Exception:
                pass

    def _copy_last(self, icon=None, item=None):
        if self.app.last_text and self.app.copy_to_clipboard(self.app.last_text):
            log.info("last transcript copied to clipboard")

    def _repaste_last(self, icon=None, item=None):
        if self.app.last_text:
            self.app.paste_text(self.app.last_text)

    def _open_last_wav(self, icon=None, item=None):
        if os.path.exists(TMP_WAV):
            try:
                os.startfile(TMP_WAV)
            except Exception as exc:
                log.warning("could not open %s: %s", TMP_WAV, exc)
        else:
            log.info("no recording file yet")

    def _open_log(self, icon=None, item=None):
        if os.path.exists(LOG_PATH):
            try:
                os.startfile(LOG_PATH)
            except Exception as exc:
                log.warning("could not open %s: %s", LOG_PATH, exc)
        else:
            log.info("no log file yet")

    def _reload_model(self, icon=None, item=None):
        threading.Thread(
            target=self.app.load_model, kwargs={"force": True}, daemon=True
        ).start()

    def _restart_admin(self, icon=None, item=None):
        self.app.restart_as_admin()

    def _on_exit(self, icon=None, item=None):
        log.info("exit requested from the tray menu")
        self.app.stop()
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass


# ------------------------------------------------------------------- app ----
class VoiceDictationApp:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

        self.shortcut = shortcuts.try_parse(
            getattr(self.settings, "trigger_shortcut", ""),
            fallback=f"mouse:{self.settings.trigger}",
        )
        self.paused = False
        self.running = True
        self.cleanup_enabled = bool(self.settings.cleanup_enabled)
        self.last_text = ""
        self.model_error: str | None = None

        self._model = None
        self.device = ""
        self._rewriter: rewrite.Rewriter | None = None
        self._rewriter_lock = threading.Lock()
        self._rewriter_loading: str | None = None  # model id being loaded
        self._rewriter_failed: set[str] = set()     # not retried until re-chosen
        self._last_alternative = ""
        self._current_context = context_awareness.AppContext()
        # (time, text) of recent pastes: topic context for the English fixer.
        self._recent: list[tuple[float, str]] = []
        self._corrections_cache: tuple[float | None, list] = (None, [])
        self._model_lock = threading.Lock()
        self._model_last_attempt = 0.0
        self._supported_languages: dict[str, str] | None = None
        self._last_language = ""

        self._frames: list[np.ndarray] = []
        self._stream = None
        self._recording = False
        self._record_started_at = 0.0
        self._stream_error = False
        self._last_status_log = 0.0
        self._opening_stream = False
        self._busy = False

        self._rec_lock = threading.Lock()
        self._audio_open_lock = threading.Lock()
        self._job_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._kb = keyboard.Controller()

        self._hotword_cache: tuple[
            tuple[float | None, str] | None, str
        ] = (None, "")

        self.clipboard_history = ClipboardHistory(
            int(getattr(self.settings, "clipboard_history_size", 5))
        )
        self._clip_seen = ""

        # The transcript panel is created lazily (it imports tkinter) and is
        # only entered into `self.overlay` once it is actually usable.
        self.overlay = None
        if self.settings.overlay_enabled:
            try:
                from overlay import TranscriptOverlay

                self.overlay = TranscriptOverlay(self.settings, log=log)
            except Exception:
                log.exception("the transcript panel is unavailable")
                self.overlay = None

        self.tray = DictationTray(self)
        self.trigger = None
        self._capturing = False
        self._capture = None

    # -- state ------------------------------------------------------------
    def is_elevated(self) -> bool:
        if not platform_support.IS_WINDOWS:
            return False
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    @property
    def trigger_name(self) -> str:
        """Legacy name of a mouse shortcut (x1/x2), else its display text."""
        if self.shortcut.is_mouse:
            return self.shortcut.button
        return self.shortcut.display()

    def _idle_tooltip(self) -> str:
        return f"Hold {self.shortcut.display()} to dictate"

    def refresh_state(self, busy: bool | None = None) -> None:
        if not self.running:
            return
        if busy is not None:
            self._busy = busy
        if self.model_error:
            state, tooltip = "error", "Model failed - open log"
        elif self._busy:
            state, tooltip = "transcribing", "Transcribing..."
        elif self.paused:
            state, tooltip = "paused", "Dictation paused"
        elif self._recording:
            state, tooltip = "recording", "Recording speech..."
        elif self._model is None:
            state, tooltip = "loading", "Loading model..."
        else:
            state, tooltip = "idle", self._idle_tooltip()

        self.tray.set_state(state, tooltip)

    def _sync_overlay(self, state: str) -> None:
        """Mirror the app state into the transcript panel, if it is running."""
        overlay = self.overlay
        if overlay is None:
            return
        try:
            overlay.set_state(state, self._record_started_at or None)
        except Exception:
            log.debug("transcript panel state update failed", exc_info=True)

    def _add_overlay_entry(self, text: str) -> None:
        overlay = self.overlay
        if overlay is None or not text:
            return
        try:
            overlay.add_entry(text)
        except Exception:
            log.debug("could not add transcript to the panel", exc_info=True)

    def _notify_panel(self, message: str) -> None:
        """Show a single line in the panel when nothing was pasted."""
        overlay = self.overlay
        if overlay is None or not message:
            return
        try:
            overlay.notify(message)
        except Exception:
            log.debug("could not show a panel notice", exc_info=True)

    def show_clipboard_history(self) -> None:
        """Left-clicking the tray icon shows the recent clipboard items."""
        if self.overlay is None:
            self.set_overlay_enabled(True)
        items = self.clipboard_history.items()
        log.info("clipboard history opened (%d item(s))", len(items))
        if self.overlay is None:
            return
        try:
            self.overlay.show_clipboard(items)
        except Exception:
            log.debug("could not show the clipboard panel", exc_info=True)

    def clear_clipboard_history(self) -> None:
        self.clipboard_history.clear()
        log.info("clipboard history cleared")
        overlay = self.overlay
        if overlay is not None and overlay.clipboard_open:
            try:
                overlay.show_clipboard([])
            except Exception:
                pass

    def _clipboard_watcher(self) -> None:
        """Remember every text that lands on the clipboard.

        Polling is deliberate: a clipboard-format listener would need its own
        message-only window, and reading four times a second is invisible.
        """
        while self.running:
            time.sleep(0.6)
            if not self.running:
                return
            try:
                text = platform_support.get_clipboard()
            except Exception:
                continue
            if not isinstance(text, str) or not text.strip():
                self._clip_seen = ""
                continue
            if text == self._clip_seen:
                continue
            self._clip_seen = text
            self.clipboard_history.add(text)

    def set_overlay_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.settings.set("overlay_enabled", enabled)
        if not enabled:
            if self.overlay is not None:
                try:
                    self.overlay.stop()
                except Exception:
                    log.debug("could not stop the transcript panel", exc_info=True)
                self.overlay = None
            log.info("transcript panel disabled")
        elif self.overlay is None:
            try:
                from overlay import TranscriptOverlay

                self.overlay = TranscriptOverlay(self.settings, log=log)
                self.overlay.start()
                log.info("transcript panel enabled")
            except Exception:
                log.exception("could not start the transcript panel")
                self.overlay = None
        self.refresh_state()
        if self.tray._icon:
            try:
                self.tray._icon.update_menu()
            except Exception:
                pass

    def restart_as_admin(self):
        if not platform_support.IS_WINDOWS:
            return
        try:
            params = " ".join(f'"{arg}"' for arg in sys.argv)
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, params, None, 1
            )
            self.stop()
            if self.tray._icon:
                self.tray._icon.stop()
            sys.exit(0)
        except Exception as exc:
            log.error("could not relaunch as administrator: %s", exc)

    # -- shortcut ---------------------------------------------------------
    def _listener_active(self) -> bool:
        return self.running and not self.paused and not self._capturing

    def set_shortcut(self, value: str) -> bool:
        """Validate, persist and apply a new shortcut without a restart."""
        try:
            shortcut = shortcuts.parse(value)
        except ValueError as exc:
            log.warning("shortcut %r rejected: %s", value, exc)
            return False
        problem = shortcut.problem()
        if problem:
            log.warning("shortcut %r rejected: %s", value, problem)
            return False

        self.shortcut = shortcut
        self.settings.set("trigger_shortcut", shortcut.canonical())
        if shortcut.is_mouse and shortcut.button in ("x1", "x2"):
            # Keep the pre-2.0 key valid in case an older build is started.
            self.settings.set("trigger", shortcut.button)
        log.info("shortcut set to %s (%s)", shortcut.display(), shortcut.canonical())
        if self.trigger is not None:
            self.trigger.reset()
        self.refresh_state()
        if self.tray._icon:
            try:
                self.tray._icon.update_menu()
            except Exception:
                pass
        return True

    def set_trigger(self, trigger: str) -> None:
        """Backwards-compatible helper for the old x1/x2 config values."""
        if trigger in ("x1", "x2"):
            self.set_shortcut(f"mouse:{trigger}")

    def begin_shortcut_capture(self) -> None:
        if self._capturing:
            return
        if self.paused:
            log.info("resume dictation before recording a shortcut")
            return
        if self._recording:
            log.info("finish the current dictation before recording a shortcut")
            return

        if self.overlay is None:
            self.set_overlay_enabled(True)
        overlay = self.overlay
        if overlay is None:
            log.warning("the shortcut recorder needs the transcript panel")
            return

        from capture import ShortcutCapture

        self._capturing = True
        if self.trigger is not None:
            self.trigger.reset()
        overlay.on_capture_cancel = self.cancel_shortcut_capture
        overlay.show_capture("Waiting for a key, combination or click…")
        self._capture = ShortcutCapture(
            on_done=self._shortcut_captured,
            on_cancel=self._shortcut_cancelled,
            on_preview=self._shortcut_preview,
            log=log,
        )
        if not self._capture.start():
            self._capturing = False
            self._capture = None
            overlay.finish_capture(
                "Global input is not permitted — check the system settings", ok=False
            )
            log.warning("shortcut recorder could not read global input")

    def cancel_shortcut_capture(self) -> None:
        capture = self._capture
        if capture is not None:
            capture.cancel()
        else:
            self._end_capture()
            if self.overlay is not None:
                self.overlay.hide_capture()

    def _shortcut_preview(self, text: str) -> None:
        if self.overlay is not None and self._capturing:
            self.overlay.update_capture(text)

    def _shortcut_captured(self, shortcut: shortcuts.Shortcut) -> None:
        ok = self.set_shortcut(shortcut.canonical())
        text = shortcut.display() + (" is now the trigger" if ok else " is not usable")
        if self.overlay is not None:
            self.overlay.finish_capture(text, ok=ok)
        self._end_capture()

    def _shortcut_cancelled(self) -> None:
        if self.overlay is not None:
            self.overlay.hide_capture()
        log.info("shortcut recording cancelled")
        self._end_capture()

    def _end_capture(self) -> None:
        self._capturing = False
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.stop()
        if self.overlay is not None:
            self.overlay.on_capture_cancel = None
        if self.trigger is not None:
            self.trigger.reset()

    def set_language(self, language: str) -> None:
        self.settings.set("language", language)
        log.info("dictation language set to %s", language)
        self.refresh_state()

    def set_rewrite(self, enabled: bool) -> None:
        self.settings.set("rewrite_enabled", enabled)
        log.info("English rewrite %s", "enabled" if enabled else "disabled")
        if enabled:
            self._ensure_rewriter()
        self._refresh_menu()

    def set_context_awareness(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.settings.set("context_awareness", enabled)
        log.info("app context awareness %s", "enabled" if enabled else "disabled")
        self._refresh_menu()

    def set_rewrite_model(self, model_id: str) -> None:
        """Swap the English fixer. The current one keeps working until the new
        one is loaded, then it is freed."""
        self._rewriter_failed.discard(model_id)  # choosing it again = retry
        if model_id == self.settings.rewrite_model and model_id == getattr(
            self._rewriter, "model_id", None
        ):
            return
        self.settings.set("rewrite_model", model_id)
        log.info("English fixer model set to %s", model_id)
        self._ensure_rewriter()
        self._refresh_menu()

    def _refresh_menu(self) -> None:
        if self.tray._icon:
            try:
                self.tray._icon.update_menu()
            except Exception:
                pass

    def _make_rewriter(self, model_id: str) -> "rewrite.Rewriter":
        device = self.device or resolve_torch_device(
            getattr(self.settings, "torch_device", "auto")
        )
        if model_id in WORKER_MODELS:
            return rewrite.RemoteRewriter(
                model_id, device,
                python=worker_python(str(getattr(self.settings, "rewrite_worker_python", "") or "")),
                worker=WORKER_PATH,
                log_path=WORKER_LOG_PATH,
            )
        return rewrite.Rewriter(model_id, device)

    def _make_api_rewriter(self, model_id: str) -> "rewrite.OpenAIRewriter":
        api_base = str(getattr(self.settings, "rewrite_api_base", "") or "").strip()
        api_model = str(getattr(self.settings, "rewrite_api_model", "") or "").strip() or model_id
        api_key = str(getattr(self.settings, "rewrite_api_key", "") or "").strip()
        return rewrite.OpenAIRewriter(api_model, api_base, api_key)

    def _ensure_rewriter(self) -> None:
        """Start loading the chosen fixer in the background if it isn't the
        active one yet. Never blocks: Gemma 4 takes ~2 minutes to load."""
        if not self.settings.rewrite_enabled:
            return
        wanted = self.settings.rewrite_model
        with self._rewriter_lock:
            current = self._rewriter
            if current is not None and current.model_id == wanted and current.ready:
                return
            if self._rewriter_loading == wanted or wanted in self._rewriter_failed:
                return
            self._rewriter_loading = wanted
        threading.Thread(
            target=self._switch_rewriter, args=(wanted,), daemon=True, name="rewrite-load"
        ).start()

    def _switch_rewriter(self, wanted: str, fallback: bool = False) -> None:
        try:
            self._do_switch_rewriter(wanted, fallback)
        finally:
            self._refresh_menu()  # drop the "loading..." label

    def _do_switch_rewriter(self, wanted: str, fallback: bool = False) -> None:
        api_base = str(getattr(self.settings, "rewrite_api_base", "") or "").strip()
        using_api = bool(api_base) and not fallback
        log.info(
            "loading English fixer %s%s",
            wanted,
            " (fallback)" if fallback else (" (API)" if using_api else ""),
        )
        if not using_api and wanted in WORKER_MODELS:
            # Gemma 4 needs ~10 GB; with the speech model that fills the Arc's
            # shared 16 GB, so the current fixer must go first. Dictations in
            # the meantime get corrections.txt only.
            with self._rewriter_lock:
                old, self._rewriter = self._rewriter, None
            if old is not None:
                log.info("freeing %s to make room", old.model_id)
                old.close()
        new = self._make_api_rewriter(wanted) if using_api else self._make_rewriter(wanted)
        ok = new.load()
        old = None
        with self._rewriter_lock:
            if self._rewriter_loading == wanted:
                self._rewriter_loading = None
            still_wanted = fallback or self.settings.rewrite_model == wanted
            if ok and still_wanted:
                old, self._rewriter = self._rewriter, new
        if ok and not still_wanted:
            new.close()  # the user picked something else while it loaded
            return
        if ok:
            log.info("English fixer now %s", new.model_id)
            if old is not None and old is not new:
                old.close()
            return
        log.warning("English fixer %s failed: %s", wanted, new.error)
        self._rewriter_failed.add(wanted)
        current = self._rewriter
        if using_api and (current is None or not current.ready):
            # API endpoint unreachable (e.g. off LAN or llama-server stopped):
            # fall back to the local model cleanly.
            self._do_switch_rewriter(wanted, fallback=True)
        elif wanted != rewrite.DEFAULT_MODEL and (current is None or not current.ready):
            self._do_switch_rewriter(rewrite.DEFAULT_MODEL, fallback=True)

    def _active_rewriter(self) -> "rewrite.Rewriter | None":
        """The fixer to use now: the chosen one, or the previous one while the
        chosen one loads. None when nothing is ready (the raw text is pasted)."""
        self._ensure_rewriter()
        current = self._rewriter
        if current is not None and current.ready:
            return current
        return None

    def _load_rewriter(self) -> None:
        """Kept for callers that start the fixer at startup."""
        self._ensure_rewriter()

    def _hotword_terms(self) -> list[str]:
        try:
            with open(HOTWORDS_PATH, "r", encoding="utf-8") as handle:
                return parse_hotwords(handle.read())
        except OSError:
            return []

    def _corrections(self) -> list:
        try:
            mtime = os.path.getmtime(CORRECTIONS_PATH)
        except OSError:
            return []
        cached_mtime, pairs = self._corrections_cache
        if cached_mtime != mtime:
            try:
                with open(CORRECTIONS_PATH, "r", encoding="utf-8-sig") as handle:
                    pairs = rewrite.parse_corrections(handle.read())
            except OSError as exc:
                log.warning("could not read corrections: %s", exc)
                pairs = []
            self._corrections_cache = (mtime, pairs)
        return pairs

    def apply_corrections(self, text: str) -> str:
        pairs = self._corrections()
        if not text or not pairs:
            return text
        fixed = rewrite.apply_corrections(text, pairs)
        if fixed != text:
            log.info("corrections: %r", fixed)
        return fixed

    def recent_context(self, now: float | None = None) -> str:
        """The last two dictations from the past 3 minutes."""
        now = time.time() if now is None else now
        self._recent = [(t, s) for t, s in self._recent if now - t <= 180.0][-2:]
        return " ".join(s for _t, s in self._recent)

    def remember(self, text: str, now: float | None = None) -> None:
        self._recent.append((time.time() if now is None else now, text))
        self._recent = self._recent[-2:]

    def maybe_rewrite(
        self,
        text: str,
        alternative: str = "",
        app_context: context_awareness.AppContext | None = None,
    ) -> str:
        if not text or not self.settings.rewrite_enabled:
            return text
        rewriter = self._active_rewriter()
        if rewriter is None:
            log.info("English fixer still loading; pasting the transcript as heard")
            return text
        corrections = self._corrections()
        if alternative and corrections:
            alternative = rewrite.apply_corrections(alternative, corrections)
        terms = self._hotword_terms()
        style_hint = ""
        if app_context is not None and not app_context.sensitive:
            terms = context_awareness.merge_hotwords(terms, app_context.dynamic_terms)
            style_hint = context_awareness.style_hint_for_category(app_context.category)
        return rewriter.rewrite(
            text,
            accent=str(getattr(self.settings, "accent", "") or ""),
            terms=terms,
            alternative=alternative,
            recent=self.recent_context(),
            style_hint=style_hint,
        )

    def set_cleanup(self, enabled: bool) -> None:
        self.cleanup_enabled = enabled
        self.settings.set("cleanup_enabled", enabled)
        log.info("text cleanup %s", "enabled" if enabled else "disabled")
        if self.tray._icon:
            try:
                self.tray._icon.update_menu()
            except Exception:
                pass

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.trigger is not None:
            self.trigger.reset()
        if self.paused:
            self.close_audio_stream()
            log.info("dictation PAUSED")
            self.refresh_state()
            return
        if not self._open_audio_stream():
            self.paused = True
            log.warning("resume failed: microphone unavailable")
            self.tray.set_state("paused", "Microphone unavailable")
            return
        log.info("dictation RESUMED")
        self.refresh_state()

    # -- model ------------------------------------------------------------
    def load_model(self, force: bool = False):
        with self._model_lock:
            if self._model is not None and not force:
                return self._model
            if force:
                self._model = None
                self.model_error = None

            # Avoid a 5-15s retry on every single dictation after a failure.
            now = time.time()
            if self.model_error and not force and now - self._model_last_attempt < 30:
                return None
            self._model_last_attempt = now

            self.tray.set_state("loading", "Loading model...")
            started = time.time()
            try:
                from qwen_asr import Qwen3ASRModel

                device = resolve_torch_device(
                    getattr(self.settings, "torch_device", "auto")
                )
                log.info(
                    "loading ASR model %s on %s ...", self.settings.model_id, device
                )
                try:
                    model = Qwen3ASRModel.from_pretrained(
                        self.settings.model_id, **model_load_kwargs(device)
                    )
                except Exception:
                    if device == "cpu":
                        raise
                    # A broken GPU driver or runtime should cost speed, not
                    # the whole app: fall back to the CPU.
                    log.exception("could not load on %s; falling back to cpu", device)
                    device = "cpu"
                    model = Qwen3ASRModel.from_pretrained(self.settings.model_id)
                self.device = device
            except Exception as exc:
                self.model_error = f"{type(exc).__name__}: {exc}"
                log.exception("could not load the ASR model")
                self.tray.set_state("error", "Model failed to load - open log")
                return None

            self._model = model
            self.model_error = None
            try:
                self._supported_languages = {
                    name.lower(): name for name in model.get_supported_languages()
                }
            except Exception:
                self._supported_languages = {}

            log.info("model ready on %s in %.1fs", self.device, time.time() - started)
            self._warm_up(model)
            self.refresh_state()
            self._ensure_rewriter()
            return self._model

    def _warm_up(self, model) -> None:
        """Run one silent second so the first real dictation is not the slow one.

        The first inference after a load pays for lazy kernel setup and thread
        pool warm-up, which shows up as an extra second of latency exactly when
        the user is waiting for their first paste. On a GPU the first call also
        compiles kernels for the prompt and audio shapes (several seconds), so
        warm up with the real prompt and a dictation-sized clip, twice.
        """
        sample_rate = int(self.settings.sample_rate)
        rng = np.random.default_rng(0)
        clip = (rng.standard_normal(sample_rate * 3) * 1e-3).astype(np.float32)
        language = self._effective_language(model)
        context = self._context()
        started = time.time()
        plain = compose_context(self._hotword_terms() if self.settings.use_hotwords else [])
        try:
            for audio in (clip, clip[: sample_rate * 2]):
                model.transcribe(
                    audio=(audio, sample_rate), context=context, language=language
                )
                if getattr(self.settings, "second_hearing", True):
                    model.transcribe(
                        audio=[(audio, sample_rate)] * 2,
                        context=[context, plain],
                        language=[language] * 2,
                    )
        except Exception as exc:
            log.debug("model warm-up failed: %s", exc)
            return
        log.info("model warm-up in %.1fs", time.time() - started)

    def _effective_language(self, model) -> str | None:
        name = str(self.settings.language or "Auto").strip()
        if not name or name.lower() == "auto":
            return None
        if self._supported_languages is None:
            try:
                self._supported_languages = {
                    item.lower(): item for item in model.get_supported_languages()
                }
            except Exception:
                self._supported_languages = {}
        resolved = self._supported_languages.get(name.lower())
        if resolved is None:
            log.warning("language %r unsupported by this model; using auto", name)
        return resolved

    def _context(self, extra_terms: tuple[str, ...] | list[str] = ()) -> str:
        accent = str(getattr(self.settings, "accent", "") or "").strip()
        terms: list[str] = []
        mtime: float | None = None
        extra_key = tuple(extra_terms)

        if self.settings.use_hotwords:
            try:
                mtime = os.path.getmtime(HOTWORDS_PATH)
            except OSError:
                mtime = None
            if mtime is not None and not extra_key:
                cached_key, cached_context = self._hotword_cache
                if cached_key == (mtime, accent):
                    return cached_context
            if mtime is not None:
                try:
                    with open(HOTWORDS_PATH, "r", encoding="utf-8") as handle:
                        terms = parse_hotwords(handle.read())
                except Exception as exc:
                    log.warning("could not read hotwords: %s", exc)
                    terms = []

        if extra_key:
            terms = context_awareness.merge_hotwords(terms, extra_key)
        context = compose_context(terms, accent)
        if mtime is not None and not extra_key:
            self._hotword_cache = ((mtime, accent), context)
        if terms:
            log.debug("hotwords in use: %s", ", ".join(terms))
        return context

    def transcribe(
        self,
        audio: np.ndarray,
        app_context: context_awareness.AppContext | None = None,
    ) -> str:
        model = self.load_model()
        if model is None:
            return ""
        sample_rate = int(self.settings.sample_rate)
        language = self._effective_language(model)
        extra_terms = (
            app_context.dynamic_terms
            if app_context is not None and not app_context.sensitive
            else ()
        )
        text, detected = self._run_asr(model, audio, sample_rate, language, extra_terms)

        if not is_pasteable(text):
            # Accented or clipped speech often comes back empty, as symbols or
            # in the wrong language, because the language detector guessed
            # wrong. One retry in a known language is cheap: it only runs when
            # the first attempt produced nothing usable.
            fallback = self._fallback_language()
            allow_auto = language is not None and fallback == language
            # English only: for any other pinned language a detection retry
            # could never be accepted (it is kept only when it stays
            # English), so do not spend a second inference on it.
            if allow_auto and not looks_like_english(language):
                allow_auto = False
            if allow_auto:
                # A pinned language (English, say) stops the detector from
                # flipping, but it also leaves nothing to retry in: the same
                # pinned call returns the same empty result. Let detection
                # try once and keep that only when it still says English,
                # so a noise clip cannot paste Chinese over your cursor.
                fallback = None
            if fallback != language or allow_auto:
                log.info(
                    "retrying %s after an unusable %s result",
                    "with language detection"
                    if allow_auto
                    else f"in {fallback}",
                    detected or "language-detected",
                )
                retry, detected_retry = self._run_asr(
                    model, audio, sample_rate, fallback, extra_terms
                )
                if is_pasteable(retry) and (
                    not allow_auto or looks_like_english(detected_retry)
                ):
                    text, detected = retry, detected_retry

        if is_pasteable(text) and detected:
            self._last_language = detected
            if language is None:
                log.info("detected language: %s", detected)
        return text

    def _run_asr(
        self,
        model,
        audio: np.ndarray,
        sample_rate: int,
        language,
        extra_terms: tuple[str, ...] | list[str] = (),
    ):
        context = self._context(extra_terms)
        # Second hearing: the same audio again with only the hotwords as
        # prompt, batched into the same GPU call (+~0.15s). Where the two
        # disagree the speaker was unclear, which the English fixer uses.
        plain_terms = self._hotword_terms() if self.settings.use_hotwords else []
        if extra_terms:
            plain_terms = context_awareness.merge_hotwords(plain_terms, extra_terms)
        plain = compose_context(plain_terms)
        second = bool(getattr(self.settings, "second_hearing", True)) and plain != context
        if second:
            results = model.transcribe(
                audio=[(audio, sample_rate)] * 2,
                context=[context, plain],
                language=[language] * 2,
            )
        else:
            results = model.transcribe(
                audio=(audio, sample_rate),
                context=context,
                language=language,
            )
        self._last_alternative = ""
        if not results:
            return "", ""
        if second and len(results) > 1:
            alternative = (results[1].text or "").strip()
            if not echoes_context(alternative, plain) and is_pasteable(alternative):
                self._last_alternative = normalize_transcript(alternative)
        result = results[0]
        text = (result.text or "").strip()
        if echoes_context(text, context):
            log.info("dropped a transcript that only repeated the prompt: %r", text)
            text = ""
        return text, str(getattr(result, "language", "") or "")

    def _fallback_language(self) -> str | None:
        """The language to retry in: the last good one, else English."""
        supported = self._supported_languages or {}
        for name in (self._last_language, "English"):
            resolved = supported.get(str(name or "").lower())
            if resolved:
                return resolved
        return None

    # -- audio ------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        # The stream stays open so the mouse hook never waits for the audio
        # device. Only collect data while a recording is active.
        if status:
            # "input overflow" only means this block arrived late, which is
            # normal while the model loads and pegs every core. It is not a
            # reason to tear the device down; just record it now and then.
            now = time.time()
            if now - self._last_status_log > 10.0:
                self._last_status_log = now
                log.warning("audio stream status: %s", status)
        if not self._recording:
            return
        try:
            with self._rec_lock:
                if self._recording:
                    self._frames.append(indata.copy())
        except Exception:
            log.exception("audio callback failed")

    def _on_stream_finished(self):
        """PortAudio calls this when the stream stops without being asked,
        which is how a removed or reassigned microphone shows up."""
        if self.running and not self.paused:
            log.warning("microphone stream stopped unexpectedly; will reopen")
            self._stream_error = True

    def _open_audio_stream(self) -> bool:
        """Open the microphone once and keep it warm.

        Opening an input stream takes 200-300 ms on this laptop, which is too
        slow to do inside a low-level mouse hook (Windows removes hooks that
        block) and would clip the first word of every recording.
        """
        with self._audio_open_lock:
            if self._stream is not None:
                return True
            if self._opening_stream:
                return False
            self._opening_stream = True
            try:
                stream = sd.InputStream(
                    samplerate=int(self.settings.sample_rate),
                    channels=1,
                    dtype="float32",
                    device=self.settings.device,
                    # 64 ms blocks. The default is 0, which lets PortAudio pick
                    # tiny blocks; the log showed "input overflow" whenever the
                    # model pegged the CPU, and a larger block rides that out.
                    blocksize=1024,
                    callback=self._audio_callback,
                    finished_callback=self._on_stream_finished,
                )
                stream.start()
            except Exception as exc:
                log.error("could not open the microphone: %s", exc)
                return False
            finally:
                self._opening_stream = False
            self._stream = stream
            self._stream_error = False
            log.info("microphone stream open and ready")
            return True

    def _open_stream_in_background(self) -> None:
        def work():
            if self._open_audio_stream():
                self.refresh_state()

        threading.Thread(target=work, daemon=True, name="mic-open").start()

    def close_audio_stream(self) -> None:
        """Stop capture and release the microphone (used by pause/exit)."""
        with self._rec_lock:
            self._recording = False
            self._frames = []
        with self._audio_open_lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
                log.info("microphone stream closed")

    def resume_capture(self) -> bool:
        return self._open_audio_stream()

    def start_recording(self) -> None:
        if self.paused or not self.running:
            return
        with self._rec_lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True
            self._record_started_at = time.time()
            if getattr(self.settings, "context_awareness", True):
                self._current_context = context_awareness.capture_active_context()
            else:
                self._current_context = context_awareness.AppContext()
        # Never block the mouse hook: if the mic is not open yet, open it on a
        # worker thread and start collecting as soon as frames arrive.
        if self._stream is None:
            self._open_stream_in_background()
        if self._current_context.active:
            log.debug(
                "active context: app=%s category=%s sensitive=%s terms=%s",
                self._current_context.process_name,
                self._current_context.category,
                self._current_context.sensitive,
                ", ".join(self._current_context.dynamic_terms),
            )
        log.info("recording started")
        self.refresh_state()

    def stop_recording_and_process(self) -> None:
        with self._rec_lock:
            if not self._recording:
                return
            self._recording = False
            frames, self._frames = self._frames, []
            app_context = self._current_context
            self._current_context = context_awareness.AppContext()
        self._record_started_at = 0.0

        if not frames:
            log.info("no audio captured (microphone not ready)")
            self._notify_panel("Microphone not ready")
            self.refresh_state()
            return

        audio = np.concatenate(frames).reshape(-1).astype(np.float32, copy=False)
        raw_seconds = len(audio) / float(self.settings.sample_rate)

        if self.settings.trim_silence:
            audio = trim_silence(audio, int(self.settings.sample_rate))
            audio = np.ascontiguousarray(audio, dtype=np.float32)

        seconds = len(audio) / float(self.settings.sample_rate)
        verdict = classify_duration(
            seconds, float(self.settings.min_seconds), float(self.settings.max_seconds)
        )

        if verdict == "short":
            log.info(
                "ignored a %.2fs clip (minimum is %.1fs); hold the button while speaking",
                seconds,
                float(self.settings.min_seconds),
            )
            self._notify_panel("Hold the shortcut while you speak")
            self.refresh_state()
            return

        if verdict == "long":
            cap = int(float(self.settings.max_seconds) * float(self.settings.sample_rate))
            log.warning(
                "clip was %s, trimming to the %.0fs maximum",
                format_duration(raw_seconds),
                float(self.settings.max_seconds),
            )
            audio = np.ascontiguousarray(audio[:cap], dtype=np.float32)
            seconds = len(audio) / float(self.settings.sample_rate)

        if seconds < float(self.settings.min_seconds):
            self.refresh_state()
            return

        if not looks_like_speech(audio, int(self.settings.sample_rate)):
            log.info(
                "ignored a flat %.2fs clip (no speech modulation); "
                "check the microphone",
                seconds,
            )
            self._notify_panel("Heard nothing. Check the microphone")
            self.refresh_state()
            return

        if getattr(self.settings, "auto_gain", True):
            audio, gain = normalize_loudness(audio)
            if gain > 1.0:
                log.info("boosted quiet audio x%.1f", gain)

        log.info("queued %s of audio for transcription", format_duration(seconds))
        self._job_queue.put((audio, seconds, app_context))

    def _transcription_worker(self) -> None:
        """Serialize inference so rapid dictations cannot overlap."""
        while True:
            item = self._job_queue.get()
            if item is None:
                self._job_queue.task_done()
                return
            if not self.running:
                self._job_queue.task_done()
                continue

            if len(item) == 3:
                audio, seconds, app_context = item
            else:
                audio, seconds = item
                app_context = context_awareness.AppContext()
            try:
                self.refresh_state(busy=True)
                if self.settings.keep_recording:
                    try:
                        sf.write(TMP_WAV, audio, int(self.settings.sample_rate))
                    except Exception as exc:
                        log.warning("could not save %s: %s", TMP_WAV, exc)

                log.info("transcribing %s of audio ...", format_duration(seconds))
                # Load outside the timer so the logged latency is inference
                # only, not a cold model load.
                if self._model is None and self.model_error is None:
                    self.load_model()
                started = time.time()
                raw_text = normalize_transcript(self.transcribe(audio, app_context))
                text = postprocess_text(raw_text) if self.cleanup_enabled else raw_text
                elapsed = time.time() - started

                if raw_text != text:
                    log.debug("raw transcript: %r", raw_text)
                log.info("transcribed in %.2fs: %r", elapsed, text)

                if text and is_pasteable(text):
                    alternative = self._last_alternative
                    if alternative and alternative != raw_text:
                        log.info("second hearing: %r", alternative)
                    text = self.apply_corrections(text)
                    if self.settings.rewrite_enabled:
                        heard = text
                        text = normalize_transcript(
                            self.maybe_rewrite(heard, alternative, app_context)
                        )
                        if text != heard:
                            log.info("fixed English: %r", text)
                    self.remember(text)

                dropped = False
                if text and not is_pasteable(text):
                    log.info("discarded a transcript that is not clean text")
                    text = ""
                    dropped = True
                    self._notify_panel("Nothing clear enough to paste")

                if text and self.running:
                    self.last_text = text
                    self._add_overlay_entry(text)
                    self.paste_text(text)
                elif not text:
                    log.info("no speech detected")
                    if not dropped:
                        self._notify_panel("Heard nothing. Check the microphone")
            except Exception:
                log.exception("transcription failed")
            finally:
                self._job_queue.task_done()
                self.refresh_state(busy=False)

    # -- clipboard / paste ------------------------------------------------
    def copy_to_clipboard(self, text: str, attempts: int = 5) -> bool:
        for attempt in range(attempts):
            if platform_support.set_clipboard(text):
                return True
            if attempt == attempts - 1:
                log.error("could not write to the clipboard")
            time.sleep(0.05)
        return False

    def _clipboard_get(self) -> str | None:
        return platform_support.get_clipboard()

    def paste_text(self, text: str) -> None:
        if not text:
            return
        previous = self._clipboard_get()
        if not self.copy_to_clipboard(text):
            return

        modifier_name = platform_support.paste_modifier()
        modifier = getattr(keyboard.Key, modifier_name, keyboard.Key.ctrl)
        key = platform_support.paste_key()
        try:
            time.sleep(float(self.settings.paste_delay))
            with self._kb.pressed(modifier):
                self._kb.press(key)
                self._kb.release(key)
            log.info("pasted %d characters", len(text))
            time.sleep(float(self.settings.clipboard_restore_delay))
        except Exception:
            log.exception("could not send the paste chord")
        finally:
            # Only restore if our text is still on the clipboard, so a paste
            # that lands late never gets overwritten by the old contents.
            if should_restore_clipboard(
                bool(getattr(self.settings, "auto_clipboard", True)),
                previous,
                self._clipboard_get(),
                text,
            ):
                self.copy_to_clipboard(previous)

    # -- trigger ----------------------------------------------------------
    def create_trigger(self):
        return create_trigger(
            get_shortcut=lambda: self.shortcut,
            on_start=self.start_recording,
            on_stop=self.stop_recording_and_process,
            active=self._listener_active,
            log=log,
            suppress_mouse=lambda: bool(self.settings.suppress_browser_nav),
        )

    # -- watchdog ---------------------------------------------------------
    def _watchdog(self) -> None:
        """Auto-stop long recordings and recover a dropped microphone."""
        last_reopen = 0.0
        while self.running:
            time.sleep(0.25)
            if not self.running:
                return
            try:
                if (
                    self._recording
                    and self._record_started_at
                    and time.time() - self._record_started_at
                    > float(self.settings.max_seconds) + 2
                ):
                    log.warning("maximum recording length reached; stopping")
                    threading.Thread(
                        target=self.stop_recording_and_process, daemon=True
                    ).start()

                if self._stream_error:
                    self._stream_error = False
                    log.warning("audio stream reported an error; reopening")
                    self.close_audio_stream()
                    last_reopen = time.time()
                    if not self.paused:
                        self._open_stream_in_background()
                elif (
                    self._stream is None
                    and not self.paused
                    and not self._opening_stream
                    and time.time() - last_reopen > 3.0
                ):
                    last_reopen = time.time()
                    self._open_stream_in_background()
            except Exception:
                log.exception("watchdog iteration failed")

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        log.info("=" * 62)
        log.info("%s %s", APP_NAME, VERSION)
        log.info("model      : %s", self.settings.model_id)
        log.info("language   : %s", self.settings.language)
        log.info("trigger    : %s (%s)", self.shortcut.display(), self.shortcut.canonical())
        log.info("platform   : %s", platform_support.SYSTEM)
        log.info("clipboard  : %s", platform_support.clipboard_backend_name())
        log.info("cleanup    : %s", "on" if self.cleanup_enabled else "off")
        log.info(
            "rewrite    : %s",
            self.settings.rewrite_model if self.settings.rewrite_enabled else "off",
        )
        log.info("min/max    : %.1fs / %.0fs", self.settings.min_seconds, self.settings.max_seconds)
        log.info("panel      : %s", "on" if self.overlay is not None else "off")
        log.info("clipboard  : auto=%s history=%d", "on" if self.settings.auto_clipboard else "off", self.clipboard_history.limit)
        log.info("=" * 62)

        self.tray.start()

        if self.overlay is not None:
            try:
                self.overlay.start()
                log.info("transcript panel started")
            except Exception:
                log.exception("could not start the transcript panel")
                self.overlay = None

        if not self._open_audio_stream():
            log.warning("microphone not available yet; will retry on first press")

        self._worker = threading.Thread(
            target=self._transcription_worker, daemon=True, name="asr-worker"
        )
        self._worker.start()
        threading.Thread(target=self._watchdog, daemon=True, name="watchdog").start()
        threading.Thread(target=self.load_model, daemon=True, name="model-load").start()
        threading.Thread(
            target=self._clipboard_watcher, daemon=True, name="clipboard-watch"
        ).start()

        try:
            self.trigger = self.create_trigger()
            self.trigger.run()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        try:
            self._job_queue.put(None)
        except Exception:
            pass
        self.close_audio_stream()
        if self.overlay is not None:
            try:
                self.overlay.stop()
            except Exception:
                log.debug("could not stop the transcript panel", exc_info=True)
            self.overlay = None
        if self.trigger is not None:
            try:
                self.trigger.stop()
            except Exception:
                pass
            self.trigger = None
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        log.info("voice dictation stopped")


# ---------------------------------------------------------------- doctor ----
def run_doctor(settings: Settings, full: bool = False) -> int:
    """Check the things that most often break a fresh install."""
    problems = 0
    print(f"{APP_NAME} {VERSION} - environment check")
    print("-" * 62)
    print(f"python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"config      : {settings.path}")
    print(f"platform    : {platform_support.SYSTEM}")
    try:
        hw = platform_support.system_hardware_report()
        if hw.get("cpu"):
            print(f"cpu         : {hw['cpu']} ({hw['cores']} cores)")
        if hw.get("ram_total_gb") is not None:
            free_s = f", {hw['ram_free_gb']} GB free" if hw.get("ram_free_gb") is not None else ""
            print(f"memory      : {hw['ram_total_gb']} GB total{free_s}")
        if hw.get("disk_free_gb") is not None:
            print(f"disk space  : {hw['disk_free_gb']} GB free")
        if hw.get("gpus"):
            for g in hw["gpus"]:
                print(f"gpu         : {g}")
        if hw.get("npus"):
            for n in hw["npus"]:
                print(f"npu         : {n}")
        if platform_support.IS_WINDOWS and hw.get("vc_redist") is not None:
            print(f"  {'ok ' if hw['vc_redist'] else 'warn'} vc_redist     {'installed' if hw['vc_redist'] else 'not found (needed for torch XPU)'}")
    except Exception:
        pass
    print(f"shortcut    : {shortcuts.describe(settings.trigger_shortcut)}")
    print(f"clipboard   : {platform_support.clipboard_backend_name()}")
    for status, text in platform_support.permission_report():
        print(f"  {status:<4} {text}")
        if status == "fail":
            problems += 1

    for module in (
        "numpy",
        "sounddevice",
        "soundfile",
        "PIL",
        "pynput",
        "pyperclip",
        "pystray",
        "tkinter",
        "qwen_asr",
        "torch",
    ):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            print(f"  ok  {module:<12} {version}")
        except Exception as exc:
            problems += 1
            print(f"  FAIL {module:<12} {exc}")

    try:
        device = resolve_torch_device(getattr(settings, "torch_device", "auto"))
        print(f"  {'ok ' if device != 'cpu' else '-- '} compute      {device}")
        if device == "cpu":
            print("       no GPU visible to torch; transcription will be slow.")
            print("       Intel Arc: pip install torch==2.15.0 --index-url "
                  "https://download.pytorch.org/whl/xpu")
    except Exception as exc:
        print(f"  FAIL compute      {exc}")

    try:
        default_in = sd.default.device[0]
        name = sd.query_devices(default_in)["name"]
        stream = sd.InputStream(
            samplerate=int(settings.sample_rate), channels=1, dtype="float32"
        )
        stream.start()
        stream.stop()
        stream.close()
        print(f"  ok  microphone   #{default_in} {name} @ {settings.sample_rate} Hz")
    except Exception as exc:
        problems += 1
        print(f"  FAIL microphone   {exc}")

    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    model_dir = "models--" + str(settings.model_id).replace("/", "--")
    cached = os.path.isdir(os.path.join(cache_root, model_dir))
    print(f"  {'ok ' if cached else 'FAIL'} model cache  {model_dir}")
    if not cached:
        problems += 1
        print("       download it once with: huggingface-cli download " + str(settings.model_id))

    print(f"  {'ok ' if autostart_enabled() else '-- '} autostart")
    print(f"  ok  hotwords     {HOTWORDS_PATH} ({'present' if os.path.exists(HOTWORDS_PATH) else 'not created yet'})")
    print(f"  ok  log          {LOG_PATH}")

    if full and cached:
        print("-" * 62)
        print("loading the model and transcribing silence (this takes a few seconds)...")
        try:
            from qwen_asr import Qwen3ASRModel

            device = resolve_torch_device(getattr(settings, "torch_device", "auto"))
            started = time.time()
            model = Qwen3ASRModel.from_pretrained(
                settings.model_id, **model_load_kwargs(device)
            )
            print(f"  ok  model loaded on {device} in {time.time() - started:.1f}s")
            silence = np.zeros(int(settings.sample_rate * 1.5), dtype=np.float32)
            for label in ("first", "second"):
                started = time.time()
                model.transcribe(audio=(silence, int(settings.sample_rate)))
                print(f"  ok  {label} inference in {time.time() - started:.2f}s")
        except Exception as exc:
            problems += 1
            print(f"  FAIL inference: {exc}")

    print("-" * 62)
    if problems:
        print(f"{problems} problem(s) found.")
    else:
        print("Everything looks good.")
    return 1 if problems else 0


# ------------------------------------------------------------------ main ----
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local voice dictation.")
    parser.add_argument("--config", default=CONFIG_PATH, help="path to config.json")
    parser.add_argument("--doctor", action="store_true", help="check the environment")
    parser.add_argument(
        "--full", action="store_true", help="with --doctor, also load the model"
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--set-shortcut",
        metavar="KEYS",
        help='save a trigger without the tray, e.g. "ctrl+alt+j" or "mouse:x2"',
    )
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose, console=args.doctor or args.verbose)
    settings = load_settings(args.config)

    if args.set_shortcut:
        try:
            shortcut = shortcuts.parse(args.set_shortcut)
        except ValueError as exc:
            print(f"Invalid shortcut: {exc}")
            return 2
        problem = shortcut.problem()
        if problem:
            print(f"Invalid shortcut: {problem}")
            return 2
        settings.set("trigger_shortcut", shortcut.canonical())
        if shortcut.is_mouse and shortcut.button in ("x1", "x2"):
            settings.set("trigger", shortcut.button)
        print(f"Shortcut saved: {shortcut.display()} ({shortcut.canonical()})")
        return 0

    if args.doctor:
        return run_doctor(settings, full=args.full)

    owner = acquire_single_instance()
    if owner is False:
        message = f"{APP_NAME} is already running (look for the tray icon)."
        log.warning(message)
        if sys.stderr is None and platform_support.IS_WINDOWS:
            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x40)
        else:
            print(message)
        return 0
    if owner is None:
        log.error("could not create the single-instance lock; continuing anyway")

    app = VoiceDictationApp(settings)
    try:
        app.start()
    finally:
        release_single_instance(owner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
