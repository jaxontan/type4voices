"""Small platform differences, in one place.

Everything that is genuinely OS specific lives here: the paste chord, the
clipboard tools, the non-activating panel flags, the work area, autostart and
the permission hints a fresh install needs.

Windows is the tested platform. The macOS and Linux branches are written
against the documented APIs but need verification on real hardware.
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys

SYSTEM = "windows" if sys.platform == "win32" else "darwin" if sys.platform == "darwin" else "linux"
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "macos"
IS_LINUX = SYSTEM == "linux"

AUTOSTART_LABEL = "com.local.voicedictation"
AUTOSTART_NAME = "voice-dictation.desktop"


# ------------------------------------------------------------------ basics --
def ui_font() -> str:
    return {
        "windows": "Segoe UI",
        "macos": "SF Pro Text",
        "linux": "DejaVu Sans",
    }.get(SYSTEM, "TkDefaultFont")


def paste_modifier() -> str:
    """The modifier in the paste chord: Ctrl+V or Cmd+V."""
    return "cmd" if IS_MACOS else "ctrl"


def paste_key() -> str:
    return "v"


def python_executable() -> str:
    """Interpreter to use for autostart entries and relaunches."""
    exe = os.path.abspath(sys.executable)
    if IS_WINDOWS:
        candidate = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.exists(candidate):
            return candidate
    return exe


# --------------------------------------------------------------- clipboard --
def _linux_backend() -> str | None:
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return "wl-clipboard"
    if shutil.which("xclip"):
        return "xclip"
    if shutil.which("xsel"):
        return "xsel"
    return None


def clipboard_backend_name() -> str:
    """Which tool will be used, for --doctor and log messages."""
    if IS_LINUX:
        return _linux_backend() or "none"
    return "pyperclip"


def _run_tool(command: list[str], data: bytes | None = None) -> bytes | None:
    try:
        result = subprocess.run(
            command,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None


def get_clipboard() -> str | None:
    if IS_LINUX:
        backend = _linux_backend()
        if backend == "wl-clipboard":
            raw = _run_tool(["wl-paste", "--no-newline"])
        elif backend == "xclip":
            raw = _run_tool(["xclip", "-selection", "clipboard", "-o"])
        elif backend == "xsel":
            raw = _run_tool(["xsel", "--clipboard", "--output"])
        else:
            raw = None
        if raw is not None:
            return raw.decode("utf-8", errors="replace")
    try:
        import pyperclip

        return pyperclip.paste()
    except Exception:
        return None


def set_clipboard(text: str) -> bool:
    if IS_LINUX:
        backend = _linux_backend()
        payload = text.encode("utf-8")
        if backend == "wl-clipboard":
            return _run_tool(["wl-copy"], payload) is not None
        if backend == "xclip":
            return _run_tool(["xclip", "-selection", "clipboard", "-i"], payload) is not None
        if backend == "xsel":
            return _run_tool(["xsel", "--clipboard", "--input"], payload) is not None
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        return False


# -------------------------------------------------------------------- panel --
GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2


def prepare_panel(root) -> int:
    """Make the Tk panel floating and non-activating.

    Returns the native window handle on Windows, otherwise 0.
    """
    if IS_WINDOWS:
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            hwnd = root.winfo_id()
            parent = user32.GetParent(hwnd) or hwnd
            for handle in {hwnd, parent}:
                ex = user32.GetWindowLongW(handle, GWL_EXSTYLE)
                user32.SetWindowLongW(
                    handle,
                    GWL_EXSTYLE,
                    ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                )
            try:
                dwm = ctypes.WinDLL("dwmapi")
                value = ctypes.c_int(DWMWCP_ROUND)
                dwm.DwmSetWindowAttribute(
                    ctypes.c_void_p(parent),
                    DWMWA_WINDOW_CORNER_PREFERENCE,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
            except Exception:
                pass
            return int(parent)
        except Exception:
            return 0

    if IS_MACOS:
        _macos_prepare_panel(root)
        return 0

    # X11: a dock window is shown on every desktop and normally refuses focus.
    try:
        root.attributes("-type", "dock")
    except Exception:
        pass
    return 0


def _macos_prepare_panel(root) -> None:
    try:
        from AppKit import (  # type: ignore
            NSApp,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowStyleMaskNonactivatingPanel,
        )

        width = max(1, root.winfo_width())
        height = max(1, root.winfo_height())
        for window in NSApp.windows():
            frame = window.frame()
            if int(frame.size.width) != width or int(frame.size.height) != height:
                continue
            window.setStyleMask_(
                window.styleMask() | NSWindowStyleMaskNonactivatingPanel
            )
            # Float above normal windows, follow the user across Spaces, and
            # stay visible over full-screen apps.
            window.setLevel_(3)
            window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
            window.setHidesOnDeactivate_(False)
    except Exception:
        # Without PyObjC the panel still works; it may take focus on a click.
        pass


def screen_work_area(root=None) -> tuple[int, int, int, int]:
    """Desktop area excluding menu bar / taskbar, in pixels."""
    if IS_WINDOWS:
        try:
            user32 = ctypes.windll.user32

            class RECT(ctypes.Structure):
                _fields_ = (
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                )

            rect = RECT()
            if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                return rect.left, rect.top, rect.right, rect.bottom
            return 0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) - 48
        except Exception:
            pass

    if IS_MACOS:
        try:
            from AppKit import NSScreen  # type: ignore

            screen = NSScreen.mainScreen()
            frame = screen.frame()
            visible = screen.visibleFrame()
            total_height = int(frame.size.height)
            left = int(visible.origin.x)
            width = int(visible.size.width)
            height = int(visible.size.height)
            top = total_height - (int(visible.origin.y) + height)
            return left, top, left + width, top + height
        except Exception:
            pass

    if IS_LINUX:
        area = _x11_work_area()
        if area:
            return area

    if root is not None:
        try:
            return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight() - 48
        except Exception:
            pass
    try:
        user32 = ctypes.windll.user32
        return 0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) - 48
    except Exception:
        return 0, 0, 1920, 1032


def _x11_work_area() -> tuple[int, int, int, int] | None:
    if not shutil.which("xprop"):
        return None
    raw = _run_tool(["xprop", "-root", "_NET_WORKAREA"])
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    match = re.search(r"=\s*([\d\s,]+)", text)
    if not match:
        return None
    try:
        values = [int(part) for part in re.split(r"[,\s]+", match.group(1).strip()) if part]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    x, y, width, height = values[:4]
    if width <= 0 or height <= 0:
        return None
    return x, y, x + width, y + height


# ---------------------------------------------------------------- autostart --
def _windows_run_key():
    import winreg

    return winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"


def autostart_command(script: str | None = None) -> list[str]:
    script = script or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictate.py")
    return [python_executable(), script]


def autostart_label() -> str:
    """Tray wording for "start when I log in"."""
    if IS_MACOS:
        return "Start at Login"
    if IS_LINUX:
        return "Start with the desktop"
    return "Start with Windows"


def _plist_path() -> str:
    return os.path.expanduser(
        f"~/Library/LaunchAgents/{AUTOSTART_LABEL}.plist"
    )


def _desktop_path() -> str:
    return os.path.expanduser(f"~/.config/autostart/{AUTOSTART_NAME}")


def autostart_enabled() -> bool:
    if IS_WINDOWS:
        try:
            import winreg

            root, path = _windows_run_key()
            with winreg.OpenKey(root, path) as key:
                value, _ = winreg.QueryValueEx(key, "VoiceDictation")
            return bool(value)
        except FileNotFoundError:
            return False
        except OSError:
            return False
    target = _plist_path() if IS_MACOS else _desktop_path()
    return os.path.exists(target)


def set_autostart(enabled: bool) -> bool:
    if IS_WINDOWS:
        try:
            import winreg

            root, path = _windows_run_key()
            command = subprocess.list2cmdline(autostart_command())
            with winreg.OpenKey(root, path, 0, winreg.KEY_SET_VALUE) as key:
                if enabled:
                    winreg.SetValueEx(key, "VoiceDictation", 0, winreg.REG_SZ, command)
                else:
                    try:
                        winreg.DeleteValue(key, "VoiceDictation")
                    except FileNotFoundError:
                        pass
            return True
        except OSError:
            return False

    try:
        if IS_MACOS:
            path = _plist_path()
            if not enabled:
                if os.path.exists(path):
                    os.remove(path)
                return True
            os.makedirs(os.path.dirname(path), exist_ok=True)
            command = autostart_command()
            arguments = "".join(
                f"        <string>{part}</string>\n" for part in command
            )
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                    '<plist version="1.0">\n<dict>\n'
                    f"    <key>Label</key>\n    <string>{AUTOSTART_LABEL}</string>\n"
                    "    <key>ProgramArguments</key>\n    <array>\n"
                    f"{arguments}"
                    "    </array>\n"
                    "    <key>RunAtLoad</key>\n    <true/>\n"
                    "    <key>ProcessType</key>\n    <string>Interactive</string>\n"
                    "</dict>\n</plist>\n"
                )
            return True

        path = _desktop_path()
        if not enabled:
            if os.path.exists(path):
                os.remove(path)
            return True
        os.makedirs(os.path.dirname(path), exist_ok=True)
        command = subprocess.list2cmdline(autostart_command())
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "[Desktop Entry]\n"
                "Type=Application\n"
                "Name=Voice Dictation\n"
                f"Exec={command}\n"
                "Terminal=false\n"
                "X-GNOME-Autostart-enabled=true\n"
            )
        return True
    except Exception:
        return False


# -------------------------------------------------------------- permissions --
def permission_report() -> list[tuple[str, str]]:
    """(status, text) lines for --doctor: "ok", "warn" or "fail"."""
    lines: list[tuple[str, str]] = []

    if IS_MACOS:
        try:
            import Quartz  # type: ignore

            trusted = bool(Quartz.AXIsProcessTrusted())
            lines.append(
                (
                    "ok" if trusted else "warn",
                    "Accessibility (keyboard/mouse hooks): "
                    + ("granted" if trusted else "not granted - System Settings > Privacy & Security > Accessibility"),
                )
            )
        except Exception as exc:
            lines.append(("warn", f"Accessibility check unavailable: {exc}"))
        try:
            import Quartz  # type: ignore

            listen = bool(Quartz.CGPreflightListenEventAccess())
            lines.append(
                (
                    "ok" if listen else "warn",
                    "Input Monitoring: "
                    + ("granted" if listen else "not granted - System Settings > Privacy & Security > Input Monitoring"),
                )
            )
        except Exception:
            pass
        lines.append(("ok", "Microphone: macOS prompts on first recording; allow it in Privacy & Security > Microphone"))
        lines.append(("warn", "Universal Clipboard (Handoff) can copy text to other Apple devices"))

    if IS_LINUX:
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
            lines.append(
                ("warn", "Wayland session: global hotkeys and the tray are restricted; log in with X11/Xorg")
            )
        else:
            lines.append(("ok", "X11 session: global shortcuts available"))
        backend = clipboard_backend_name()
        lines.append(
            (
                "ok" if backend != "none" else "warn",
                "Clipboard tool: " + backend
                + ("" if backend != "none" else " - install xclip or wl-clipboard"),
            )
        )

    return lines



