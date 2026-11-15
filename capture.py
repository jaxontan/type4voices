"""Record the next key, combination or mouse click as the dictation shortcut.

The recorder listens globally, so it does not need a focused dialog. The app
shows progress in the transcript panel and asks this class to stop when a
valid shortcut arrives.

On Windows every keyboard event is suppressed while recording, so trying F8
or Ctrl+Alt+J does not also trigger something in the app behind the panel.
macOS and Linux listeners cannot suppress events reliably; the panel is
non-activating, so the only side effect is that a rejected key may type.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from pynput import keyboard, mouse

import shortcuts
from trigger import pynput_token


class ShortcutCapture:
    def __init__(
        self,
        on_done: Callable[[shortcuts.Shortcut], None],
        on_cancel: Callable[[], None],
        on_preview: Callable[[str], None] | None = None,
        log=None,
        ignore_for: float = 0.4,
    ):
        self.on_done = on_done
        self.on_cancel = on_cancel
        self.on_preview = on_preview or (lambda _text: None)
        self.log = log
        self.ignore_for = ignore_for

        self._started_at = 0.0
        self._done = False
        self._modifiers: set[str] = set()
        self._pressed: set[str] = set()
        self._keyboard_listener: keyboard.Listener | None = None
        self._mouse_listener: mouse.Listener | None = None
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> bool:
        self._started_at = time.time()
        self._done = False
        kwargs = {}
        if sys.platform == "win32":
            kwargs["win32_event_filter"] = self._win32_filter
        try:
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
                **kwargs,
            )
            self._keyboard_listener.start()
        except Exception as exc:
            if self.log:
                self.log.warning("could not capture keyboard input: %s", exc)
            self._keyboard_listener = None
        try:
            self._mouse_listener = mouse.Listener(on_click=self._on_click)
            self._mouse_listener.start()
        except Exception as exc:
            if self.log:
                self.log.warning("could not capture mouse input: %s", exc)
            self._mouse_listener = None

        if self._keyboard_listener is None and self._mouse_listener is None:
            self._stop_listeners()
            return False
        return True

    def stop(self) -> None:
        """Stop listening without reporting a result (app shutdown)."""
        self._done = True
        self._stop_listeners()

    def cancel(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._stop_listeners()
        self.on_cancel()

    def _stop_listeners(self) -> None:
        for listener in (self._keyboard_listener, self._mouse_listener):
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
        self._keyboard_listener = None
        self._mouse_listener = None

    def _ignored(self) -> bool:
        return time.time() - self._started_at < self.ignore_for

    # -- Windows suppression ---------------------------------------------
    def _win32_filter(self, msg, data):
        # False tells pynput to swallow the event. Keep delivering it to the
        # callbacks, but never to the application behind the panel.
        return not self._done

    # -- keyboard ---------------------------------------------------------
    def _on_key_press(self, key) -> None:
        if self._done or self._ignored():
            return
        token = pynput_token(key)
        if not token:
            return

        if token == "esc":
            self.cancel()
            return

        if token in self._pressed:
            return
        self._pressed.add(token)

        if token in shortcuts.MODIFIERS:
            self._modifiers.add(token)
            if self._modifiers:
                parts = [
                    shortcuts.modifier_label(item)
                    for item in sorted(self._modifiers, key=shortcuts.MODIFIER_ORDER.index)
                ]
                self.on_preview("+".join(parts) + "+…")
            return

        keys = sorted(self._modifiers, key=shortcuts.MODIFIER_ORDER.index) + [token]
        try:
            candidate = shortcuts.build(keys)
        except ValueError:
            return
        problem = candidate.problem()
        if problem:
            self.on_preview(problem)
            return
        self._finish(candidate)

    def _on_key_release(self, key) -> None:
        token = pynput_token(key)
        if not token:
            return
        self._pressed.discard(token)
        if token in shortcuts.MODIFIERS:
            self._modifiers.discard(token)
            if not self._modifiers and not self._done:
                self.on_preview("Waiting for a key, combination or click…")

    # -- mouse ------------------------------------------------------------
    def _on_click(self, x, y, button, pressed) -> None:
        if not pressed or self._done or self._ignored():
            return
        name = getattr(button, "name", "")
        if name not in ("x1", "x2", "middle"):
            return
        self._finish(shortcuts.mouse(name))

    # -- finish -----------------------------------------------------------
    def _finish(self, shortcut: shortcuts.Shortcut) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        # Stopping a listener from its own callback can deadlock on some
        # backends, so do it on a separate thread.
        threading.Thread(target=self._stop_listeners, daemon=True).start()
        self.on_done(shortcut)
