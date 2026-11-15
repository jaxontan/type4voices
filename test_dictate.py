"""Offline tests for the pure logic in dictate.py and text_cleanup.py.

Run with:  .venv\\Scripts\\python.exe -m unittest -v test_dictate
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dictate
import overlay
import platform_support
import shortcuts
from text_cleanup import postprocess_text


class DurationTests(unittest.TestCase):
    def test_classify_duration(self):
        self.assertEqual(dictate.classify_duration(0.4, 1.0, 120.0), "short")
        self.assertEqual(dictate.classify_duration(1.0, 1.0, 120.0), "ok")
        self.assertEqual(dictate.classify_duration(119.9, 1.0, 120.0), "ok")
        self.assertEqual(dictate.classify_duration(120.1, 1.0, 120.0), "long")

    def test_format_duration(self):
        self.assertEqual(dictate.format_duration(0.5), "0.5s")
        self.assertEqual(dictate.format_duration(59.4), "59.4s")
        self.assertEqual(dictate.format_duration(60), "1m00s")
        self.assertEqual(dictate.format_duration(125), "2m05s")


class TrimSilenceTests(unittest.TestCase):
    SR = 16000

    def test_trims_leading_and_trailing_silence(self):
        tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(self.SR) / self.SR)).astype(np.float32)
        audio = np.concatenate(
            [np.zeros(self.SR, np.float32), tone, np.zeros(self.SR, np.float32)]
        )
        trimmed = dictate.trim_silence(audio, self.SR)
        self.assertLess(len(trimmed), len(audio))
        self.assertGreaterEqual(len(trimmed), len(tone) * 0.95)
        self.assertLessEqual(len(trimmed), len(tone) * 1.5)

    def test_all_silence_is_unchanged(self):
        audio = np.zeros(self.SR, np.float32)
        self.assertEqual(len(dictate.trim_silence(audio, self.SR)), len(audio))

    def test_short_and_empty_input_is_unchanged(self):
        empty = np.zeros(0, np.float32)
        self.assertEqual(len(dictate.trim_silence(empty, self.SR)), 0)
        tiny = np.full(10, 0.1, np.float32)
        self.assertEqual(len(dictate.trim_silence(tiny, self.SR)), 10)

    def test_keeps_quiet_speech_relative_to_peak(self):
        quiet = (0.01 * np.sin(2 * np.pi * 200 * np.arange(self.SR) / self.SR)).astype(np.float32)
        audio = np.concatenate([np.zeros(self.SR, np.float32), quiet])
        trimmed = dictate.trim_silence(audio, self.SR)
        self.assertGreaterEqual(len(trimmed), len(quiet) * 0.95)


class SpeechGateTests(unittest.TestCase):
    SR = 16000

    @staticmethod
    def syllables(scale: float = 0.15) -> np.ndarray:
        """Four 120 ms blobs with gaps, the way speech hits a frame meter."""
        audio = np.zeros(2 * SpeechGateTests.SR, np.float32)
        step = int(0.3 * SpeechGateTests.SR)
        length = int(0.12 * SpeechGateTests.SR)
        wave = scale * np.sin(2 * np.pi * 180 * np.arange(length) / SpeechGateTests.SR)
        for index in range(4):
            start = int(0.15 * SpeechGateTests.SR) + index * step
            audio[start : start + length] = wave.astype(np.float32)
        return audio

    def test_silence_is_not_speech(self):
        self.assertFalse(dictate.looks_like_speech(np.zeros(self.SR, np.float32), self.SR))

    def test_empty_is_not_speech(self):
        self.assertFalse(dictate.looks_like_speech(np.zeros(0, np.float32), self.SR))

    def test_steady_noise_is_not_speech(self):
        noise = np.random.default_rng(1).normal(0, 0.003, self.SR).astype(np.float32)
        self.assertLess(dictate.speech_modulation_db(noise, self.SR), 3.0)
        self.assertFalse(dictate.looks_like_speech(noise, self.SR))

    def test_faint_hiss_is_not_speech(self):
        noise = np.random.default_rng(2).normal(0, 1e-5, self.SR).astype(np.float32)
        self.assertFalse(dictate.looks_like_speech(noise, self.SR))

    def test_syllables_count_as_speech(self):
        audio = self.syllables()
        self.assertGreaterEqual(dictate.speech_modulation_db(audio, self.SR), 3.0)
        self.assertTrue(dictate.looks_like_speech(audio, self.SR))

    def test_quiet_syllables_still_pass(self):
        # A quiet room and a mic at 10% gain should not be mistaken for noise.
        self.assertTrue(dictate.looks_like_speech(self.syllables(0.02), self.SR))


class TranscriptGateTests(unittest.TestCase):
    def test_has_words_accepts_speech_and_numbers(self):
        self.assertTrue(dictate.has_words("Hello"))
        self.assertTrue(dictate.has_words("4:30"))
        self.assertTrue(dictate.has_words("你好"))
        self.assertTrue(dictate.has_words("ok."))

    def test_has_words_rejects_empty_and_symbols(self):
        self.assertFalse(dictate.has_words(""))
        self.assertFalse(dictate.has_words("   "))
        self.assertFalse(dictate.has_words("????"))
        self.assertFalse(dictate.has_words("... !!! ??"))
        self.assertFalse(dictate.has_words(None))

    def test_pasteable_accepts_plain_text(self):
        self.assertTrue(dictate.is_pasteable("The meeting is at 4:30."))
        self.assertTrue(dictate.is_pasteable("你好，世界。"))

    def test_pasteable_rejects_symbols_and_mojibake(self):
        self.assertFalse(dictate.is_pasteable(""))
        self.assertFalse(dictate.is_pasteable("????"))
        self.assertFalse(dictate.is_pasteable("r-\ufffdo%?\ufffd\ufffd"))
        self.assertFalse(dictate.is_pasteable("bad\x07text"))


class HotwordTests(unittest.TestCase):
    def test_parse_hotwords_handles_comments_blanks_and_duplicates(self):
        raw = "OpenCode\n\n# a comment\n  PyTorch  \nopencode\nQwen3-ASR # trailing\n"
        self.assertEqual(
            dictate.parse_hotwords(raw), ["OpenCode", "PyTorch", "Qwen3-ASR"]
        )

    def test_parse_hotwords_empty(self):
        self.assertEqual(dictate.parse_hotwords(""), [])
        self.assertEqual(dictate.parse_hotwords("# only comments\n"), [])

    def test_compose_context(self):
        self.assertEqual(dictate.compose_context([]), "")
        context = dictate.compose_context(["OpenCode", "PyTorch"])
        self.assertIn("OpenCode, PyTorch", context)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_creates_defaults(self):
        settings = dictate.load_settings(self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(settings.model_id, dictate.DEFAULTS["model_id"])
        self.assertEqual(settings.min_seconds, 0.5)
        self.assertGreaterEqual(settings.min_seconds, 0.2)

    def test_values_are_coerced_and_clamped(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "min_seconds": "2.5",
                    "max_seconds": 1,
                    "trigger": "X1",
                    "cleanup_enabled": "no",
                    "device": 3,
                },
                handle,
            )
        settings = dictate.load_settings(self.path)
        self.assertEqual(settings.min_seconds, 2.5)
        self.assertGreater(settings.max_seconds, settings.min_seconds)
        self.assertEqual(settings.trigger, "x1")
        self.assertFalse(settings.cleanup_enabled)
        self.assertEqual(settings.device, 3)

    def test_invalid_trigger_falls_back(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"trigger": "middle"}, handle)
        self.assertEqual(dictate.load_settings(self.path).trigger, "x2")

    def test_unknown_keys_are_ignored(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"nonsense": 1}, handle)
        settings = dictate.load_settings(self.path)
        self.assertFalse(hasattr(settings, "nonsense"))
        self.assertEqual(settings.model_id, dictate.DEFAULTS["model_id"])

    def test_broken_json_falls_back_to_defaults(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        settings = dictate.load_settings(self.path)
        self.assertEqual(settings.min_seconds, 0.5)

    def test_save_round_trip(self):
        settings = dictate.load_settings(self.path)
        settings.set("language", "English")
        reloaded = dictate.load_settings(self.path)
        self.assertEqual(reloaded.language, "English")


class TrayImageTests(unittest.TestCase):
    def test_every_state_produces_an_rgba_image(self):
        for state in dictate.STATES:
            image = dictate.create_tray_image(state)
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.mode, "RGBA")

    def test_idle_and_recording_are_visibly_different(self):
        idle = dictate.create_tray_image("idle").convert("RGB")
        recording = dictate.create_tray_image("recording").convert("RGB")
        self.assertNotEqual(list(idle.getdata()), list(recording.getdata()))

    def test_only_red_and_green_lamps_are_ever_lit(self):
        # The active lamp must be one of the two signal colours, never a
        # third colour (amber, gray, ...).
        for state in dictate.STATES:
            active = dictate.STATE_LAMPS[state]
            self.assertIn(active, (None, "red", "green"))


class ShortcutTests(unittest.TestCase):
    def test_parse_mouse_buttons(self):
        self.assertEqual(shortcuts.parse("mouse:x2").canonical(), "mouse:x2")
        self.assertEqual(shortcuts.parse("mouse:x1").canonical(), "mouse:x1")
        self.assertEqual(shortcuts.parse("mouse:middle").canonical(), "mouse:middle")
        # Version 1 values still load.
        self.assertEqual(shortcuts.parse("x2").canonical(), "mouse:x2")

    def test_parse_combo_normalizes_aliases_and_order(self):
        shortcut = shortcuts.parse("alt+control+J")
        self.assertEqual(shortcut.canonical(), "ctrl+alt+j")
        self.assertTrue(shortcut.matches({"ctrl", "alt", "j"}))
        self.assertFalse(shortcut.matches({"ctrl", "alt", "k"}))
        self.assertFalse(shortcut.matches({"ctrl", "j"}))

    def test_function_keys_and_named_keys(self):
        self.assertEqual(shortcuts.parse("F8").canonical(), "f8")
        self.assertEqual(shortcuts.parse("ctrl+space").canonical(), "ctrl+space")
        with self.assertRaises(ValueError):
            shortcuts.parse("hyper+j")
        with self.assertRaises(ValueError):
            shortcuts.parse("ctrl+alt")

    def test_bare_letters_are_rejected_but_function_keys_are_fine(self):
        self.assertIsNotNone(shortcuts.parse("j").problem())
        self.assertIsNotNone(shortcuts.parse("space").problem())
        self.assertIsNone(shortcuts.parse("f8").problem())
        self.assertIsNone(shortcuts.parse("ctrl+alt+j").problem())
        self.assertIsNone(shortcuts.parse("mouse:x2").problem())

    def test_build_from_captured_keys(self):
        shortcut = shortcuts.build(["shift", "j", "ctrl"])
        self.assertEqual(shortcut.canonical(), "ctrl+shift+j")
        with self.assertRaises(ValueError):
            shortcuts.build(["ctrl", "alt"])

    def test_display_differs_for_windows_and_macos(self):
        shortcut = shortcuts.parse("cmd+shift+f9")
        self.assertEqual(shortcut.display("macos"), "Shift+Cmd+F9")
        self.assertEqual(shortcut.display("windows"), "Shift+Win+F9")
        self.assertEqual(
            shortcuts.parse("mouse:x2").display(), "Side button 2 (Forward)"
        )

    def test_windows_virtual_keys(self):
        self.assertEqual(shortcuts.vk_to_token(0x41), "a")
        self.assertEqual(shortcuts.vk_to_token(0x70), "f1")
        self.assertEqual(shortcuts.vk_to_token(0xA2), "ctrl")
        self.assertIsNone(shortcuts.vk_to_token(0xBA))  # ';' is not supported


class ShortcutMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, data):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def test_old_trigger_becomes_a_shortcut(self):
        self._write({"trigger": "x1"})
        settings = dictate.load_settings(self.path)
        self.assertEqual(settings.trigger, "x1")
        self.assertEqual(settings.trigger_shortcut, "mouse:x1")

    def test_invalid_shortcut_falls_back(self):
        self._write({"trigger_shortcut": "not+a+key"})
        settings = dictate.load_settings(self.path)
        self.assertEqual(settings.trigger_shortcut, "mouse:x2")

    def test_keyboard_shortcut_round_trips(self):
        self._write({"trigger_shortcut": "ctrl+alt+j"})
        settings = dictate.load_settings(self.path)
        self.assertEqual(settings.trigger_shortcut, "ctrl+alt+j")


class PlatformTests(unittest.TestCase):
    def test_clipboard_backend_is_named(self):
        self.assertIn(
            platform_support.clipboard_backend_name(),
            ("pyperclip", "wl-clipboard", "xclip", "xsel", "none"),
        )

    def test_paste_chord_matches_the_platform(self):
        expected = "cmd" if sys.platform == "darwin" else "ctrl"
        self.assertEqual(platform_support.paste_modifier(), expected)

    def test_autostart_command_mentions_the_script(self):
        self.assertIn("dictate.py", platform_support.autostart_command()[-1])


class AppShortcutTests(unittest.TestCase):
    """The app applies a shortcut without restarting."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"overlay_enabled": False, "tray_enabled": False}, handle)
        self.app = dictate.VoiceDictationApp(dictate.load_settings(self.path))

    def tearDown(self):
        self.app.running = False
        self.tmp.cleanup()

    def test_set_shortcut_updates_state_and_config(self):
        self.assertTrue(self.app.set_shortcut("ctrl+alt+j"))
        self.assertEqual(self.app.shortcut.canonical(), "ctrl+alt+j")
        saved = dictate.load_settings(self.path)
        self.assertEqual(saved.trigger_shortcut, "ctrl+alt+j")

    def test_mouse_shortcut_keeps_legacy_key_in_sync(self):
        self.assertTrue(self.app.set_shortcut("mouse:x1"))
        self.assertEqual(dictate.load_settings(self.path).trigger, "x1")

    def test_invalid_shortcuts_are_refused(self):
        self.assertFalse(self.app.set_shortcut("j"))
        self.assertFalse(self.app.set_shortcut("not+a+key"))
        self.assertEqual(self.app.shortcut.canonical(), "mouse:x2")

    def test_capture_suppresses_the_trigger(self):
        self.assertTrue(self.app._listener_active())
        self.app._capturing = True
        self.assertFalse(self.app._listener_active())
        self.app._capturing = False
        self.app.paused = True
        self.assertFalse(self.app._listener_active())


class AutostartTests(unittest.TestCase):
    def test_command_points_at_pythonw_and_the_script(self):
        command = dictate.autostart_command()
        self.assertIn("dictate.py", command)
        self.assertIn("pythonw.exe", command.lower())


class TextCleanupIntegrationTests(unittest.TestCase):
    def test_model_style_output_is_tidied(self):
        self.assertEqual(
            postprocess_text("Ah, I think. Um, this is a test."),
            "I think. This is a test.",
        )
        self.assertEqual(
            postprocess_text("Open github dot com period"), "Open github.com."
        )


class NormalizeTests(unittest.TestCase):
    def test_collapses_spaces_and_trims(self):
        self.assertEqual(dictate.normalize_transcript("  hello   world  "), "hello world")

    def test_removes_zero_width_characters(self):
        self.assertEqual(dictate.normalize_transcript("he\u200bllo\ufeff"), "hello")

    def test_tightens_space_before_punctuation(self):
        self.assertEqual(dictate.normalize_transcript("Hello , world ."), "Hello, world.")

    def test_leaves_chinese_text_and_punctuation_alone(self):
        self.assertEqual(dictate.normalize_transcript(" 你好，世界。 "), "你好，世界。")

    def test_empty_text(self):
        self.assertEqual(dictate.normalize_transcript(""), "")
        self.assertEqual(dictate.normalize_transcript(None), "")


class _FakeResult:
    def __init__(self, text, language):
        self.text = text
        self.language = language


class _FakeModel:
    """Returns per-language canned answers and records the calls."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def transcribe(self, audio=None, context="", language=None):
        self.calls.append(language)
        text = self.answers.get(language)
        if not text:
            return []
        return [_FakeResult(text, language or "English")]

    def get_supported_languages(self):
        return ["English", "Chinese"]


class LanguageRetryTests(unittest.TestCase):
    """A failed first pass gets one retry in a known language."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = dictate.load_settings(os.path.join(self.tmp.name, "config.json"))
        self.app = dictate.VoiceDictationApp(self.settings)
        self.app.running = False
        self.app.overlay = None
        self.app._supported_languages = {"english": "English", "chinese": "Chinese"}
        self.audio = np.zeros(16000, np.float32)

    def tearDown(self):
        self.app.running = False
        self.tmp.cleanup()

    def test_retries_in_english_when_auto_returns_nothing(self):
        self.settings.set("language", "Auto", persist=False)
        model = _FakeModel({None: "", "English": "Hello there"})
        self.app._model = model
        self.assertEqual(self.app.transcribe(self.audio), "Hello there")
        self.assertEqual(model.calls, [None, "English"])

    def test_retries_when_auto_returns_only_symbols(self):
        self.settings.set("language", "Auto", persist=False)
        model = _FakeModel({None: "????", "English": "Send it"})
        self.app._model = model
        self.assertEqual(self.app.transcribe(self.audio), "Send it")
        self.assertEqual(model.calls, [None, "English"])

    def test_no_retry_when_the_first_pass_is_usable(self):
        self.settings.set("language", "Auto", persist=False)
        model = _FakeModel({None: "Hello there"})
        self.app._model = model
        self.assertEqual(self.app.transcribe(self.audio), "Hello there")
        self.assertEqual(model.calls, [None])

    def test_retry_uses_the_last_good_language(self):
        self.settings.set("language", "Auto", persist=False)
        self.app._last_language = "Chinese"
        model = _FakeModel({"Chinese": "你好", "English": "Hello"})
        self.app._model = model
        self.assertEqual(self.app.transcribe(self.audio), "你好")
        self.assertEqual(model.calls, [None, "Chinese"])

    def test_pinned_language_does_not_retry_in_itself(self):
        self.settings.set("language", "Chinese", persist=False)
        model = _FakeModel({"Chinese": "", "English": "Hello"})
        self.app._model = model
        self.app._last_language = "Chinese"
        self.assertEqual(self.app.transcribe(self.audio), "")
        self.assertEqual(model.calls, ["Chinese"])

    def test_empty_both_passes_stays_empty(self):
        self.settings.set("language", "Auto", persist=False)
        model = _FakeModel({None: "", "English": ""})
        self.app._model = model
        self.assertEqual(self.app.transcribe(self.audio), "")
        self.assertEqual(model.calls, [None, "English"])


class StreamRecoveryTests(unittest.TestCase):
    """Regression tests for the audio-stream status handling.

    PortAudio reports transient 'input overflow' while the model loads and
    every core is busy. That must not tear the device down; only a stream
    that actually stopped should trigger a reopen.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        settings = dictate.load_settings(os.path.join(self.tmp.name, "config.json"))
        self.app = dictate.VoiceDictationApp(settings)
        self.app.running = True

    def tearDown(self):
        self.app.running = False
        self.tmp.cleanup()

    def test_overflow_status_does_not_mark_the_stream_as_failed(self):
        frame = np.zeros((160, 1), np.float32)
        self.app._audio_callback(frame, 160, None, "input overflow")
        self.assertFalse(self.app._stream_error)

    def test_finished_callback_marks_the_stream_as_failed(self):
        self.app._on_stream_finished()
        self.assertTrue(self.app._stream_error)

    def test_finished_callback_ignores_a_deliberate_pause(self):
        self.app.paused = True
        self.app._on_stream_finished()
        self.assertFalse(self.app._stream_error)


class TranscriptPanelTests(unittest.TestCase):
    """The panel is wired up here without ever opening a real window."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = dictate.load_settings(os.path.join(self.tmp.name, "config.json"))
        self.app = dictate.VoiceDictationApp(self.settings)
        self.app.running = True
        self.app.overlay = None

    def tearDown(self):
        self.app.running = False
        self.tmp.cleanup()

    def test_panel_settings_exist(self):
        self.assertTrue(self.settings.overlay_enabled)
        self.assertGreater(self.settings.overlay_hide_after, 0.0)
        self.assertGreaterEqual(self.settings.overlay_max_entries, 1)

    def test_notice_shows_instead_of_the_last_transcript(self):
        panel = overlay.TranscriptOverlay(self.settings)
        panel._state = "idle"
        panel._entries = [("09:00", "an older transcript")]
        panel.notify("Heard nothing. Check the microphone")
        kind, value, extra = panel._queue.get_nowait()
        self.assertEqual(kind, "notice")
        self.assertEqual(value, "Heard nothing. Check the microphone")
        self.assertGreater(extra, 0.0)

        panel._notice = value
        self.assertEqual(panel._status_text(), "Heard nothing. Check the microphone")

    def test_a_failed_paste_never_raises_without_a_panel(self):
        self.app.overlay = None
        self.app._notify_panel("Heard nothing")

    def test_elapsed_clock(self):
        self.assertEqual(overlay.format_elapsed(0), "0:00")
        self.assertEqual(overlay.format_elapsed(3.9), "0:03")
        self.assertEqual(overlay.format_elapsed(67), "1:07")
        self.assertEqual(overlay.format_elapsed(-5), "0:00")

    def test_header_text_stays_on_one_line(self):
        self.assertEqual(overlay._one_line("first\n  second "), "first second")
        line = overlay._one_line("word " * 40)
        self.assertLessEqual(len(line), 58)
        self.assertTrue(line.endswith("…"))

    def test_states_and_transcripts_reach_the_panel(self):
        seen = []

        class FakePanel:
            def set_state(self, state, started_at=None):
                seen.append(("state", state))

            def add_entry(self, text):
                seen.append(("entry", text))

        self.app.overlay = FakePanel()
        self.app._model = object()
        self.app._recording = True
        self.app.refresh_state()
        self.app._recording = False
        self.app.refresh_state()
        self.app._add_overlay_entry("hello there")

        self.assertIn(("state", "recording"), seen)
        self.assertIn(("state", "idle"), seen)
        self.assertEqual(seen[-1], ("entry", "hello there"))

    def test_refresh_state_is_safe_without_a_panel(self):
        self.app._model = object()
        self.app.refresh_state()
        self.app._add_overlay_entry("ignored")


class ClipboardTests(unittest.TestCase):
    def test_history_keeps_newest_first_and_deduplicates(self):
        history = dictate.ClipboardHistory(3)
        history.add("one")
        history.add("two")
        history.add("one")
        self.assertEqual(history.items(), ["one", "two"])
        history.add("three")
        history.add("four")
        self.assertEqual(history.items(), ["four", "three", "one"])

    def test_history_ignores_blank_and_repeated_values(self):
        history = dictate.ClipboardHistory(5)
        self.assertFalse(history.add(""))
        self.assertFalse(history.add("   "))
        self.assertTrue(history.add("text"))
        self.assertFalse(history.add("text"))
        self.assertEqual(history.items(), ["text"])

    def test_history_size_is_clamped(self):
        self.assertEqual(dictate.ClipboardHistory(0).limit, 1)
        self.assertEqual(dictate.ClipboardHistory(999).limit, 20)

    def test_auto_clipboard_decision(self):
        self.assertTrue(
            dictate.should_restore_clipboard(False, "old", "new text", "new text")
        )
        self.assertFalse(
            dictate.should_restore_clipboard(True, "old", "new text", "new text")
        )
        # A late paste that landed on different contents is never overwritten.
        self.assertFalse(
            dictate.should_restore_clipboard(False, "old", "changed", "new text")
        )
        self.assertFalse(dictate.should_restore_clipboard(False, None, "x", "x"))

    def test_clipboard_preview_is_short_and_flat(self):
        self.assertEqual(overlay.clipboard_preview("a\n b"), "a b")
        preview = overlay.clipboard_preview("word " * 80)
        self.assertLessEqual(len(preview), 150)
        self.assertTrue(preview.endswith("…"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
