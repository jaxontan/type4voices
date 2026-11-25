"""Offline unit tests for context_awareness.py and OpenAIRewriter."""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import context_awareness
import rewrite


class ContextAwarenessTests(unittest.TestCase):
    def test_classify_app_by_process(self):
        self.assertEqual(context_awareness.classify_app("Code.exe", "dictate.py"), "code")
        self.assertEqual(context_awareness.classify_app("Cursor.exe", "README.md"), "code")
        self.assertEqual(context_awareness.classify_app("WindowsTerminal.exe", "PowerShell"), "terminal")
        self.assertEqual(context_awareness.classify_app("OUTLOOK.EXE", "Inbox"), "email")
        self.assertEqual(context_awareness.classify_app("Slack.exe", "#general"), "chat")
        self.assertEqual(context_awareness.classify_app("Obsidian.exe", "Notes"), "notes")
        self.assertEqual(context_awareness.classify_app("calc.exe", "Calculator"), "general")

    def test_classify_browser_by_title(self):
        self.assertEqual(
            context_awareness.classify_app("chrome.exe", "Pull Request #42 - GitHub - Google Chrome"),
            "code",
        )
        self.assertEqual(
            context_awareness.classify_app("msedge.exe", "Inbox - Gmail - Microsoft Edge"),
            "email",
        )
        self.assertEqual(
            context_awareness.classify_app("chrome.exe", "WhatsApp Web - Google Chrome"),
            "chat",
        )
        self.assertEqual(
            context_awareness.classify_app("chrome.exe", "Roadmap - Notion - Google Chrome"),
            "notes",
        )

    def test_sensitive_windows_are_blocked(self):
        self.assertTrue(context_awareness.is_sensitive_context("1Password.exe", "Vault"))
        self.assertTrue(context_awareness.is_sensitive_context("KeePassXC.exe", "Passwords"))
        self.assertTrue(
            context_awareness.is_sensitive_context("chrome.exe", "DBS digibank Online Banking - Chrome")
        )
        self.assertTrue(
            context_awareness.is_sensitive_context("msedge.exe", "Sign in to your account")
        )
        ctx = context_awareness.capture_active_context(
            window_provider=lambda: ("1Password.exe", "MySecretVault ComfyUI.py")
        )
        self.assertTrue(ctx.sensitive)
        self.assertEqual(ctx.window_title, "")
        self.assertEqual(ctx.dynamic_terms, ())

    def test_extract_terms_from_title(self):
        title = "dictate.py - types4voices - ComfyUI DGX-Spark GB10 AIGC - Visual Studio Code"
        terms = context_awareness.extract_terms_from_title(title)
        self.assertIn("dictate.py", terms)
        self.assertIn("ComfyUI", terms)
        self.assertIn("DGX-Spark", terms)
        self.assertIn("GB10", terms)
        self.assertIn("AIGC", terms)
        # Window chrome stop words should not leak in.
        low = {t.lower() for t in terms}
        self.assertNotIn("visual", low)
        self.assertNotIn("studio", low)
        self.assertNotIn("code", low)

    def test_merge_hotwords_deduplicates_case_insensitively(self):
        merged = context_awareness.merge_hotwords(
            ["PyTorch", "Qwen3-ASR"], ["pytorch", "ComfyUI", "GB10"]
        )
        self.assertEqual(merged, ["PyTorch", "Qwen3-ASR", "ComfyUI", "GB10"])

    def test_capture_active_context_populates_category_and_terms(self):
        ctx = context_awareness.capture_active_context(
            window_provider=lambda: ("Code.exe", "rewrite.py - types4voices - Visual Studio Code")
        )
        self.assertFalse(ctx.sensitive)
        self.assertEqual(ctx.category, "code")
        self.assertIn("rewrite.py", ctx.dynamic_terms)
        self.assertTrue(context_awareness.style_hint_for_category(ctx.category))


class SelfCorrectionAndEditTests(unittest.TestCase):
    def test_spoken_self_correction_number_change_passes_guardrail(self):
        ok, reason = rewrite.accept_rewrite(
            "Let's meet at 3 pm, oh wait make it 4 pm tomorrow.",
            "Let's meet at 4 pm tomorrow.",
        )
        self.assertTrue(ok, reason)

    def test_plain_lost_number_without_cue_still_rejected(self):
        ok, reason = rewrite.accept_rewrite(
            "Let's meet at 3 pm on 14 December.",
            "Let's meet on 14 December.",
        )
        self.assertFalse(ok)
        self.assertIn("3", reason)

    def test_style_hint_included_in_system_prompt(self):
        msgs = rewrite.build_messages(
            "check config ui",
            style_hint="Keep technical terms exact.",
        )
        self.assertIn("Keep technical terms exact.", msgs[0]["content"])

    def test_accept_edit_guardrails(self):
        self.assertTrue(rewrite.accept_edit("hello world", "Hello, world!")[0])
        self.assertFalse(rewrite.accept_edit("hello world", "")[0])
        self.assertFalse(rewrite.accept_edit("hello", " ".join(["x"] * 200))[0])


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    reject_extra = False
    last_payload = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "gemma4-e2b"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).last_payload = payload
        if type(self).reject_extra and "cache_prompt" in payload:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "unknown field cache_prompt"}')
            return
        user_msg = payload["messages"][-1]["content"]
        if user_msg.startswith("Instruction:"):
            reply = "Edited output."
        else:
            reply = "I've already told him many times."
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": reply}}]
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


class OpenAIRewriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _FakeOpenAIHandler.reject_extra = False
        _FakeOpenAIHandler.last_payload = None

    def test_health_check_and_rewrite_round_trip(self):
        r = rewrite.OpenAIRewriter("gemma4-e2b", f"http://127.0.0.1:{self.port}/v1")
        self.assertTrue(r.load())
        self.assertTrue(r.ready)
        out = r.rewrite(
            "I already tell him many time.",
            style_hint="Keep phrasing concise.",
        )
        self.assertEqual(out, "I've already told him many times.")
        self.assertTrue(_FakeOpenAIHandler.last_payload.get("cache_prompt"))

    def test_retries_without_extra_fields_on_400(self):
        _FakeOpenAIHandler.reject_extra = True
        r = rewrite.OpenAIRewriter("gemma4-e2b", f"http://127.0.0.1:{self.port}/v1")
        self.assertTrue(r.load())
        out = r.generate("I already tell him many time.")
        self.assertEqual(out, "I've already told him many times.")
        self.assertNotIn("cache_prompt", _FakeOpenAIHandler.last_payload)

    def test_edit_selection_round_trip(self):
        r = rewrite.OpenAIRewriter("gemma4-e2b", f"http://127.0.0.1:{self.port}/v1")
        self.assertTrue(r.load())
        self.assertEqual(r.edit_selection("raw draft text", "make it shorter"), "Edited output.")

    def test_unreachable_server_fails_load_cleanly(self):
        r = rewrite.OpenAIRewriter("gemma4-e2b", "http://127.0.0.1:1/v1", health_timeout=0.5)
        self.assertFalse(r.load())
        self.assertFalse(r.ready)
        self.assertTrue(r.error)


if __name__ == "__main__":
    unittest.main()
