"""Pure-function tests for the Claude speech engine (no network, no login)."""
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

import numpy as np

import anthropic_stt as stt


class LanguageTests(unittest.TestCase):
    def test_tray_names_become_codes(self):
        self.assertEqual(stt.code_for_language("English"), "en")
        self.assertEqual(stt.code_for_language("Cantonese"), "yue")
        self.assertEqual(stt.code_for_language("German"), "de")

    def test_unknown_or_empty_falls_back_to_english(self):
        self.assertEqual(stt.code_for_language("Auto"), "en")
        self.assertEqual(stt.code_for_language("Klingon"), "en")
        self.assertEqual(stt.code_for_language(None), "en")


class AudioTests(unittest.TestCase):
    def test_float_audio_becomes_little_endian_pcm16(self):
        pcm = stt._pcm16(np.array([0.0, 1.0, -1.0], np.float32), 16000)
        self.assertEqual(len(pcm), 6)
        self.assertEqual(np.frombuffer(pcm, "<i2").tolist(), [0, 32767, -32767])

    def test_other_sample_rates_are_resampled_to_16k(self):
        pcm = stt._pcm16(np.zeros(8000, np.float32), 8000)
        self.assertEqual(len(pcm), 16000 * 2)

    def test_empty_clip_is_empty(self):
        self.assertEqual(stt._pcm16(np.zeros(0, np.float32), 16000), b"")


class EndpointTests(unittest.TestCase):
    def test_query_carries_the_observed_contract(self):
        url = stt._endpoint_url("de", [])
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertTrue(url.startswith("wss://api.anthropic.com/api/ws/speech_to_text/voice_stream?"))
        self.assertEqual(query["encoding"], ["linear16"])
        self.assertEqual(query["sample_rate"], ["16000"])
        self.assertEqual(query["channels"], ["1"])
        self.assertEqual(query["language"], ["de"])
        self.assertEqual(query["stt_provider"], ["deepgram-nova3"])
        self.assertNotIn("keyterms", query)

    def test_keyterms_are_capped_and_trimmed(self):
        terms = [f"term{index}" for index in range(60)] + ["x" * 200]
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(stt._endpoint_url("en", terms)).query
        )
        keyterms = query["keyterms"]
        self.assertEqual(len(keyterms), 50)
        self.assertTrue(all(len(term) <= 64 for term in keyterms))


class CredentialTests(unittest.TestCase):
    def test_missing_login_explains_what_to_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = stt.credentials_path
            stt.credentials_path = lambda: os.path.join(tmp, ".credentials.json")
            try:
                with self.assertRaises(stt.SttError) as caught:
                    stt.access_token()
            finally:
                stt.credentials_path = original
        self.assertIn("Claude", str(caught.exception))

    def test_write_back_keeps_a_token_the_client_refreshed_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".credentials.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"claudeAiOauth": {"refreshToken": "newer", "accessToken": "x"}}')
            wrote = stt._write_back(path, "the-one-we-used", {"accessToken": "ours"})
            self.assertFalse(wrote)
            with open(path, encoding="utf-8") as handle:
                self.assertIn('"newer"', handle.read())


class LoginTests(unittest.TestCase):
    def test_missing_client_cannot_start_a_login(self):
        with mock.patch.object(stt, "claude_command", lambda: None):
            self.assertFalse(stt.start_login())
            self.assertIsNone(stt.is_logged_in())

    def test_status_comes_from_the_client_itself(self):
        result = mock.Mock(stdout='{"loggedIn": true}', returncode=0)
        with mock.patch.object(stt, "claude_command", lambda: "claude"), \
                mock.patch.object(stt.subprocess, "run", lambda *a, **k: result):
            self.assertIs(stt.is_logged_in(), True)

    def test_unreadable_status_is_not_taken_as_signed_in(self):
        result = mock.Mock(stdout="not json", returncode=0)
        with mock.patch.object(stt, "claude_command", lambda: "claude"), \
                mock.patch.object(stt.subprocess, "run", lambda *a, **k: result):
            self.assertIsNone(stt.is_logged_in())


if __name__ == "__main__":
    unittest.main()
