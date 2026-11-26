"""Pure-function tests for the Gemini speech engine (no network, no key)."""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import gemini_stt as stt


class LanguageTests(unittest.TestCase):
    def test_auto_becomes_empty_list(self):
        self.assertEqual(stt.code_for_language("Auto"), [])
        self.assertEqual(stt.code_for_language(None), [])

    def test_known_names_become_bcp47_lists(self):
        self.assertEqual(stt.code_for_language("English"), ["en-US"])
        self.assertEqual(stt.code_for_language("Japanese"), ["ja-JP"])
        self.assertEqual(stt.code_for_language("German"), ["de-DE"])

    def test_unknown_falls_back_to_auto(self):
        self.assertEqual(stt.code_for_language("Klingon"), [])


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


class SetupTests(unittest.TestCase):
    def test_setup_carries_model_and_verbatim_mode(self):
        setup = stt._setup_message(["en-US"], ["Frogineer"])
        self.assertEqual(setup["setup"]["model"], "models/gemini-3.5-transcribe-live")
        self.assertEqual(setup["setup"]["generationConfig"]["responseModalities"], ["TEXT"])
        tr = setup["setup"]["inputAudioTranscription"]
        self.assertEqual(tr["languageCodes"], ["en-US"])
        self.assertEqual(tr["mode"], "VERBATIM")
        self.assertEqual(tr["customVocabulary"], ["Frogineer"])

    def test_empty_keyterms_omit_vocabulary(self):
        setup = stt._setup_message([], [])
        self.assertNotIn("customVocabulary", setup["setup"]["inputAudioTranscription"])

    def test_keyterms_are_capped_and_trimmed(self):
        terms = [f"term{index}" for index in range(200)] + ["x" * 200]
        setup = stt._setup_message([], terms)
        vocab = setup["setup"]["inputAudioTranscription"]["customVocabulary"]
        self.assertEqual(len(vocab), 100)
        self.assertTrue(all(len(term) <= 64 for term in vocab))


class ApiKeyTests(unittest.TestCase):
    def test_missing_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                # ensure no env keys
                for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
                    os.environ.pop(name, None)
                with self.assertRaises(stt.SttError) as caught:
                    stt.transcribe(np.zeros(1600, np.float32), api_key_value="")
        self.assertIn("API key", str(caught.exception))

    def test_env_fallback(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(stt.api_key(None), "env-key")


class FakeConn:
    def __init__(self, messages):
        self.sent = []
        self._messages = list(messages)
        self.timeout = None

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        import websocket

        if self._messages:
            return self._messages.pop(0)
        raise websocket.WebSocketTimeoutException()

    def settimeout(self, value):
        self.timeout = value

    def close(self):
        pass


class TranscribeFlowTests(unittest.TestCase):
    def _hear_one_final(self):
        joined = " ".join(self._chosen_conn.sent)
        self.assertIn('"audio"', joined)
        self.assertIn('"mimeType": "audio/pcm;rate=16000"', joined)
        self.assertIn("audioStreamEnd", joined)
        self.assertEqual(self._chosen_conn.sent[0], '{"setup": {"model": "models/gemini-3.5-transcribe-live", "generationConfig": {"responseModalities": ["TEXT"]}, "inputAudioTranscription": {"languageCodes": [], "mode": "VERBATIM"}}}')

    def test_final_transcript_is_returned(self):
        messages = [
            '{"setupComplete": {}}',
            '{"serverContent": {"interimInputTranscription": {"text": "Testing one"}}}',
            '{"serverContent": {"inputTranscription": {"text": "Testing one two three."}}}',
            '{"serverContent": {"generationComplete": true}}',
            '{"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "3.0s"}}',
        ]
        fake = FakeConn(messages)
        self._chosen_conn = fake
        with mock.patch.object(stt, "PACE_SECONDS", 0), \
                mock.patch("gemini_stt.websocket.create_connection", return_value=fake):
            text = stt.transcribe(
                np.ones(48000, np.float32) * 0.1,
                sample_rate=16000,
                language_codes=[],
                api_key_value="test-key",
            )
        self.assertEqual(text, "Testing one two three.")
        self._hear_one_final()

    def test_multiple_finals_are_joined(self):
        messages = [
            '{"setupComplete": {}}',
            '{"serverContent": {"inputTranscription": {"text": "Hello"}}}',
            '{"serverContent": {"inputTranscription": {"text": "world"}}}',
            '{"serverContent": {"generationComplete": true}}',
        ]
        fake = FakeConn(messages)
        self._chosen_conn = fake
        with mock.patch.object(stt, "PACE_SECONDS", 0), \
                mock.patch("gemini_stt.websocket.create_connection", return_value=fake):
            text = stt.transcribe(
                np.ones(16000, np.float32) * 0.1,
                api_key_value="test-key",
            )
        self.assertEqual(text, "Hello world")
        self._hear_one_final()

    def test_server_error_raises(self):
        import websocket

        messages = [
            '{"setupComplete": {}}',
            '{"error": {"message": "bad audio"}}',
        ]
        fake = FakeConn(messages)
        with mock.patch.object(stt, "PACE_SECONDS", 0), \
                mock.patch("gemini_stt.websocket.create_connection", return_value=fake):
            with self.assertRaises(stt.SttError) as caught:
                stt.transcribe(
                    np.ones(16000, np.float32) * 0.1,
                    api_key_value="test-key",
                )
        self.assertIn("bad audio", str(caught.exception))

    def test_bad_status_raises_key_error(self):
        import websocket

        exc = websocket.WebSocketBadStatusException("nope", 403, resp_body=b'{"error":{"message":"API key not valid"}}')
        with mock.patch("gemini_stt.websocket.create_connection", side_effect=exc):
            with self.assertRaises(stt.SttError) as caught:
                stt.transcribe(
                    np.ones(16000, np.float32) * 0.1,
                    api_key_value="test-key",
                )
        self.assertIn("rejected", str(caught.exception).lower())

    def test_interim_fallback_when_no_final(self):
        import websocket

        messages = [
            '{"setupComplete": {}}',
            '{"serverContent": {"interimInputTranscription": {"text": "partial words"}}}',
        ]
        fake = FakeConn(messages)
        with mock.patch.object(stt, "PACE_SECONDS", 0), \
                mock.patch("gemini_stt.websocket.create_connection", return_value=fake):
            text = stt.transcribe(
                np.ones(16000, np.float32) * 0.1,
                api_key_value="test-key",
                timeout=0.0,
            )
        self.assertEqual(text, "partial words")

    def test_empty_clip_returns_empty(self):
        text = stt.transcribe(
            np.zeros(0, np.float32),
            api_key_value="test-key",
        )
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
