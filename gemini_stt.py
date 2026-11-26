"""Gemini 3.5 Transcribe Live, over the public Live API (an API key).

This is the Google equivalent of anthropic_stt.py: the recorded clip is
streamed to Google's speech endpoint over the same WebSocket the Live API
uses, authenticated with an API key the user keeps in config.json. Unlike the
Claude path, this is a documented, supported API with a free tier (about
0.005 USD per minute on the paid tier).

Wire contract (see ai.google.dev/gemini-api/docs/live-api/live-transcribe):

    wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<api key>

    first message:
      {"setup": {"model": "models/gemini-3.5-transcribe-live",
                 "generationConfig": {"responseModalities": ["TEXT"]},
                 "inputAudioTranscription": {"languageCodes": [...],
                                            "mode": "VERBATIM",
                                            "customVocabulary": [...]}}}

    then realtimeInput.audio frames (raw 16-bit PCM, 16 kHz, little-endian,
    100 ms / 3200 bytes each), paced at no more than 2x real time, then
      {"realtimeInput": {"audioStreamEnd": true}}

    <- {"serverContent": {"interimInputTranscription": {"text": ...}}}
       {"serverContent": {"inputTranscription": {"text": ...}}}   (final, one
                                                                  per turn)
       {"serverContent": {"generationComplete": true}}
       {"voiceActivity": {"type": "ACTIVITY_END", ...}}
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

try:
    import websocket
except Exception:  # pragma: no cover - only on a broken install
    websocket = None

DEFAULT_LANGUAGE = "en"

# The app's tray menu speaks display names; the endpoint wants BCP-47 codes.
# Empty list means "auto-detect", which is the safer default for short clips.
LANGUAGE_CODES = {
    "auto": [],
    "english": ["en-US"],
    "chinese": ["zh-CN"],
    "cantonese": ["zh-HK"],
    "japanese": ["ja-JP"],
    "korean": ["ko-KR"],
    "spanish": ["es-ES"],
    "french": ["fr-FR"],
    "german": ["de-DE"],
    "russian": ["ru-RU"],
}


def code_for_language(name: str | None) -> list[str]:
    return list(LANGUAGE_CODES.get(str(name or "").strip().lower(), []))


MODEL = "gemini-3.5-transcribe-live"
WS_HOST = "generativelanguage.googleapis.com"
WS_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
ENDPOINT_SAMPLE_RATE = 16000
FRAME_BYTES = 3200  # 100 ms of 16 kHz mono PCM16
FRAME_SECONDS = 0.1
PACE_SECONDS = 0.05  # 2x real time keeps the server happy and halves latency
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 15.0
FINAL_TIMEOUT = 12.0

USER_AGENT = "type4voices/2.0 (+local dictation client)"


class SttError(Exception):
    """Carries a sentence worth showing in the transcript panel."""


def api_key(value: str | None = None) -> str:
    """Resolve an API key from the argument, the environment, or config."""
    if value:
        text = str(value).strip()
        if text:
            return text
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        found = os.environ.get(env_name)
        if found and found.strip():
            return found.strip()
    return ""


def _pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Float mono at any rate -> 16 kHz mono PCM16, which is all it accepts."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != ENDPOINT_SAMPLE_RATE and samples.size:
        target = max(1, int(round(samples.size * ENDPOINT_SAMPLE_RATE / sample_rate)))
        source_axis = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target_axis = np.linspace(0.0, 1.0, target, endpoint=False)
        samples = np.interp(target_axis, source_axis, samples).astype(np.float32)
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _url(key: str) -> str:
    return f"wss://{WS_HOST}{WS_PATH}?key={urllib.parse.quote(key, safe='')}"


def _setup_message(language_codes: list[str], keyterms: list[str] | None) -> dict:
    transcription: dict = {
        "languageCodes": list(language_codes or []),
        "mode": "VERBATIM",
    }
    terms = []
    for term in (keyterms or [])[:100]:
        clean = " ".join(str(term).split())[:64]
        if clean:
            terms.append(clean)
    if terms:
        transcription["customVocabulary"] = terms
    return {
        "setup": {
            "model": f"models/{MODEL}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": transcription,
        }
    }


def _server_error(body: str) -> str:
    try:
        payload = json.loads(body or "{}")
    except Exception:
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("status") or "")
    if isinstance(error, str):
        return error
    return ""


def transcribe(
    audio: np.ndarray,
    sample_rate: int = ENDPOINT_SAMPLE_RATE,
    language_codes: list[str] | None = None,
    keyterms: list[str] | None = None,
    api_key_value: str | None = None,
    mode: str = "VERBATIM",
    timeout: float = FINAL_TIMEOUT,
) -> str:
    """Stream one recorded clip and return the final transcript.

    The whole clip is uploaded at 2x real time, then ``audioStreamEnd`` asks
    for the final. Returns ``""`` for silence. Raises ``SttError`` with a
    readable sentence for anything the user can fix.
    """
    if websocket is None:
        raise SttError("The Gemini speech engine is unavailable (websocket-client missing)")

    key = api_key(api_key_value)
    if not key:
        raise SttError(
            "No Gemini API key. Add \"gemini_api_key\" to config.json "
            "or set GEMINI_API_KEY."
        )

    pcm = _pcm16(audio, sample_rate)
    if not pcm:
        return ""

    # VERBATIM mode is the default and keeps the spoken words intact.
    transcription_mode = "VERBATIM" if str(mode).upper() != "SMART" else "SMART"
    setup = _setup_message(language_codes or [], keyterms)
    setup["setup"]["inputAudioTranscription"]["mode"] = transcription_mode

    try:
        conn = websocket.create_connection(
            _url(key), timeout=CONNECT_TIMEOUT, suppress_origin=True
        )
    except websocket.WebSocketBadStatusException as exc:
        status = getattr(exc, "status_code", None)
        resp_body = getattr(exc, "resp_body", None)
        body_str: str
        if resp_body is None:
            body_str = ""
        elif isinstance(resp_body, str):
            body_str = resp_body
        elif isinstance(resp_body, bytes):
            body_str = resp_body.decode("utf-8", "replace")
        else:
            body_str = str(resp_body)
        detail = _server_error(body_str)
        if status in (400, 401, 403):
            raise SttError(
                "The Gemini API key was rejected"
                + (f" ({detail})" if detail else "")
                + ". Check gemini_api_key in config.json."
            ) from None
        if status == 429:
            raise SttError(
                "Gemini quota exceeded (free tier limits). Try again later "
                "or enable billing."
            ) from None
        raise SttError(
            f"Could not reach the Gemini speech endpoint (HTTP {status})."
        ) from None
    except Exception as exc:
        raise SttError(f"Could not reach the Gemini speech endpoint: {exc}") from None

    finals: list[str] = []
    interim = ""
    activity_end = False
    generation_complete = False
    input_final_seen = False
    try:
        conn.settimeout(CONNECT_TIMEOUT)
        conn.send(json.dumps(setup))
        deadline = time.time() + CONNECT_TIMEOUT
        while time.time() < deadline:
            try:
                message = conn.recv()
            except websocket.WebSocketTimeoutException:
                break
            if isinstance(message, bytes):
                message = message.decode("utf-8", "replace")
            err = _server_error(message)
            if err:
                raise SttError(f"Gemini speech error: {err}")
            if "setupComplete" in str(message):
                break
        else:
            raise SttError("Gemini did not finish setup.")

        conn.settimeout(0.05)
        for offset in range(0, len(pcm), FRAME_BYTES):
            try:
                conn.send(
                    json.dumps(
                        {
                            "realtimeInput": {
                                "audio": {
                                    "mimeType": f"audio/pcm;rate={ENDPOINT_SAMPLE_RATE}",
                                    "data": base64.b64encode(
                                        pcm[offset : offset + FRAME_BYTES]
                                    ).decode("ascii"),
                                }
                            }
                        }
                    )
                )
            except websocket.WebSocketConnectionClosedException:
                raise SttError(
                    "Gemini closed the connection while streaming audio."
                ) from None
            # Drain anything that arrives while we stream.
            while True:
                try:
                    message = conn.recv()
                except websocket.WebSocketTimeoutException:
                    break
                except Exception:
                    break
                if isinstance(message, bytes):
                    message = message.decode("utf-8", "replace")
                err = _server_error(message)
                if err:
                    raise SttError(f"Gemini speech error: {err}")
                try:
                    payload = json.loads(message)
                except Exception:
                    continue
                sc = payload.get("serverContent") or {}
                if sc.get("interimInputTranscription"):
                    interim = str(
                        sc["interimInputTranscription"].get("text") or interim
                    )
                if sc.get("inputTranscription"):
                    text = str(sc["inputTranscription"].get("text") or "")
                    if text:
                        finals.append(text)
                        input_final_seen = True
                if sc.get("generationComplete"):
                    generation_complete = True
                va = payload.get("voiceActivity") or {}
                if va.get("type") == "ACTIVITY_END":
                    activity_end = True
            time.sleep(PACE_SECONDS)

        conn.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))

        conn.settimeout(1.0)
        deadline = time.time() + timeout
        while time.time() < deadline and not (
            input_final_seen and (generation_complete or activity_end)
        ):
            try:
                message = conn.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                break
            if isinstance(message, bytes):
                message = message.decode("utf-8", "replace")
            err = _server_error(message)
            if err:
                raise SttError(f"Gemini speech error: {err}")
            try:
                payload = json.loads(message)
            except Exception:
                continue
            sc = payload.get("serverContent") or {}
            if sc.get("interimInputTranscription"):
                interim = str(
                    sc["interimInputTranscription"].get("text") or interim
                )
            if sc.get("inputTranscription"):
                text = str(sc["inputTranscription"].get("text") or "")
                if text:
                    finals.append(text)
                    input_final_seen = True
            if sc.get("generationComplete"):
                generation_complete = True
            va = payload.get("voiceActivity") or {}
            if va.get("type") == "ACTIVITY_END":
                activity_end = True
    finally:
        try:
            conn.close()
        except Exception:
            pass

    text = " ".join(finals).strip() or interim.strip()
    return " ".join(text.split())
