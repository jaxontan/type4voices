"""Claude Code's own speech-to-text stream, on your own subscription.

This is the path the sibling cloudstream app proved out: the audio goes to the
same endpoint Claude Code's ``/voice`` mode uses, authenticated with the token
that client already wrote to ``~/.claude/.credentials.json``. Nothing here is a
public Anthropic API — the endpoint, query string, control frames and reply
types were copied from the Claude Code client itself, and automating them
likely breaks Anthropic's terms. That is the trade: no per-minute billing, no
second account, no local 4.4 GB model in memory while you talk.

Wire contract (keep this in sync with cloudstream's protocol.json):

    wss://api.anthropic.com/api/ws/speech_to_text/voice_stream
      ?encoding=linear16&sample_rate=16000&channels=1
      &endpointing_ms=300&utterance_end_ms=1000&language=<code>
      &use_conversation_engine=true&stt_provider=deepgram-nova3
    headers: authorization: Bearer <access token>, x-app: cli

    -> binary PCM16 frames, ``{"type":"KeepAlive"}``, ``{"type":"CloseStream"}``
    <- ``{"type":"TranscriptText","data":...}``   cumulative, interim
       ``{"type":"TranscriptEndpoint"}``          end of utterance, final
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

DEFAULT_LANGUAGE = "en"

# The app's tray menu speaks display names; the endpoint wants BCP-47 codes.
LANGUAGE_CODES = {
    "auto": DEFAULT_LANGUAGE,
    "english": "en",
    "chinese": "zh",
    "cantonese": "yue",
    "japanese": "ja",
    "korean": "ko",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "russian": "ru",
}


def code_for_language(name: str | None) -> str:
    return LANGUAGE_CODES.get(str(name or "").strip().lower(), DEFAULT_LANGUAGE)

# Observed in the installed Claude Code client (versions/2.1.x), not guessed.
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
WS_URL = "wss://api.anthropic.com/api/ws/speech_to_text/voice_stream"
DEFAULT_SCOPES = (
    "user:file_upload",
    "user:inference",
    "user:mcp_servers",
    "user:plugins",
    "user:profile",
    "user:sessions:claude_code",
)

ENDPOINT_SAMPLE_RATE = 16000
FRAME_BYTES = 640          # 20 ms of 16 kHz mono PCM16
KEEPALIVE_SECONDS = 8.0
CONNECT_TIMEOUT = 8.0
READ_TIMEOUT = 30.0
REFRESH_TIMEOUT = 15.0
USER_AGENT = "type4voices/2.0 (+local dictation client)"


class SttError(Exception):
    """Carries a sentence worth showing in the transcript panel."""


def credentials_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude", ".credentials.json")


def _no_window() -> int:
    return 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def claude_command() -> str | None:
    """The installed Claude Code executable, which owns the login flow."""
    override = os.environ.get("TYPE4VOICES_CLAUDE_BIN")
    if override and os.path.exists(override):
        return override
    found = shutil.which("claude")
    if found and os.path.exists(found):
        return found
    home = os.path.expanduser("~")
    for candidate in (
        os.path.join(home, ".local", "bin", "claude.exe"),
        os.path.join(home, ".local", "bin", "claude"),
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def is_logged_in(timeout: float = 10.0) -> bool | None:
    """Ask Claude Code itself. ``None`` means the question could not be asked."""
    binary = claude_command()
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_no_window(),
        )
    except Exception:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return None
    if "loggedIn" not in payload:
        return None
    return bool(payload.get("loggedIn"))


def start_login() -> bool:
    """Open the browser sign-in that Claude Code already ships.

    ``claude auth login`` runs the OAuth flow and writes the credentials file
    this module reads. It is spawned with no console window and no inherited
    handles, so nothing appears on screen except the browser page the user
    actually has to interact with.
    """
    binary = claude_command()
    if not binary:
        return False
    try:
        subprocess.Popen(
            [binary, "auth", "login"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_no_window(),
            close_fds=True,
        )
    except Exception:
        return False
    return True


def wait_for_login(timeout: float = 300.0, interval: float = 2.0) -> bool:
    """Block until Claude Code reports a login, or the timeout runs out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_logged_in() is True:
            return True
        time.sleep(interval)
    return False


def _read_credentials() -> tuple[str, dict]:
    path = credentials_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise SttError(
            "No Claude login found. Run Claude Code once to sign in."
        ) from None
    except Exception as exc:
        raise SttError(f"Could not read the Claude login: {exc}") from None
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if not isinstance(oauth, dict) or not oauth.get("accessToken"):
        raise SttError("The Claude login file has no access token. Sign in again.")
    return path, oauth


def _write_back(path: str, used_refresh_token: str, oauth: dict) -> bool:
    """Put the rotated token where Claude Code expects it, safely.

    Refresh tokens rotate: keeping the new one to ourselves would sign the
    user's own client out. The compare-and-swap on the refresh token we used
    means that if Claude Code refreshed first, its result wins and we touch
    nothing.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            current = json.load(handle)
    except Exception:
        return False
    if not isinstance(current, dict):
        return False
    if (current.get("claudeAiOauth") or {}).get("refreshToken") != used_refresh_token:
        return False
    current["claudeAiOauth"] = {**(current.get("claudeAiOauth") or {}), **oauth}
    tmp = f"{path}.type4voices.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    return True


def _refresh(path: str, oauth: dict) -> dict:
    """Exchange the stored refresh token, exactly as the installed client does."""
    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        raise SttError("The Claude login has no refresh token. Sign in again.")
    expires = oauth.get("refreshTokenExpiresAt")
    if isinstance(expires, (int, float)) and time.time() * 1000 >= expires:
        raise SttError("The Claude login expired. Run Claude Code to sign in again.")

    scopes = oauth.get("scopes") or list(DEFAULT_SCOPES)
    body = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": oauth.get("clientId") or CLIENT_ID,
            "scope": " ".join(scopes),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        TOKEN_URL,
        data=body,
        # Cloudflare in front of the token endpoint answers 403 (error 1010) to
        # the default python-urllib agent, so identify ourselves plainly.
        headers={"content-type": "application/json", "user-agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REFRESH_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error_description", "")
        except Exception:
            pass
        if exc.code in (400, 401):
            raise SttError(
                "The Claude login was rejected"
                + (f" ({detail})" if detail else "")
                + ". Run Claude Code to sign in again."
            ) from None
        raise SttError(f"Could not refresh the Claude login: HTTP {exc.code}") from None
    except Exception as exc:
        raise SttError(f"Could not refresh the Claude login: {exc}") from None

    access = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not access or not isinstance(expires_in, int):
        raise SttError("The Claude login refresh returned no usable token.")

    now = time.time() * 1000
    updated = dict(oauth)
    updated["accessToken"] = access
    updated["refreshToken"] = payload.get("refresh_token") or refresh_token
    updated["expiresAt"] = now + expires_in * 1000
    if isinstance(payload.get("refresh_token_expires_in"), int):
        updated["refreshTokenExpiresAt"] = now + payload["refresh_token_expires_in"] * 1000
    if payload.get("scope"):
        updated["scopes"] = [s for s in payload["scope"].split(" ") if s]

    _write_back(path, refresh_token, updated)
    return updated


def access_token(auto_refresh: bool = True) -> str:
    """A usable access token, refreshed first if it has expired."""
    path, oauth = _read_credentials()
    expires = oauth.get("expiresAt")
    expired = isinstance(expires, (int, float)) and time.time() * 1000 + 60_000 >= expires
    if expired:
        if not auto_refresh:
            raise SttError("The Claude login expired. Run Claude Code to refresh it.")
        oauth = _refresh(path, oauth)
    return str(oauth["accessToken"])


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


def _server_message(body) -> str:
    """Pull the human sentence out of an error reply, if there is one."""
    try:
        if isinstance(body, bytes):
            body = body.decode("utf-8", "replace")
        payload = json.loads(body or "{}")
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or "")
        return str(error or payload.get("message") or "")
    except Exception:
        return ""


def _endpoint_url(language: str, keyterms: list[str] | None) -> str:
    params = [
        ("encoding", "linear16"),
        ("sample_rate", str(ENDPOINT_SAMPLE_RATE)),
        ("channels", "1"),
        ("endpointing_ms", "300"),
        ("utterance_end_ms", "1000"),
        ("language", language or DEFAULT_LANGUAGE),
        ("use_conversation_engine", "true"),
        ("stt_provider", "deepgram-nova3"),
    ]
    for term in (keyterms or [])[:50]:
        clean = " ".join(str(term).split())[:64]
        if clean:
            params.append(("keyterms", clean))
    return WS_URL + "?" + urllib.parse.urlencode(params)


def transcribe(
    audio: np.ndarray,
    sample_rate: int = ENDPOINT_SAMPLE_RATE,
    language: str = DEFAULT_LANGUAGE,
    keyterms: list[str] | None = None,
    auto_refresh: bool = True,
) -> str:
    """Stream one recorded clip and return the final transcript.

    The whole clip is sent as fast as the socket takes it (a few seconds of
    speech is well under a second of upload), then ``CloseStream`` asks for the
    final. Returns ``""`` for silence, which is a valid answer, and raises
    ``SttError`` with a readable sentence for anything the user can fix.
    """
    import websocket  # imported late: only this engine needs the dependency

    pcm = _pcm16(audio, sample_rate)
    if not pcm:
        return ""

    token = access_token(auto_refresh=auto_refresh)
    url = _endpoint_url(language, keyterms)
    headers = [f"authorization: Bearer {token}", "x-app: cli"]

    try:
        # The endpoint rejects a browser-style Origin ("WebSocket connection
        # origin is not allowed"), and this is not a browser: send none.
        conn = websocket.create_connection(
            url, header=headers, timeout=CONNECT_TIMEOUT, suppress_origin=True
        )
    except websocket.WebSocketBadStatusException as exc:
        if exc.status_code in (401, 403):
            detail = _server_message(getattr(exc, "resp_body", b""))
            raise SttError(
                "The Claude login was not accepted"
                + (f" ({detail})" if detail else "")
                + ". Run Claude Code to sign in again."
            ) from None
        raise SttError(
            f"Could not reach the Anthropic speech endpoint (HTTP {exc.status_code})."
        ) from None
    except Exception as exc:
        raise SttError(f"Could not reach the Anthropic speech endpoint: {exc}") from None

    partial = ""
    final = ""
    try:
        conn.settimeout(READ_TIMEOUT)
        conn.send(json.dumps({"type": "KeepAlive"}))
        for offset in range(0, len(pcm), FRAME_BYTES):
            conn.send_binary(pcm[offset: offset + FRAME_BYTES])
        conn.send(json.dumps({"type": "CloseStream"}))

        deadline = time.time() + READ_TIMEOUT
        while time.time() < deadline:
            try:
                message = conn.recv()
            except websocket.WebSocketTimeoutException:
                break
            if isinstance(message, bytes):
                continue
            try:
                payload = json.loads(message)
            except Exception:
                continue
            kind = str(payload.get("type") or "").lower()
            if kind == "transcripttext":
                partial = str(payload.get("data") or "")
            elif kind == "transcriptendpoint":
                final = partial or str(payload.get("data") or "")
                break
            elif "error" in kind:
                error = payload.get("error")
                if isinstance(error, dict):
                    detail = error.get("message") or error.get("type")
                else:
                    detail = error
                detail = (
                    detail
                    or payload.get("description")
                    or payload.get("message")
                    or payload.get("error_code")
                    or "unknown error"
                )
                raise SttError(f"Anthropic speech error: {detail}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return " ".join((final or partial).split())
