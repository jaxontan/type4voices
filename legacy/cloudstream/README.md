# cloudstream

Push-to-talk dictation for developers. Hold a key, talk, text lands in whatever
window has focus.

Your audio goes to a provider **you** chose, over **your** credentials, from
**your** machine. There is no cloudstream server, because there is no cloudstream
server to trust. Tokens live in memory and in the credential file your own client
already wrote; nothing is uploaded anywhere except the audio you deliberately
dictate.

## What is actually built

| Piece | State |
| --- | --- |
| 16 kHz mono PCM16 capture, 20 ms frames, device-rate resampling | working |
| Non-focusable HUD overlay, dim interim → bright final | working |
| Tap-to-toggle hotkey, 15 s silence auto-commit, cancel hotkey | working |
| Text injection into the focused app (clipboard restore, no clobbering) | working |
| Provider interface, priority chain, mid-utterance failover with audio replay | working |
| Buffered-utterance replay so a dead socket costs latency, not the sentence | working |
| Glossary + identifier correction, transcript history (text only) | partial — glossary is stored and passed to providers, the fuzzy corrector itself is next |
| `mock` provider (offline, proves the loop) | working |
| `claude-voice` adapter (transport, lifecycle, keepalive, failover, diagnostics) | working code, wire schema filled from the client's own voice client — **live probe needs a fresh token** |
| `gemini-api` provider (Gemini 3.5 Transcribe Live over BidiGenerateContent) | working (API key in settings or `GEMINI_API_KEY`) |
| `gemini-web` provider | not built, not recommended |
| `local` (whisper.cpp) provider | not built |

## The three providers, honestly

1. **Claude, your subscription.** Streams to the voice endpoint your own plan
   already pays for, using the token your own Claude Code client wrote. Marginal
   cost ~zero and best-in-class on code identifiers. The catch is that it is not a
   public API: fields change without notice, and automating it likely violates
   Anthropic's terms. It ships behind an explicit toggle. The wire schema in
   `protocol.json` is copied from the client's own `voiceStreamSTT` module
   (endpoint, query, `KeepAlive`/`CloseStream`, `TranscriptText`/`TranscriptEndpoint`),
   never guessed, and `npm run probe -- claude-voice` is what proves it still
   works.

2. **Gemini API.** The sanctioned path. Officially documented, stable, cheap per
   minute, native audio, supports the vocabulary biasing that makes technical
   dictation work. This is what a product should be built on.

3. **Gemini web session.** Would reuse a signed-in browser session. Against
   Google's terms, very brittle, and it puts a logged-in Google account behind a
   tool that types into windows. Listed in settings with that warning, not built.

The shape to avoid, and the reason this is a local app: a **server holding many
users' logins and pooling their quota**. That is credential custody plus account
sharing. The people who get banned are your users. Same feature, none of that
liability, is a local client where each user's own tokens never leave their own
keychain.

## Run it

```powershell
npm install
npm run build
npm start
```

First launch opens settings. Leave `Mock (offline)` enabled to try the loop with
no credentials at all: `Ctrl+Shift+Space`, speak, wait, and the transcript is
pasted into whatever was focused.

| Shortcut | Action |
| --- | --- |
| `Ctrl+Shift+Space` | start / finish dictation |
| `Ctrl+Shift+E` | cancel, discard audio |
| `Ctrl+Shift+,` | settings |

## Verify without a microphone or a network

```powershell
npm run selftest        # session + registry + mock provider, no windows
npm run selftest:audio  # real mic -> worklet -> resample -> IPC -> buffer
npm run smoke           # boots both windows, exits
npm run probe           # one second of silence through provider #1
npm run probe -- claude-voice  # the same, through the Claude endpoint
```

`selftest:audio` prints a byte count. Roughly 32 kB per second of speech means the
capture path is healthy; `0` means the microphone never delivered a frame.

## Turning on Claude voice

The adapter refuses to start until the protocol is verified, because a
plausible-looking frame schema fails silently and you blame the microphone.
The schema shipped in `protocol.json` is copied from Claude Code's own voice
client, so normally there is nothing to fill in.

1. Run your own Claude Code client once to sign in. After that, "auto-refresh
   Claude login" (on by default) keeps it working: an expired access token is
   exchanged at the same endpoint and with the same client id the installed
   Claude Code client uses, and the rotated pair is written back to
   `~/.claude/.credentials.json` so that client stays signed in too. Turn the
   toggle off to make an expired token fail before the network instead.
2. `npm run probe -- claude-voice` — silence must transcribe to `""` *without* an
   error. Text from silence means you are decoding the wrong field.
3. Enable the provider in settings and make it first in the chain (mock is first
   by default), then dictate.

The write-back is a compare-and-swap on the refresh token we sent: if Claude
Code refreshed first, its result is kept and ours is discarded. Refresh tokens
are single-use-ish, so doing it any other way would sign your own client out.

If fields drift, follow `tools/ws-sniff.md`, edit `protocol.json` only, and set
`capturedAt`/`capturedFrom` to your capture. No code changes. That is the entire
maintenance story for the unofficial provider.

## Layout

```
src/main/
  index.ts            app lifecycle, windows, IPC, phase machine
  session.ts          utterance orchestration: buffer -> stream -> failover -> replay
  hotkey.ts           global shortcuts, degrades one key at a time
  injector.ts         clipboard + Ctrl/Cmd+V at the OS level, then restore
  tokens.ts           opt-in read/refresh of ~/.claude/.credentials.json, rotating write-back
  store.ts            settings + text-only history in userData
  providers/
    types.ts          SttSession contract and shared errors
    registry.ts       catalog, priority chain, unbuilt providers
    mock.ts           offline provider
    quota.ts          "is it really unlimited" probe
    claude/protocol.json   observed wire schema (the only thing you edit on drift)
src/renderer/         overlay (capture + HUD), settings window
tools/ws-sniff.md     how to capture the real protocol
```

Audio is never written to disk. History stores text, provider, duration and
character count — nothing you did not see on screen.

## Next, in order

1. Gemini API provider — the one that makes this a product rather than a demo.
2. The glossary corrector: fuzzy match against identifiers, then one cleanup pass.
3. Hold-to-talk: needs a low-level keyboard hook (`uiohook-napi`), since
   Electron's global shortcuts have no key-up event.
4. Local whisper.cpp fallback — offline, free, no terms risk.
5. Tray icon + packaging.
