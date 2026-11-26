# Capturing the real voice protocol

`src/main/providers/claude/protocol.json` ships **unverified on purpose**. A
plausible-looking frame schema is worse than none: it fails quietly and produces
garbage transcripts you will blame on the microphone.

So capture one real session from your own account, on your own machine, then fill
the JSON. No code changes are needed afterwards.

## What you need

- Your own working Claude Code client, signed in.
- A TLS-intercepting proxy you control, e.g. `mitmproxy`:

  ```powershell
  pipx install mitmproxy
  mitmweb --listen-port 8080 --save-stream-file capture.flow
  ```

- Trust the mitmproxy CA in your OS trust store so the client's TLS check passes.

## Capture

1. Start `mitmweb` and note the port.
2. Point only your own client at it for the duration of the capture:

   ```powershell
   $env:HTTPS_PROXY = "http://127.0.0.1:8080"
   $env:VOICE_STREAM_BASE_URL = "<the base URL your client uses>"
   ```

   `HTTPS_PROXY` does not tunnel `wss://` in every client. If the WebSocket does
   not appear in the flow list, the honest fallback is to add temporary logging
   to your own copy of the client, or to capture at the socket layer with
   Wireshark **only if** the connection is plaintext to a local proxy you
   terminated yourself. Do not attempt to defeat TLS verification on a
   third-party service; if you cannot read the frames with your own trusted
   proxy, stop here and use the Gemini API provider instead.
3. Trigger one short utterance ("hello world") and let it finalize.
4. You should see a WebSocket upgrade plus frames in both directions.

## Fill in protocol.json

Record only what you actually saw:

| Field | Where it comes from |
| --- | --- |
| `transport.wsUrl` | the WebSocket request URL, without query string |
| `transport.query` | query params, `{accessToken}` allowed as a placeholder |
| `transport.headers` | request headers you saw, minus anything already unique per-session |
| `audio.binary` | `true` if raw PCM frames were sent, `false` if JSON-wrapped |
| `audio.message` | the JSON envelope, with the base64 payload replaced by `"{base64}"` |
| `control.keepalive` | the keepalive message and its interval, if the server idles sockets out |
| `control.start` | the first message sent after the socket opens, if any |
| `control.commit` | the message that signals end-of-utterance |
| `transport.keytermsParam` | query param that repeats per glossary term, or `""` if unsupported |
| `receive.typePath` | dot path to the message kind, e.g. `"type"` |
| `receive.textPath` | dot path to the transcript text, e.g. `"data"` |
| `receive.interimTypes` | kind values that carry a running transcript |
| `receive.finalTypes` | kind values that end the utterance (may carry no text; then the last interim is promoted) |
| `receive.errorTypes` | kind values that report an error |
| `receive.errorPaths` | dot paths tried in order for the error detail |

Then set `"verified": true`, `"capturedAt"` to the epoch ms of the capture, and
`"capturedFrom"` to something honest like `"claude-code 2.x, own account"`.

## Sanity checks before trusting it

- `npm run probe -- claude-voice` sends one second of silence and reports the
  round trip. Silent audio should transcribe to `""` **without** an error. If you
  get text back from silence, you are decoding the wrong field.
- Dictate "use state and post gress client" with `useState` and `postgres` in
  your glossary. The corrector should fix both.
- Kill the network mid-utterance. The fallback provider should still produce the
  sentence, because the audio was buffered.

## When it breaks

Provider fields change without announcement. Re-capture, diff `protocol.json`,
ship the JSON. That is the entire maintenance story, and it is why this file is
JSON and not TypeScript.

If the schema turns out to require defeating TLS verification or forging client
attestation, stop: that is the line between "reading your own traffic" and
"circumventing access controls", and the Gemini API provider is right there.
