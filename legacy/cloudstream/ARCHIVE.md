# cloudstream — archived Electron prototype

This folder is the original TypeScript/Electron prototype, kept for reference.
It is **not** the product, is **not** built, and is **not** shipped.

The working app is the Python project two levels up (`type4voices`). The two
were one idea built twice, and this one was the less complete attempt: a mock
provider, a written-but-unverified Claude adapter, a Gemini Live adapter, and a
non-focusable HUD — but no local engine, no packaging, and no git history.

## What was folded into type4voices

| cloudstream idea | Where it lives now |
| --- | --- |
| Ordered provider chain | `stt_engine_order` in `config.json` |
| Full-clip buffer, replay into the next provider on failure | `App.transcribe_with_engine` in `dictate.py` |
| Provider catalog / priority resolution | `resolve_engine_order` + `ENGINE_IDS` in `dictate.py` |
| Gemini Live wiring | `gemini_stt.py` (already present, more complete) |
| Non-focusable HUD | `overlay.py` |
| Claude voice stream | `anthropic_stt.py` (kept; see the terms warning) |
| Loop selftests (`selftest`, `probe`) | `test_anthropic_stt.py`, `test_gemini_stt.py`, `dictate.py --doctor` |

## Why it is not deleted

The wire-schema notes under `tools/`, and `src/main/providers/claude/protocol.json`,
are the record of the Claude endpoint's shape. `anthropic_stt.py` keeps its
contract in one docstring, but this folder is the provenance.

Do not run `npm install` here expecting the product. The product is Python.
