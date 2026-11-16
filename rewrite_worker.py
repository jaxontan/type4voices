"""Run an English-fixer model in its own Python environment.

Gemma 4 needs transformers 5, but qwen-asr (the speech model) pins
transformers 4.57.6, so the two cannot share `.venv`. dictate.py starts this
script with `.venv-t5`'s Python and talks to it over stdin/stdout, one JSON
object per line:

    -> {"text": ..., "accent": ..., "terms": [...], "alternative": ..., "recent": ...}
    <- {"text": "..."}            or  {"error": "..."}

The first line it prints is {"ready": true} once the model is loaded and warm,
or {"error": "..."} if loading failed. Logs go to stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rewrite  # noqa: E402


def _send(out, payload: dict) -> None:
    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
    out.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s worker %(levelname)s %(message)s",
    )
    # stdout carries the protocol only; keep stray prints off it.
    out = sys.stdout
    sys.stdout = sys.stderr
    out.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    r = rewrite.Rewriter(args.model, args.device)
    if not r.load():
        _send(out, {"error": r.error or "could not load"})
        return 1
    _send(out, {"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            text = r.generate(
                req["text"], req.get("accent", ""), req.get("terms") or [],
                req.get("alternative", ""), req.get("recent", ""),
            )
            _send(out, {"text": text})
        except Exception as exc:  # keep serving; the app pastes the raw text
            logging.exception("generate failed")
            _send(out, {"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
