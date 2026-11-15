"""Download the configured ASR model so the app can run offline.

dictate.py forces Hugging Face offline mode, so the model must exist in the
local cache before the first run. Run this once per model:

    .venv\\Scripts\\python.exe download_model.py
    .venv\\Scripts\\python.exe download_model.py --model Qwen/Qwen3-ASR-0.6B
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# This is the one place that is allowed to touch the network.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"


def configured_model() -> str:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle).get("model_id") or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-download an ASR model.")
    parser.add_argument("--model", default=None, help="model id or local path")
    args = parser.parse_args(argv)

    model_id = args.model or configured_model()
    print(f"Downloading {model_id} (this is a one-time, multi-GB download)...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is missing; run the setup script for your system "
            "(windows\\setup.bat, macos/setup.sh or linux/setup.sh) first.",
            file=sys.stderr,
        )
        return 1

    try:
        path = snapshot_download(model_id)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Check your internet connection and that the model id is correct.",
            file=sys.stderr,
        )
        return 1

    print(f"Model cached at: {path}")
    print("You can now start the app; it no longer needs the network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
