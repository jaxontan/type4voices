#!/usr/bin/env sh
# Run Voice Dictation in the foreground (console output goes to the terminal).
# The tray icon appears after the model starts loading.
set -eu

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi

exec .venv/bin/python dictate.py "$@"
