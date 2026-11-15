#!/usr/bin/env sh
# Start Voice Dictation without a terminal window on macOS/Linux.
# Close it from the tray menu (Quit) or with: pkill -f dictate.py
set -eu

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi

nohup .venv/bin/python dictate.py >/dev/null 2>&1 &
echo "Voice Dictation started in the background (see dictate.log)."
