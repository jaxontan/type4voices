#!/usr/bin/env sh
# macOS: create .venv, install the pinned dependencies and run the environment
# check. The app lives in the folder above this one, so everything is written
# there. For Windows use windows/setup.bat, for Linux linux/setup.sh.
set -eu

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "Python 3.10+ was not found. Install it, then run this script again."
    exit 1
fi

echo "Using $($PYTHON --version 2>&1) at $(command -v "$PYTHON")"

if [ ! -d .venv ]; then
    echo "Creating .venv (this takes a moment)..."
    "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Download the model once (needs ~4.4 GB and internet):"
echo "  .venv/bin/python download_model.py"
echo
.venv/bin/python dictate.py --doctor || true
