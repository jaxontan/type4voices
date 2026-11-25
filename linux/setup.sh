#!/usr/bin/env sh
# Linux: create .venv, install the pinned dependencies and run the environment
# check. The app lives in the folder above this one, so everything is written
# there. For Windows use windows/setup.bat, for macOS macos/setup.sh.
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
    echo "============================================================"
    echo "Python 3.10+ was not found on your system."
    echo "Install it with your distribution's package manager:"
    if command -v apt >/dev/null 2>&1; then
        echo "  sudo apt update && sudo apt install -y python3 python3-venv python3-tk libportaudio2 xclip"
    elif command -v dnf >/dev/null 2>&1; then
        echo "  sudo dnf install -y python3 python3-tkinter portaudio-devel xclip"
    elif command -v pacman >/dev/null 2>&1; then
        echo "  sudo pacman -S python python-virtualenv tk portaudio xclip"
    else
        echo "  Please install Python 3.10+, venv, tkinter, portaudio, and xclip."
    fi
    echo "============================================================"
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
