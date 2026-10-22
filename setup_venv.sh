#!/usr/bin/env bash
# Create a Python 3.11 virtual environment and install all project requirements.
# Usage:
#   ./setup_venv.sh          # auto: CUDA wheels when nvidia-smi is found
#   ./setup_venv.sh --cpu    # force CPU-only PyTorch
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY=""
CPU_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --cpu|-cpu) CPU_ONLY=1 ;;
    esac
done

py_is_311() {
    "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>/dev/null
}

if [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
    if ! py_is_311 "$PY"; then
        echo "Existing .venv is not Python 3.11; recreating..."
        rm -rf "$VENV"
        PY=""
    fi
elif [ -x "$VENV/Scripts/python.exe" ]; then
    PY="$VENV/Scripts/python.exe"
    if ! py_is_311 "$PY"; then
        echo "Existing .venv is not Python 3.11; recreating..."
        rm -rf "$VENV"
        PY=""
    fi
fi

if [ -z "$PY" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "Creating Python 3.11 environment with uv..."
        uv venv --python 3.11 --seed "$VENV"
    elif command -v python3.11 >/dev/null 2>&1; then
        echo "Creating Python 3.11 environment with python3.11..."
        python3.11 -m venv "$VENV"
    elif command -v py >/dev/null 2>&1 && py -3.11 -c "import sys" >/dev/null 2>&1; then
        echo "Creating Python 3.11 environment with py launcher..."
        py -3.11 -m venv "$VENV"
    else
        echo "Python 3.11 was not found. Install Python 3.11 or uv and try again." >&2
        exit 1
    fi
    if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"; else PY="$VENV/Scripts/python.exe"; fi
fi

echo
echo "Using interpreter: $PY"
"$PY" -m pip install --upgrade pip setuptools wheel

if [ "$CPU_ONLY" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "Installing CUDA 12.8 PyTorch 2.8.0..."
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
else
    echo
    echo "Installing CPU PyTorch 2.8.0..."
    "$PY" -m pip install torch==2.8.0 torchaudio==2.8.0
fi

echo
echo "Installing requirements..."
"$PY" -m pip install -r requirements.txt

echo
echo "Installing local Habibi-TTS package..."
"$PY" -m pip install --no-deps -e ./third_party/Habibi-TTS

"$PY" -m pip check || echo "WARNING: pip check reported dependency issues (see above)."

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg was not found on PATH. Install it before using Whisper or audio decoding."
fi

echo
echo "Setup complete."
if [ -f "$VENV/bin/activate" ]; then
    echo "Activate with: source $VENV/bin/activate"
else
    echo "Activate with: source $VENV/Scripts/activate"
fi
echo "Download models/fonts with: python scripts/download_models.py all"
echo "Run prototypes from prototypes/<name>/README.md"
