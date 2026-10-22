# Python environment

This project requires **Python 3.11**. The system default (Python 3.14) is not
supported by the pinned PyTorch, MediaPipe, and Whisper versions.

## Automatic setup

PowerShell:

```powershell
.\setup_venv.ps1
```

Command Prompt:

```bat
setup_venv.bat
```

Git Bash / Linux / macOS:

```bash
bash setup_venv.sh
```

Use `-Cpu` (PowerShell) or `--cpu` (batch/shell) to force CPU-only PyTorch.
Use `.\setup_venv.ps1 -Recreate` to rebuild an existing environment.

The scripts create `.venv`, install CUDA 12.8 PyTorch when `nvidia-smi` is
available, install `requirements.txt`, install the local Habibi-TTS package in
editable mode, and run `pip check`.

## Manual setup

```powershell
uv venv --python 3.11 --seed .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/Habibi-TTS
python -m pip check
```

Replace the CUDA PyTorch command with the following for CPU-only:

```powershell
python -m pip install torch==2.8.0 torchaudio==2.8.0
```

`ffmpeg` must be installed and available on `PATH` for Whisper and audio
decoding.

After installing the environment, download the assets:

```powershell
python scripts\download_models.py all
```

See the root `README.md` for individual model targets.
