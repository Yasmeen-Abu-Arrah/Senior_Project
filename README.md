# Senior Project

Speech and language therapy prototypes for Arabic-speaking children, including story generation, speech recognition, lip/mouth tracking, multimodal assessment, and Arabic text-to-speech.

## Requirements

- Git
- Python 3.11 (the setup script installs/provisions it through `uv` when available)
- ffmpeg on `PATH`
- Optional: NVIDIA GPU with a recent driver for CUDA 12.8 PyTorch

Do not use Python 3.12 or newer for the pinned environment; PyTorch 2.8, MediaPipe, and Whisper are installed for Python 3.11.

## Clone

```powershell
git clone https://github.com/Yasmeen-Abu-Arrah/Senior_Project.git
cd Senior_Project
```

## Install

Windows PowerShell:

```powershell
.\setup_venv.ps1
```

Windows Command Prompt:

```bat
setup_venv.bat
```

Linux / macOS / Git Bash:

```bash
bash setup_venv.sh
```

The setup script creates `.venv`, installs CUDA 12.8 PyTorch when `nvidia-smi` is available, installs `requirements.txt`, and installs the bundled Habibi-TTS package from `third_party/Habibi-TTS`.

Force CPU-only PyTorch:

```powershell
.\setup_venv.ps1 -Cpu
```

Recreate an existing environment:

```powershell
.\setup_venv.ps1 -Recreate
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

## Download models and assets

Large files are intentionally not committed. Download them after cloning:

```powershell
python scripts\download_models.py all
```

`all` downloads the Qwen text model, Qwen ASR model, Whisper `base`, the Amiri font, the MediaPipe face landmarker, and the Habibi-TTS Unified checkpoint. This is several GB and requires internet access.

Download only what a prototype needs:

```powershell
python scripts\download_models.py text
python scripts\download_models.py asr
python scripts\download_models.py whisper --whisper-model base
python scripts\download_models.py whisper --whisper-model small
python scripts\download_models.py whisper --whisper-model large-v3
python scripts\download_models.py whisper-all
python scripts\download_models.py font
python scripts\download_models.py face
python scripts\download_models.py habibi
```

List Whisper model names:

```powershell
python scripts\download_models.py --list
```

Downloaded files go to:

- `models/qwen_text_model`
- `models/qwen_model`
- `models/whisper_models`
- `models/face_landmarker.task`
- `assets/fonts/Amiri-Regular.ttf`
- Hugging Face cache for Habibi-TTS checkpoints

The stutter dataset is kept in `data/stuttered/` for local demos and training.
`data/` is ignored by Git.

`models/` and `assets/fonts/` are ignored by Git. Do not commit model weights, checkpoints, or generated audio.

The custom `models/stutter_classifier/` artifact is not publicly downloadable; copy it from the project archive when setting up stutter analysis.

## Run the prototypes

Run each app from the repository root with the virtual environment active.

| Folder | Command | Purpose |
| --- | --- | --- |
| `prototypes/story_generator` | `python prototypes/story_generator/app.py` | Generate safe Arabic stories with Qwen2.5-3B-Instruct |
| `prototypes/storyboard` | `python prototypes/storyboard/app.py` | Generate stories and render an Arabic storyboard with the Amiri font |
| `prototypes/qwen_asr` | `python prototypes/qwen_asr/app.py` | Offline Qwen3-ASR transcription |
| `prototypes/whisper_transcriber` | `python prototypes/whisper_transcriber/app.py` | Choose a local Whisper model and transcribe microphone audio |
| `prototypes/whisper_microphone` | `python prototypes/whisper_microphone/app.py` | Minimal Arabic speech checker using Whisper `base` |
| `prototypes/multimodal_therapy` | `python prototypes/multimodal_therapy/app.py` | Combine Whisper audio analysis with MediaPipe lip tracking |
| `prototypes/lips_webcam` | `python prototypes/lips_webcam/app.py` | Standalone OpenCV/MediaPipe mouth tracking demo |
| `prototypes/habibi_tts` | `python prototypes/habibi_tts/app.py` | Arabic text-to-speech with Habibi-TTS Unified |
| `prototypes/guided_reading` | `python prototypes/guided_reading/app.py` | Reference-guided reading with stutter detection |
| `prototypes/interactive_story` | `python prototypes/interactive_story/app.py` | Interactive Arabic story with narration and speech checks |
| `prototypes/pronunciation_analyzer` | `python prototypes/pronunciation_analyzer/app.py` | MFCC/DTW pronunciation comparison |
| `prototypes/smart_screening` | `python prototypes/smart_screening/app.py` | Clinical screening, lip tracking, and articulation tasks |
| `prototypes/stutter_analysis` | `python prototypes/stutter_analysis/app.py` | Live and file-based stutter classification |
| `prototypes/games/maze_craze_v1` | Open `index.html` in a browser | Browser maze game, version 1 |
| `prototypes/games/maze_craze_v2` | Open `index.html` in a browser | Browser maze game, version 2 |
| `prototypes/games/Brick Spell` | Open `index.html` in a browser | Brick-based spelling game |
| `prototypes/games/Coin Word Hunt` | Open `index.html` in a browser | Word-hunt platformer |
| `prototypes/games/Word Hunt Gallery` | Open `index.html` in a browser | Word shooting-gallery game |

Quarto prototype write-ups are preserved in `docs/prototypes/`.

Each prototype folder has its own `README.md` with inputs, outputs, and model requirements.

Original flat source files from both project folders are preserved under
`archive/` for reference. The `prototypes/` copies are the maintained,
path-corrected entry points.

## Repository layout

```text
Senior_Project/
├── archive/                 # Original source code from both projects
├── assets/
│   ├── audio/samples/       # Small audio samples
│   ├── diagrams/            # Architecture and workflow diagrams
│   ├── fonts/               # Downloaded locally, ignored by Git
│   └── images/              # Screenshots and figures
├── docs/                    # Environment notes, prototype write-ups, preserved site and documents
├── data/                    # Local stutter and Quran audio datasets, ignored by Git
├── models/                  # Downloaded models, ignored by Git
├── prototypes/              # One folder per runnable prototype, plus games
├── scripts/                 # Download and maintenance scripts
├── third_party/Habibi-TTS/  # Bundled Habibi-TTS source
├── requirements.txt
├── setup_venv.ps1
├── setup_venv.bat
└── setup_venv.sh
```

## Team workflow

- Commit source, documentation, diagrams, screenshots, and small assets.
- Do not commit `.venv/`, `models/`, `assets/fonts/*`, generated audio, or files matching `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.onnx`, `*.gguf`, `*.ckpt`.
- If a new model is needed, add a download target to `scripts/download_models.py` and document it in the relevant prototype README.
- Keep each prototype self-contained: its app, its README, and any small test assets.

## Troubleshooting

- `torch.cuda.is_available()` is false: install the CUDA build with `.\setup_venv.ps1` (without `-Cpu`), or accept CPU mode.
- `ffmpeg` errors: install ffmpeg and confirm `ffmpeg -version` works in the same terminal.
- Missing model error: run the matching download command from the section above.
- Habibi-TTS is slow on first launch: it downloads the Unified checkpoint and vocoder into the Hugging Face cache.
- If `.venv` has the wrong Python version: run `.\setup_venv.ps1 -Recreate`.
