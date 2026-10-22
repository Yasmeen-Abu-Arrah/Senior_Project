# Models

Downloaded model files live here. This directory is ignored by Git because the
weights are several GB.

Use the downloader from the repository root:

```powershell
python scripts\download_models.py text
python scripts\download_models.py asr
python scripts\download_models.py whisper --whisper-model base
python scripts\download_models.py face
```

Expected layout:

```text
models/
├── face_landmarker.task       # MediaPipe face landmarker for smart screening
├── kokoro_cached_model/       # Optional local Kokoro cache (not used by current apps)
├── qwen_model/                # Qwen3-ASR-1.7B
├── qwen_text_model/           # Qwen2.5-3B-Instruct
├── stutter_classifier/        # Custom wav2vec2 stutter classifier
└── whisper_models/            # Whisper .pt files
```

The model folders and `face_landmarker.task` are copied here for local offline
runs, but Git ignores all of them. Teammates who clone the repository must run
the download script or receive the files separately.

`stutter_classifier/` is a trained project artifact and is not available from a
public download. Copy that folder from the project archive if it is missing.

Do not commit files from this directory.
