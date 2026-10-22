# Qwen3-ASR

Offline Arabic/English transcription with Qwen3-ASR-1.7B.

## Setup

```powershell
python scripts\download_models.py asr
python prototypes\qwen_asr\app.py
```

The model is loaded from `models/qwen_model`.

## Use

1. Choose Arabic or English.
2. Record from the microphone or upload an audio file.
3. Click the analysis button.
4. Read the transcript in the output panel.

Word-level timestamps require a forced aligner that is not part of this
prototype, so the timestamp panel is informational only.
