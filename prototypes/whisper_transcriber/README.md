# Whisper Transcriber

Offline transcription with a user-selected local Whisper model.

## Setup

```powershell
python scripts\download_models.py whisper --whisper-model base
python prototypes\whisper_transcriber\app.py
```

Download other sizes as needed:

```powershell
python scripts\download_models.py whisper --whisper-model small
python scripts\download_models.py whisper-all
```

`whisper-all` is large. Models are stored in `models/whisper_models`.

## Use

1. Select a model from the dropdown.
2. Choose English or Arabic.
3. Record audio from the microphone.
4. Click transcribe.
