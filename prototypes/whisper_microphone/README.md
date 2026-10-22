# Whisper Microphone

Minimal Arabic speech checker for the target word `قاف`.

## Setup

```powershell
python scripts\download_models.py whisper --whisper-model base
python prototypes\whisper_microphone\app.py
```

## Use

1. Record the target word.
2. Click the check button.
3. The app compares the Whisper transcript to the target word and reports a
   match, a likely qaf omission/substitution, or a mismatch.

This prototype uses Whisper `base` for speed. Use the transcriber prototype to
test larger models.
