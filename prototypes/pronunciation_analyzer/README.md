# Pronunciation Analyzer

Compares a child's pronunciation with a reference recording using Whisper
transcription, MFCC features, DTW alignment, and edit distance.

Run from the repository root:

```powershell
python prototypes\pronunciation_analyzer\app.py
```

Needs `models/whisper_models` with the `large-v3` model:

```powershell
python scripts\download_models.py whisper --whisper-model large-v3
```
