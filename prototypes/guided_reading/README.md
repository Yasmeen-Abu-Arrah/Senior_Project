# Guided Reading Stutter Detector

Generates an Arabic reference sentence with Habibi-TTS, then compares a child's
reading against it using Whisper word timestamps to flag prolongations, blocks,
and repetitions.

Run from the repository root:

```powershell
python prototypes\guided_reading\app.py
```

Needs `models/whisper_models` and the Habibi-TTS Unified checkpoint plus vocoder
cache. See `scripts/download_models.py whisper` and `python scripts/download_models.py habibi`.
Set `FFMPEG_BIN` if ffmpeg is not already on `PATH`.
