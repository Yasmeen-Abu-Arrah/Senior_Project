# Multimodal Therapy

Combines Whisper audio analysis with MediaPipe lip tracking for a therapy-style
assessment.

## Setup

```powershell
python scripts\download_models.py whisper --whisper-model base
python prototypes\multimodal_therapy\app.py
```

A webcam is required. The app loads Whisper from `models/whisper_models` and
initializes MediaPipe Face Mesh locally.

## Use

1. Step 1: position the child's face in the camera and wait for a stable mouth
   reading.
2. Step 2: record the target word.
3. Step 3: run the multimodal evaluation and read the clinical report.

The target word and thresholds are defined in `AppConfig` at the top of
`app.py`.
