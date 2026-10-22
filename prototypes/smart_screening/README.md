# Smart Screening

Arabic speech-language screening prototype combining a decision-tree quick
screen, clinical scoring, MediaPipe lip tracking, and an articulation task wizard.

Run from the repository root:

```powershell
python prototypes\smart_screening\app.py
```

Needs `models/face_landmarker.task`:

```powershell
python scripts\download_models.py face
```
