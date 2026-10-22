# Interactive Story

Arabic children's story prototype with scene narration, speech recognition,
pronunciation comparison, and a simple reading game.

Run from the repository root:

```powershell
python prototypes\interactive_story\app.py
```

Needs `models/whisper_models` with the `turbo` model. Story narration uses
Google gTTS, so internet access is required for generated audio. Scenes currently
use generated placeholder images until real `image_path` values are supplied.
