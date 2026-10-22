# Habibi-TTS

Arabic text-to-speech with the Habibi-TTS Unified model.

## Setup

Install the environment first, then cache the checkpoint:

```powershell
python scripts\download_models.py habibi
python prototypes\habibi_tts\app.py
```

The first generation loads the Unified checkpoint and vocos vocoder into the
Hugging Face cache. This can take time and several GB of disk space.

## Use

1. Enter Arabic text.
2. Choose a dialect tag.
3. Click generate.
4. The generated WAV is written to `outputs/` and played in the browser.

This app intentionally uses only the Unified checkpoint. The official Habibi
Gradio UI loads every dialect checkpoint and is therefore much heavier.
