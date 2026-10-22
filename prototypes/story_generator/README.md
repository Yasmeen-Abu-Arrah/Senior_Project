# Story Generator

Generates safe Arabic children's stories with Qwen2.5-3B-Instruct.

## Setup

```powershell
python scripts\download_models.py text
python prototypes\story_generator\app.py
```

The model is loaded from `models/qwen_text_model`. The first load needs several GB of RAM/VRAM.

## Use

1. Enter the target therapy words.
2. Choose a story theme.
3. Click the generate button.
4. Copy the generated Arabic story from the output box.

The app opens a Gradio page and also creates a public share link while running.
