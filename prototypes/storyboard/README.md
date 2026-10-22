# Storyboard Generator

Generates a story and renders it as an Arabic storyboard using the local Amiri font.

## Setup

```powershell
python scripts\download_models.py text font
python prototypes\storyboard\app.py
```

The model is loaded from `models/qwen_text_model`. The font is loaded from
`assets/fonts/Amiri-Regular.ttf`.

## Use

1. Enter the target therapy words.
2. Choose a story theme.
3. Click the generate button.
4. The output is rendered as an RTL storyboard with therapy words highlighted.

This is the merged story-generation and presentation prototype.
