# Prototypes

Each folder is self-contained and has its own `README.md`.

| Folder | Needs | Run |
| --- | --- | --- |
| `story_generator` | Qwen text model | `python prototypes/story_generator/app.py` |
| `storyboard` | Qwen text model, Amiri font | `python prototypes/storyboard/app.py` |
| `qwen_asr` | Qwen ASR model | `python prototypes/qwen_asr/app.py` |
| `whisper_transcriber` | One or more Whisper models | `python prototypes/whisper_transcriber/app.py` |
| `whisper_microphone` | Whisper `base` | `python prototypes/whisper_microphone/app.py` |
| `multimodal_therapy` | Whisper `base`, webcam | `python prototypes/multimodal_therapy/app.py` |
| `lips_webcam` | Webcam | `python prototypes/lips_webcam/app.py` |
| `habibi_tts` | Habibi Unified checkpoint | `python prototypes/habibi_tts/app.py` |
| `guided_reading` | Whisper `turbo`, Habibi-TTS | `python prototypes/guided_reading/app.py` |
| `interactive_story` | Whisper `turbo`, internet for gTTS | `python prototypes/interactive_story/app.py` |
| `pronunciation_analyzer` | Whisper `large-v3` | `python prototypes/pronunciation_analyzer/app.py` |
| `smart_screening` | `face_landmarker.task` | `python prototypes/smart_screening/app.py` |
| `stutter_analysis` | `stutter_classifier` | `python prototypes/stutter_analysis/app.py` |
| `games/maze_craze_v1` | Browser | Open `index.html` |
| `games/maze_craze_v2` | Browser | Open `index.html` |
| `games/Brick Spell` | Browser | Open `index.html` |
| `games/Coin Word Hunt` | Browser | Open `index.html` |
| `games/Word Hunt Gallery` | Browser | Open `index.html` |

Quarto source documents for the original prototype write-ups are in
`docs/prototypes/`.

Download assets from the repository root:

```powershell
python scripts\download_models.py all
```

All Python apps resolve models and fonts relative to the repository root, so
they can be launched from any working directory.
