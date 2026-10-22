import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FONT_DIR = ROOT / "assets" / "fonts"
FONT_PATH = FONT_DIR / "Amiri-Regular.ttf"
AMIRI_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

QWEN_TEXT_REPO = "Qwen/Qwen2.5-3B-Instruct"
QWEN_ASR_REPO = "Qwen/Qwen3-ASR-1.7B"


def download_hf(repo_id, target, ignore_patterns=None):
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=ignore_patterns or [],
    )
    print(f"Done: {target}")


def download_text():
    download_hf(
        QWEN_TEXT_REPO,
        MODELS_DIR / "qwen_text_model",
        ignore_patterns=["*.msgpack", "*.h5", "*.gguf"],
    )


def download_asr():
    download_hf(
        QWEN_ASR_REPO,
        MODELS_DIR / "qwen_model",
        ignore_patterns=["*.msgpack", "*.h5"],
    )


def download_whisper(model_names, force=False):
    import whisper

    target = MODELS_DIR / "whisper_models"
    target.mkdir(parents=True, exist_ok=True)
    available = whisper._MODELS

    for name in model_names:
        if name not in available:
            raise SystemExit(f"Unknown Whisper model '{name}'. Available: {', '.join(sorted(available))}")
        destination = target / Path(available[name]).name
        if destination.exists() and not force:
            print(f"Whisper {name}: already present ({destination.name})")
            continue
        print(f"Downloading Whisper {name}...")
        whisper._download(available[name], str(target), in_memory=False)
        print(f"Done: {destination}")


def download_font(force=False):
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    if FONT_PATH.exists() and FONT_PATH.stat().st_size > 100_000 and not force:
        print(f"Font already present: {FONT_PATH}")
        return
    print(f"Downloading Amiri font -> {FONT_PATH}")
    urllib.request.urlretrieve(AMIRI_URL, FONT_PATH)
    print(f"Done: {FONT_PATH}")


def download_face(force=False):
    destination = MODELS_DIR / "face_landmarker.task"
    if destination.exists() and destination.stat().st_size > 1_000_000 and not force:
        print(f"Face landmarker already present: {destination}")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe face landmarker -> {destination}")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, destination)
    print(f"Done: {destination}")


def download_habibi():
    from huggingface_hub import hf_hub_download

    files = [
        ("SWivid/Habibi-TTS", "Unified/model_200000.safetensors"),
        ("SWivid/Habibi-TTS", "Unified/vocab.txt"),
        ("charactr/vocos-mel-24khz", "config.yaml"),
        ("charactr/vocos-mel-24khz", "pytorch_model.bin"),
    ]
    for repo_id, filename in files:
        print(f"Downloading {repo_id}/{filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename)
    print("Habibi-TTS Unified checkpoint and vocoder are cached.")


def main():
    parser = argparse.ArgumentParser(description="Download models, fonts, and checkpoints used by the prototypes.")
    parser.add_argument(
        "component",
        nargs="?",
        default="all",
        choices=["all", "text", "asr", "whisper", "whisper-all", "font", "face", "habibi"],
        help="What to download. 'all' includes text, asr, Whisper base, font, face landmarker, and Habibi.",
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper model for the 'whisper' target (default: base).")
    parser.add_argument("--force", action="store_true", help="Redownload Whisper files, the font, and the face landmarker.")
    parser.add_argument("--list", action="store_true", help="List Whisper model names and exit.")
    args = parser.parse_args()

    if args.list:
        import whisper

        print("\n".join(sorted(whisper._MODELS)))
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.component in ("all", "text"):
        download_text()
    if args.component in ("all", "asr"):
        download_asr()
    if args.component in ("all", "whisper"):
        download_whisper([args.whisper_model], force=args.force)
    if args.component == "whisper-all":
        import whisper

        download_whisper(sorted(whisper._MODELS), force=args.force)
    if args.component in ("all", "font"):
        download_font(force=args.force)
    if args.component in ("all", "face"):
        download_face(force=args.force)
    if args.component in ("all", "habibi"):
        download_habibi()

    print("\nAll requested assets are ready.")
    print(f"Models: {MODELS_DIR}")
    print(f"Font:   {FONT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
