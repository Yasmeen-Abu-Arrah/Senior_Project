import time
from importlib.resources import files
from pathlib import Path

import gradio as gr
import soundfile as sf
import tomli
import torch
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT

from habibi_tts.infer.utils_infer import infer_process
from habibi_tts.model.utils import dialect_id_map

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
DIALECTS = ["MSA", "SAU", "UAE", "ALG", "IRQ", "EGY", "MAR", "OMN", "TUN", "LEV", "SDN", "LBY", "UNK"]

_engine = None


def _load_reference():
    config_path = files("habibi_tts").joinpath("infer/example.toml")
    with open(config_path, "rb") as handle:
        config = tomli.load(handle)
    ref_audio = str(files("habibi_tts").joinpath("assets/MSA.mp3"))
    return preprocess_ref_audio_text(ref_audio, config["ref_text"], show_info=print)


def get_engine():
    global _engine
    if _engine is None:
        print(f"Loading Habibi-TTS Unified model on {DEVICE.upper()}...")
        checkpoint = str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"))
        vocab = str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt"))
        model = load_model(
            DiT,
            MODEL_CFG,
            checkpoint,
            mel_spec_type="vocos",
            vocab_file=vocab,
            device=DEVICE,
        )
        vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=False,
            local_path="",
            device=DEVICE,
        )
        ref_audio, ref_text = _load_reference()
        _engine = (model, vocoder, ref_audio, ref_text)
    return _engine


def generate_speech(text, dialect):
    if not text.strip():
        return None

    model, vocoder, ref_audio, ref_text = get_engine()
    audio, sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        text,
        model,
        vocoder,
        mel_spec_type="vocos",
        device=DEVICE,
        dialect_id=dialect_id_map[dialect],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"habibi_{int(time.time())}.wav"
    sf.write(output_path, audio, sample_rate)
    return str(output_path)


with gr.Blocks(title="Habibi-TTS Arabic Speech") as demo:
    gr.Markdown("# Habibi-TTS Arabic Speech")
    gr.Markdown("First generation downloads the Unified checkpoint and vocoder into the Hugging Face cache.")
    text_input = gr.Textbox(label="Arabic Text", lines=5)
    dialect_input = gr.Dropdown(choices=DIALECTS, value="MSA", label="Dialect")
    generate_button = gr.Button("Generate", variant="primary")
    audio_output = gr.Audio(label="Output", type="filepath")
    generate_button.click(generate_speech, [text_input, dialect_input], audio_output)


if __name__ == "__main__":
    demo.launch(share=True)
