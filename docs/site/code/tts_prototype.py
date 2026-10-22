import os
import soundfile as sf
import gradio as gr
import torch
from habibi_tts.infer.utils_infer import HabibiTTS

REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_engine = HabibiTTS(model_type="Unified", device=device)

def generate_arabic_speech(text, dialect):
    if not text.strip():
        return None
    out_path = os.path.join(REPO_ROOT, "output.wav")
    audio, sr = tts_engine.infer(gen_text=text, dialect=dialect)
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    sf.write(out_path, audio, sr)
    return out_path

with gr.Blocks() as demo:
    gr.Markdown("# Habibi-TTS Arabic Speech")
    text_input = gr.Textbox(label="Arabic Text", lines=5)
    dialect_input = gr.Dropdown(choices=["MSA","EGY","LEV","SAU","UAE"], label="Dialect")
    audio_output = gr.Audio(label="Output", type="filepath")
    btn = gr.Button("Generate")
    btn.click(generate_arabic_speech, [text_input, dialect_input], audio_output)

if __name__ == "__main__":
    demo.launch(share=True)