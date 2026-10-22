import os
from pathlib import Path
import gradio as gx  # Gradio version 5+ uses gx, or change to 'gr' for older versions
import whisper

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "whisper_models"


def get_available_models():
    """Scans the local directory and returns a list of available models."""
    if not os.path.exists(MODELS_DIR):
        return []

    # Get all .pt filenames downloaded by your download script
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]

    # Strip the '.pt' extension to get clean model names (e.g., 'base', 'small.en')
    models = [os.path.splitext(f)[0] for f in files]
    return sorted(models)


def transcribe_audio(audio_path, model_name, language):
    """Loads the chosen model offline and processes the recorded audio."""
    if audio_path is None:
        return "❌ Please record or upload an audio file first."

    if not model_name:
        return "❌ Please select a model from the dropdown menu."

    try:
        # Convert language selection to Whisper language codes
        lang_code = "en" if language == "English" else "ar"

        # Load the selected model strictly from your local folder
        model = whisper.load_model(model_name, download_root=str(MODELS_DIR))

        # Execute local transcription
        result = model.transcribe(audio_path, language=lang_code)
        return result["text"]

    except Exception as e:
        return f"❌ An error occurred during transcription: {str(e)}"


# --- Gradio UI Layout ---
available_models = get_available_models()

# Fallback string if your download script hasn't run yet
if not available_models:
    available_models = ["No local models found! Run your download script first."]

with gx.Blocks(title="Offline Whisper Transcriber") as demo:
    gx.Markdown("# 🎙️ Offline Whisper Audio Transcriber")
    gx.Markdown(
        "Select your locally downloaded model, record from your microphone, and view the transcription."
    )

    with gx.Row():
        with gx.Column():
            # 1. Configuration Inputs
            model_dropdown = gx.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="📁 Select Local Model",
                info="Scanned dynamically from models/whisper_models.",
            )

            lang_radio = gx.Radio(
                choices=["English", "Arabic"],
                value="English",
                label="🌐 Target Language",
            )

            # 2. Audio Input Component (Handles Mic recording)
            audio_input = gx.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Record Audio",
            )

            transcribe_btn = gx.Button("Transcribe Audio", variant="primary")

        with gx.Column():
            # 3. Output Window
            text_output = gx.Textbox(
                label="📝 Transcription Output",
                placeholder="Your text will appear here...",
                lines=8,
            )

    # Trigger action on button click
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_dropdown, lang_radio],
        outputs=text_output,
    )

if __name__ == "__main__":
    demo.launch()
