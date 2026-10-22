import os
from pathlib import Path
import gradio as gr  
from qwen_asr import Qwen3ASRModel
import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = ROOT / "models" / "qwen_model"

if not os.path.exists(LOCAL_MODEL_PATH) or not os.listdir(LOCAL_MODEL_PATH):
    raise FileNotFoundError("❌ Local model folder is empty! Please run your download script first.")

print("⏳ Loading Qwen3-ASR model natively from local folder...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = Qwen3ASRModel.from_pretrained(
    LOCAL_MODEL_PATH, 
    dtype=torch_dtype,
    device_map=device,
    trust_remote_code=True
)

print(f"✅ Local Model loaded successfully on {device.upper()}!")


def format_timestamps(results):
    """Parses Qwen3 ASR object payloads to format clear word-level timelines."""
    # Handle single item wrapped in array block
    if isinstance(results, list) and len(results) > 0:
        results = results[0]
        
    text_out = ""
    time_out = ""
    
    # Extract structural properties from custom Qwen class object
    if hasattr(results, "text"):
        text_out = results.text
        
    if hasattr(results, "words") and results.words:
        # Loop through each word object array mapping timestamps
        time_lines = []
        for w in results.words:
            word_str = getattr(w, "word", "")
            start = getattr(w, "start", 0.0)
            end = getattr(w, "end", 0.0)
            time_lines.append(f"⏱️ [{start:05.2f}s -> {end:05.2f}s] : {word_str}")
        time_out = "\n".join(time_lines)
    else:
        time_out = "ℹ️ No timestamp offsets returned for this audio snapshot. Ensure clean audio file input."
        
    return text_out, time_out


def process_audio(audio_source, mic_file, uploaded_files, language):
    """Routes audio targets from active tab pipelines into the local model engine."""
    # Determine which tab component contains active audio array file paths
    active_file = mic_file if audio_source == "Microphone" else uploaded_files
    
    if not active_file:
        return "❌ Empty target.", "❌ Please record audio or upload a valid media clip first."
        
    try:
        lang_tag = "Arabic" if language == "Arabic" else "English"
        
        # Call Qwen transcription pipeline with dynamic context tags
        results = model.transcribe(
            audio=active_file,
            language=lang_tag
        )
        
        # Parse return variables cleanly using our parsing method
        return format_timestamps(results)

    except Exception as e:
        return "❌ Error occurred.", f"❌ Execution Exception details: {str(e)}"


# --- Gradio Block Web View Layout Builder ---
with gr.Blocks(title="Qwen Advanced Offline STT") as demo:
    gr.Markdown("# 🎙️ Professional Offline Speech Processing Station")
    gr.Markdown("Powered by Alibaba's **Qwen3-ASR-1.7B**. Running 100% offline local compute.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ 1. Input Controls")
            lang_selection = gr.Radio(
                choices=["Arabic", "English"], 
                value="Arabic", 
                label="🌐 Target Audio Language"
            )
            
            # State tracker component to manage active tabs cleanly
            source_selector = gr.State("Microphone")
            
            with gr.Tabs() as tabs:
                with gr.Tab("🎤 Microphone Setup") as tab1:
                    mic_input = gr.Audio(
                        sources=["microphone"], 
                        type="filepath", 
                        label="Record Audio Stream"
                    )
                with gr.Tab("📁 Batch File Upload") as tab2:
                    file_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Drop Audio File Here (.mp3, .wav, .m4a)"
                    )
            
            # Event state routing when tabs shift focus 
            tab1.select(fn=lambda: "Microphone", outputs=source_selector)
            tab2.select(fn=lambda: "Upload", outputs=source_selector)
            
            submit_btn = gr.Button("Execute Local Analysis Pipeline", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 📝 2. Model Diagnostics Output")
            text_output = gr.Textbox(
                label="Plain Transcript Paragraph block", 
                lines=4, 
                placeholder="Transcribed continuous text paragraphs display here..."
            )
            
            timestamp_output = gr.Textbox(
                label="⏱️ Word-Level Structural Timestamps", 
                lines=10, 
                placeholder="Word timeline boundaries will render here..."
            )
            
    submit_btn.click(
        fn=process_audio, 
        inputs=[source_selector, mic_input, file_input, lang_selection], 
        outputs=[text_output, timestamp_output]
    )

if __name__ == "__main__":
    # Max file limit configuration set to preserve offline target memory limits
    demo.launch(max_file_size="50mb")
