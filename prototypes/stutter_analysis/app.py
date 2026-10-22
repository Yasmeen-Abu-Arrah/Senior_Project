"""
Stutter Detection — Gradio Demo  (Live + File)
================================================
• Live tab  : streams microphone audio, runs inference every ~0.5 s on the
              latest 2-second window, and renders a colour-coded waveform
              showing where each stutter type was detected.
• File tab  : uploads a clip, runs a full sliding-window analysis, and draws
              the same segmented waveform.

Run:
    python prototypes/stutter_analysis/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

from stutter_model_trainer import (
    load_model,
    SAMPLE_RATE,
    MAX_AUDIO_LEN,
    LABEL_NAMES,
    NUM_CLASSES,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).resolve().parents[2] / "models" / "stutter_classifier"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sliding-window parameters (per-segment visualisation)
WINDOW_SEC      = 1.0    # analysis window (seconds)
HOP_SEC         = 0.5    # hop between windows (seconds)
LIVE_BUFFER_SEC = 10.0   # max audio kept in the live rolling buffer

# Per-class display names (Arabic + English)
CLASS_LABELS: Dict[str, str] = {
    "fluent":       "Fluent  (طلاقة)",
    "repetition":   "Repetition  (تكرار)",
    "prolongation": "Prolongation  (إطالة)",
    "block":        "Block  (توقف)",
}

# Per-class colours used in the waveform plot
CLASS_COLORS: Dict[str, str] = {
    "fluent":       "#4CAF50",   # green
    "repetition":   "#FF9800",   # amber
    "prolongation": "#2196F3",   # blue
    "block":        "#F44336",   # red
}

# ─────────────────────────────────────────────────────────────────────────────
#  Load model once at startup
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading model from '{MODEL_DIR}' on {DEVICE} …")
model, feature_extractor = load_model(str(MODEL_DIR), device=DEVICE)
print("Model ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Core inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _infer(audio: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Run the classifier on a 1-D float32 waveform at SAMPLE_RATE Hz.
    Clips are truncated / zero-padded to MAX_AUDIO_LEN internally.
    Returns (predicted_label, {display_label: confidence}).
    """
    max_samples = int(MAX_AUDIO_LEN * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values)
        probs  = torch.softmax(logits, dim=-1)[0]

    pred_label = LABEL_NAMES[int(probs.argmax())]
    confs = {
        CLASS_LABELS.get(LABEL_NAMES[i], LABEL_NAMES[i]): float(probs[i])
        for i in range(NUM_CLASSES)
    }
    return pred_label, confs


def _sliding_window(audio: np.ndarray) -> List[Tuple[float, float, str]]:
    """
    Run sliding-window inference over *audio*.
    Returns [(start_s, end_s, label), …] — one tuple per window.
    """
    win = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(HOP_SEC    * SAMPLE_RATE)
    segments: List[Tuple[float, float, str]] = []

    for start in range(0, max(1, len(audio) - win // 2), hop):
        end   = start + win
        chunk = audio[start:end]
        if len(chunk) < win // 2:
            break
        label, _ = _infer(chunk)
        segments.append((start / SAMPLE_RATE, end / SAMPLE_RATE, label))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
#  Waveform plot
# ─────────────────────────────────────────────────────────────────────────────
def _make_plot(
    audio: np.ndarray,
    segments: List[Tuple[float, float, str]],
    title: str = "Waveform — Stutter Segmentation",
):
    """
    Draw waveform with colour-coded background bands per segment.
    Returns a matplotlib Figure.
    """
    duration = len(audio) / SAMPLE_RATE
    t = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(11, 3), facecolor="#1e1e2e")
    ax.set_facecolor("#1a1b2e")

    # Colour bands (one per segment)
    for (t0, t1, label) in segments:
        color = CLASS_COLORS.get(label, "#888888")
        ax.axvspan(t0, min(t1, duration), alpha=0.28, color=color, linewidth=0)

    # Waveform line
    ax.plot(t, audio, color="#c0caf5", linewidth=0.7, alpha=0.92)

    # Thin vertical ticks at window boundaries
    for (t0, _, _) in segments[1:]:
        ax.axvline(t0, color="#414868", linewidth=0.8, alpha=0.5)

    # Axes styling
    ax.set_xlim(0, max(duration, 0.01))
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Time (s)", color="#a9b1d6", fontsize=9)
    ax.set_ylabel("Amplitude", color="#a9b1d6", fontsize=9)
    ax.tick_params(colors="#a9b1d6", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#414868")

    # Legend
    patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[k], label=CLASS_LABELS[k], alpha=0.75)
        for k in CLASS_COLORS
    ]
    ax.legend(
        handles=patches, loc="upper right", fontsize=7.5,
        facecolor="#24283b", edgecolor="#414868",
        labelcolor="#c0caf5", framealpha=0.92,
    )
    ax.set_title(title, color="#c0caf5", fontsize=10, pad=6)
    fig.tight_layout(pad=0.8)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  File-upload handler
# ─────────────────────────────────────────────────────────────────────────────
def analyse_file(audio_path: Optional[str]):
    if audio_path is None:
        return None, {}

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.9

    segments = _sliding_window(audio)
    if not segments:
        label, confs = _infer(audio)
        segments = [(0.0, len(audio) / SAMPLE_RATE, label)]
    else:
        _, confs = _infer(audio)

    fig = _make_plot(audio, segments, title="Stutter Segmentation — File")
    return fig, confs


# ─────────────────────────────────────────────────────────────────────────────
#  Live-streaming handlers
# ─────────────────────────────────────────────────────────────────────────────
# state schema: {"audio": np.ndarray | None, "segments": list}

def _empty_state() -> Dict[str, Any]:
    return {"audio": None, "segments": []}


def stream_audio(
    stream_chunk: Optional[Tuple[int, np.ndarray]],
    state: Dict[str, Any],
):
    """
    Called by Gradio on every microphone chunk.
    Returns (new_state, waveform_figure, confidence_dict).
    """
    if stream_chunk is None:
        return state, None, {}

    sr, chunk_np = stream_chunk
    chunk_np = chunk_np.astype(np.float32)

    # Stereo → mono
    if chunk_np.ndim > 1:
        chunk_np = chunk_np.mean(axis=1)

    # Resample if the browser delivers a different sample rate
    if sr != SAMPLE_RATE:
        chunk_np = librosa.resample(chunk_np, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Normalise chunk
    peak = np.max(np.abs(chunk_np))
    if peak > 1e-6:
        chunk_np = chunk_np / peak * 0.9

    # Accumulate into rolling buffer
    prev = state.get("audio")
    buf  = chunk_np if prev is None else np.concatenate([prev, chunk_np])

    max_buf = int(LIVE_BUFFER_SEC * SAMPLE_RATE)
    if len(buf) > max_buf:
        buf = buf[-max_buf:]

    # Need at least one full window before inference
    if len(buf) < int(WINDOW_SEC * SAMPLE_RATE):
        new_state = {"audio": buf, "segments": state.get("segments", [])}
        return new_state, None, {}

    # Sliding-window inference over the whole buffer
    segments = _sliding_window(buf)

    # Confidence dict for the latest 1-second window
    last_chunk = buf[-int(WINDOW_SEC * SAMPLE_RATE):]
    _, confs   = _infer(last_chunk)

    fig       = _make_plot(buf, segments, title="Live Stutter Analysis")
    new_state = {"audio": buf, "segments": segments}
    return new_state, fig, confs


def clear_live():
    return _empty_state(), None, {}


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Stutter Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🎙️ Stutter Detection — Arabic Speech
        | Tab | How to use |
        |-----|-----------|
        | **🎤 Live Microphone** | Press the mic icon — the waveform updates in real time and colour-codes each 1 s window by stutter type |
        | **📁 File Upload** | Upload a WAV / MP3 / FLAC clip for full segmented analysis |
        """
    )

    live_state = gr.State(value=_empty_state())

    with gr.Tabs():

        # ── Tab 1 : Live microphone ────────────────────────────────────────
        with gr.Tab("🎤 Live Microphone"):

            with gr.Row():
                mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone  (click the mic icon to start recording)",
                )
                with gr.Column(scale=1):
                    live_label = gr.Label(
                        num_top_classes=NUM_CLASSES,
                        label="Current Prediction  (latest 1 s window)",
                    )
                    clear_btn = gr.Button("🗑  Clear / Reset", variant="secondary")

            live_plot = gr.Plot(
                label="Waveform — colour = detected stutter type per window",
            )

            gr.Markdown(
                "**Colour key:** "
                "🟢 Fluent &nbsp;&nbsp; "
                "🟠 Repetition &nbsp;&nbsp; "
                "🔵 Prolongation &nbsp;&nbsp; "
                "🔴 Block"
            )

            mic.stream(
                fn=stream_audio,
                inputs=[mic, live_state],
                outputs=[live_state, live_plot, live_label],
            )
            clear_btn.click(
                fn=clear_live,
                outputs=[live_state, live_plot, live_label],
            )

        # ── Tab 2 : File upload ────────────────────────────────────────────
        with gr.Tab("📁 File Upload"):

            with gr.Row():
                file_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload Audio",
                )
                file_label = gr.Label(
                    num_top_classes=NUM_CLASSES,
                    label="Overall Prediction",
                )

            file_plot = gr.Plot(
                label="Waveform — colour = detected stutter type per window",
            )

            analyse_btn = gr.Button("Analyse", variant="primary")

            analyse_btn.click(
                fn=analyse_file,
                inputs=file_input,
                outputs=[file_plot, file_label],
            )
            file_input.change(
                fn=analyse_file,
                inputs=file_input,
                outputs=[file_plot, file_label],
            )

            # Example clips from data/stuttered/
            example_wavs = sorted(
                (Path(__file__).resolve().parents[2] / "data" / "stuttered").rglob("*.wav")
            )[:6]
            if example_wavs:
                gr.Examples(
                    examples=[[str(p)] for p in example_wavs],
                    inputs=file_input,
                    outputs=[file_plot, file_label],
                    fn=analyse_file,
                    cache_examples=False,
                    label="Example Clips",
                )

    gr.Markdown(
        """
        ---
        **Model:** Wav2Vec2-base fine-tuned on synthetic Arabic stuttered speech &nbsp;|&nbsp;
        **Test accuracy:** 98.3 % &nbsp;|&nbsp; **Macro-F1:** 0.983
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
