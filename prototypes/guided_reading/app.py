"""
Guided-Reading Stutter Detector
================================
How it works
------------
1. Pick (or type) an Arabic sentence.
2. Click "Generate Reference" →
       • Habibi-TTS synthesises the sentence into clean audio.
       • Whisper runs on that TTS audio with word_timestamps=True to get
         the *expected* start / end time of every word.
3. Listen to the reference, then record yourself reading the same sentence.
4. Click "Analyse My Reading" →
       • Whisper runs on *your* recording with word_timestamps=True.
       • Each expected word is matched to your word via SequenceMatcher.
       • Three metrics are computed per word:
           - Duration ratio   (user_dur / ref_dur)  → Prolongation if > 1.8×
           - Extra pre-word silence                  → Block if > 0.40 s
           - Transcript insertions vs expected text  → Repetition
       • Results are shown as a colour-coded waveform + per-word table.

Run:
    python prototypes/guided_reading/app.py
"""

import os
import sys
import difflib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import librosa
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
import whisper
from importlib.resources import files as _pkg_files
from cached_path import cached_path
from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model import DiT
from habibi_tts.infer.utils_infer import infer_batch_process as _tts_infer_batch
from habibi_tts.model.utils import dialect_id_map as _dialect_id_map

# ─────────────────────────────────────────────────────────────────────────────
#  Paths & env
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
WHISPER_MODELS_DIR = str(ROOT / "models" / "whisper_models")
TEMP_DIR = ROOT / "temp_story"
TEMP_DIR.mkdir(exist_ok=True)

_FFMPEG_BIN = os.environ.get("FFMPEG_BIN")
if _FFMPEG_BIN:
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_SR = 16_000

# ─────────────────────────────────────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading Whisper (turbo) on {DEVICE} …")
_whisper_model = whisper.load_model(
    "turbo", device=DEVICE, download_root=WHISPER_MODELS_DIR
)

# Habibi-TTS — voice-cloning TTS built on F5-TTS.
# Requires a reference audio/text; we use the bundled MSA sample.
_V1_BASE_CFG   = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
_UNI_MODEL_PATH = str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"))
_UNI_VOCAB_PATH = str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt"))
_MSA_REF_AUDIO  = str(_pkg_files("habibi_tts").joinpath("assets/MSA.mp3"))
_MSA_REF_TEXT   = (
    "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، "
    "مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."
)

print("Loading Habibi-TTS model and vocoder …")
_tts_model = load_model(DiT, _V1_BASE_CFG, _UNI_MODEL_PATH, vocab_file=_UNI_VOCAB_PATH)
_vocoder   = load_vocoder()

# Pre-load the MSA reference audio with librosa so we never call
# torchaudio.load() (which requires the optional torchcodec package).
_ref_np, _ref_sr = librosa.load(_MSA_REF_AUDIO, sr=None, mono=True)
_ref_np = _ref_np[: 12 * _ref_sr]          # clip to ≤ 12 s like preprocess does
_ref_tensor = torch.from_numpy(_ref_np).unsqueeze(0)   # [1, T]
_MSA_REF_TEXT_PROC = _MSA_REF_TEXT.strip().rstrip(".") + ". "
print("All models ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS: Dict[str, str] = {
    "fluent":       "#4CAF50",
    "repetition":   "#FF9800",
    "prolongation": "#2196F3",
    "block":        "#F44336",
}
CLASS_LABELS: Dict[str, str] = {
    "fluent":       "Fluent (طلاقة)",
    "repetition":   "Repetition (تكرار)",
    "prolongation": "Prolongation (إطالة)",
    "block":        "Block (توقف)",
}

# Detection thresholds
PROLONG_RATIO    = 1.8   # user_dur / ref_dur above this → prolongation
BLOCK_EXTRA_GAP  = 0.40  # extra silence before word (s) above this → block
MIN_REF_DUR      = 0.06  # skip ratio check for very short reference words

# Acoustic-analysis thresholds (Tab 2)
ACOUSTIC_PROLONG_RATIO = 1.6    # duration ratio — confirmed by F0 flatness
ACOUSTIC_F0_STD_THRESH = 15.0   # Hz — F0 std below this = monotone/held → prolongation
ACOUSTIC_BLOCK_ENERGY  = 0.02   # RMS at word onset below this → block
ACOUSTIC_BLOCK_GAP     = 0.35   # extra pre-word silence (s) → block

PRESET_SENTENCES = [
    "أنا أذهب إلى المدرسة كل يوم",
    "القطار يمشي على السكة الحديدية",
    "أحب أن آكل التفاح الأحمر",
    "الشمس تشرق في الصباح الباكر",
    "أمي تطبخ طعاماً لذيذاً كل يوم",
    "الكتاب على الطاولة في الغرفة",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Audio utilities
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak * 0.90 if peak > 1e-6 else audio


def _save_wav(audio: np.ndarray, sr: int, name: str) -> str:
    path = str(TEMP_DIR / name)
    sf.write(path, audio, sr)
    return path


def _whisper_words(audio_path: str) -> Tuple[List[Dict], str]:
    """
    Transcribe audio_path with Whisper and return
    (word_list, full_text)  where word_list contains
    {"word": str, "start": float, "end": float} entries.
    """
    result = _whisper_model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        word_timestamps=True,
        temperature=0,
        beam_size=3,
        condition_on_previous_text=False,
        no_speech_threshold=0.40,
        fp16=(DEVICE == "cuda"),
    )
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                words.append({"word": text, "start": w["start"], "end": w["end"]})
    return words, result["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Alignment & stutter classification
# ─────────────────────────────────────────────────────────────────────────────
def _align_words(a_words: List[str], b_words: List[str]) -> List[Optional[int]]:
    """
    Map every position in *a_words* to an index in *b_words* (or None).
    Uses SequenceMatcher so insertions/deletions are handled gracefully.
    """
    matcher = difflib.SequenceMatcher(None, a_words, b_words, autojunk=False)
    mapping: List[Optional[int]] = [None] * len(a_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(i2 - i1, j2 - j1)):
                mapping[i1 + k] = j1 + k
    return mapping


def classify_words(
    expected: List[str],
    ref_words: List[Dict],
    user_words: List[Dict],
) -> List[Dict]:
    """
    Return one result dict per expected word with timing, ratio, and label.
    """
    ref_texts  = [w["word"] for w in ref_words]
    user_texts = [w["word"] for w in user_words]

    exp_to_ref  = _align_words(expected, ref_texts)
    exp_to_user = _align_words(expected, user_texts)

    # Any user-word index NOT in the exp→user mapping is an extra insertion
    matched_user_indices = {i for i in exp_to_user if i is not None}
    extra_user_indices   = set(range(len(user_words))) - matched_user_indices

    results: List[Dict] = []
    prev_ref_end  = 0.0
    prev_user_end = 0.0

    for i, exp_word in enumerate(expected):
        ref_w  = ref_words[exp_to_ref[i]]  if exp_to_ref[i]  is not None else None
        user_w = user_words[exp_to_user[i]] if exp_to_user[i] is not None else None

        ref_start = ref_w["start"]  if ref_w  else None
        ref_end   = ref_w["end"]    if ref_w  else None
        ref_dur   = (ref_end - ref_start) if (ref_start is not None) else None

        user_start = user_w["start"] if user_w else None
        user_end   = user_w["end"]   if user_w else None
        user_dur   = (user_end - user_start) if (user_start is not None) else None

        ref_gap  = max(0.0, (ref_start  - prev_ref_end))  if ref_start  is not None else 0.0
        user_gap = max(0.0, (user_start - prev_user_end)) if user_start is not None else 0.0
        extra_gap = user_gap - ref_gap

        ratio = (user_dur / ref_dur) if (
            user_dur is not None and ref_dur is not None and ref_dur > MIN_REF_DUR
        ) else None

        # ── label ──────────────────────────────────────────────────────────
        if user_w is None:
            label = "block"                          # word entirely skipped → block
        elif extra_gap > BLOCK_EXTRA_GAP:
            label = "block"                          # long pause before word
        elif ratio is not None and ratio > PROLONG_RATIO:
            label = "prolongation"                   # word held too long
        elif exp_to_user[i] is not None and any(
            j in extra_user_indices
            for j in range(max(0, exp_to_user[i] - 2), exp_to_user[i])
        ):
            label = "repetition"                     # extra words just before this one
        else:
            label = "fluent"

        results.append({
            "word":       exp_word,
            "ref_start":  ref_start,  "ref_end":   ref_end,
            "user_start": user_start, "user_end":  user_end,
            "ref_dur":    ref_dur,    "user_dur":  user_dur,
            "ratio":      ratio,      "extra_gap": extra_gap,
            "label":      label,
        })

        if ref_end   is not None: prev_ref_end  = ref_end
        if user_end  is not None: prev_user_end = user_end

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Acoustic Analysis helpers (Tab 2)
# ─────────────────────────────────────────────────────────────────────────────

def _refine_onset(audio: np.ndarray, sr: int, approx_start: float, approx_end: float) -> float:
    """
    Refine Whisper's approximate word start time using energy onset detection.
    Searches [approx_start − 50 ms, approx_end] for the first RMS frame that
    exceeds 5 % of the window's peak energy.
    """
    win_start = max(0.0, approx_start - 0.05)
    s = int(win_start * sr)
    e = int(min(approx_end, len(audio) / sr) * sr)
    segment = audio[s:e]
    if len(segment) < 256:
        return approx_start

    hop = 128
    rms = librosa.feature.rms(y=segment, frame_length=512, hop_length=hop)[0]
    peak = rms.max()
    if peak < 1e-6:
        return approx_start

    thresh = 0.05 * peak
    for idx, val in enumerate(rms):
        if val >= thresh:
            refined = win_start + idx * hop / sr
            return min(refined, approx_end - 0.02)
    return approx_start


def _extract_word_features(audio: np.ndarray, sr: int, start: float, end: float) -> Dict:
    """
    Extract RMS energy, F0 (probabilistic YIN), and ZCR for a word segment.
    Returns a dict with keys: rms_mean, rms_onset, f0_mean, f0_std, zcr_mean.
    """
    s = int(start * sr)
    e = int(min(end, len(audio) / sr) * sr)
    seg = audio[s:e]
    if len(seg) < 128:
        return {"rms_mean": 0.0, "rms_onset": 0.0,
                "f0_mean": 0.0, "f0_std": 999.0, "zcr_mean": 0.0}

    hop = 128
    rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=hop)[0]
    rms_mean  = float(rms.mean())
    onset_len = max(1, len(rms) // 5)          # first 20 % → onset energy
    rms_onset = float(rms[:onset_len].mean())

    try:
        f0, voiced_flag, _ = librosa.pyin(
            seg, fmin=75.0, fmax=400.0,
            sr=sr, frame_length=1024, hop_length=hop,
        )
        f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        f0_mean = float(f0_voiced.mean()) if len(f0_voiced) > 0 else 0.0
        f0_std  = float(f0_voiced.std())  if len(f0_voiced) > 1 else 999.0
    except Exception:
        f0_mean, f0_std = 0.0, 999.0

    zcr = librosa.feature.zero_crossing_rate(seg, hop_length=hop)[0]
    zcr_mean = float(zcr.mean())

    return {
        "rms_mean":  rms_mean,
        "rms_onset": rms_onset,
        "f0_mean":   f0_mean,
        "f0_std":    f0_std,
        "zcr_mean":  zcr_mean,
    }


def classify_words_acoustic(
    expected: List[str],
    ref_words: List[Dict],
    user_words: List[Dict],
    ref_audio: np.ndarray,
    ref_sr: int,
    user_audio: np.ndarray,
    user_sr: int,
) -> List[Dict]:
    """
    Classify each expected word using:
      1. Energy-refined word onset boundaries (instead of raw Whisper timestamps).
      2. F0 flatness (librosa.pyin) to confirm prolongation.
      3. Onset RMS energy to confirm blocks.

    Block       — word missing, OR pre-word silence > 0.35 s, OR onset energy < 0.02
    Prolongation — duration ratio > 1.6 AND (F0 std < 15 Hz  OR ratio > 2.16)
    Repetition  — extra Whisper insertions before the aligned word
    Fluent      — all checks pass
    """
    ref_texts  = [w["word"] for w in ref_words]
    user_texts = [w["word"] for w in user_words]
    exp_to_ref  = _align_words(expected, ref_texts)
    exp_to_user = _align_words(expected, user_texts)

    matched_user_idx = {i for i in exp_to_user if i is not None}
    extra_user_idx   = set(range(len(user_words))) - matched_user_idx

    results: List[Dict] = []
    prev_ref_end  = 0.0
    prev_user_end = 0.0

    for i, exp_word in enumerate(expected):
        ref_w  = ref_words[exp_to_ref[i]]   if exp_to_ref[i]  is not None else None
        user_w = user_words[exp_to_user[i]] if exp_to_user[i] is not None else None

        ref_start = ref_w["start"] if ref_w else None
        ref_end   = ref_w["end"]   if ref_w else None
        ref_dur   = (ref_end - ref_start) if ref_start is not None else None

        if user_w is not None:
            user_start = _refine_onset(user_audio, user_sr, user_w["start"], user_w["end"])
            user_end   = user_w["end"]
        else:
            user_start = user_end = None

        user_dur  = (user_end - user_start) if user_start is not None else None
        ref_gap   = max(0.0, (ref_start  - prev_ref_end))  if ref_start  is not None else 0.0
        user_gap  = max(0.0, (user_start - prev_user_end)) if user_start is not None else 0.0
        extra_gap = user_gap - ref_gap

        ratio = (user_dur / ref_dur) if (
            user_dur is not None and ref_dur is not None and ref_dur > MIN_REF_DUR
        ) else None

        feats = (
            _extract_word_features(user_audio, user_sr, user_start, user_end)
            if user_start is not None
            else {"rms_mean": 0.0, "rms_onset": 0.0,
                  "f0_mean": 0.0, "f0_std": 999.0, "zcr_mean": 0.0}
        )

        # ── Classify ─────────────────────────────────────────────────────
        if user_w is None:
            label = "block"
        elif extra_gap > ACOUSTIC_BLOCK_GAP or feats["rms_onset"] < ACOUSTIC_BLOCK_ENERGY:
            label = "block"
        elif (ratio is not None and ratio > ACOUSTIC_PROLONG_RATIO
              and (feats["f0_std"] < ACOUSTIC_F0_STD_THRESH
                   or ratio > ACOUSTIC_PROLONG_RATIO * 1.35)):
            label = "prolongation"
        elif exp_to_user[i] is not None and any(
            j in extra_user_idx
            for j in range(max(0, exp_to_user[i] - 2), exp_to_user[i])
        ):
            label = "repetition"
        else:
            label = "fluent"

        results.append({
            "word":       exp_word,
            "ref_start":  ref_start,   "ref_end":   ref_end,
            "user_start": user_start,  "user_end":  user_end,
            "ref_dur":    ref_dur,     "user_dur":  user_dur,
            "ratio":      ratio,       "extra_gap": extra_gap,
            "rms_onset":  feats["rms_onset"],
            "f0_mean":    feats["f0_mean"],
            "f0_std":     feats["f0_std"],
            "label":      label,
        })

        if ref_end   is not None: prev_ref_end  = ref_end
        if user_end  is not None: prev_user_end = user_end

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────
def _plot_reference(audio: np.ndarray, sr: int, words: List[Dict]) -> plt.Figure:
    duration = len(audio) / sr
    t = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor="#1e1e2e")
    ax.set_facecolor("#1a1b2e")

    for w in words:
        mid = (w["start"] + w["end"]) / 2
        ax.axvspan(w["start"], w["end"], alpha=0.22, color="#cba6f7", linewidth=0)
        ax.axvline(w["start"], color="#585b70", linewidth=0.8, alpha=0.6)
        ax.text(
            mid, 0.90, w["word"],
            ha="center", va="top",
            color="#cdd6f4", fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    ax.plot(t, audio, color="#89dceb", linewidth=0.7, alpha=0.9)
    ax.set_xlim(0, max(duration, 0.01))
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Time (s)", color="#a9b1d6", fontsize=9)
    ax.set_ylabel("Amplitude",  color="#a9b1d6", fontsize=9)
    ax.tick_params(colors="#a9b1d6", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#414868")
    ax.set_title(
        "Reference Audio — Word Timestamps from TTS + Whisper",
        color="#c0caf5", fontsize=10, pad=6,
    )
    fig.tight_layout(pad=0.8)
    return fig


def _plot_user(audio: np.ndarray, sr: int, word_results: List[Dict]) -> plt.Figure:
    duration = len(audio) / sr
    t = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor="#1e1e2e")
    ax.set_facecolor("#1a1b2e")

    for r in word_results:
        if r["user_start"] is not None:
            color = CLASS_COLORS[r["label"]]
            ax.axvspan(
                r["user_start"], min(r["user_end"], duration),
                alpha=0.30, color=color, linewidth=0,
            )
            mid = (r["user_start"] + r["user_end"]) / 2
            ax.text(
                mid, 0.90, r["word"],
                ha="center", va="top",
                color="#cdd6f4", fontsize=8,
                transform=ax.get_xaxis_transform(),
            )
            ax.axvline(r["user_start"], color="#414868", linewidth=0.8, alpha=0.5)

    ax.plot(t, audio, color="#c0caf5", linewidth=0.7, alpha=0.92)

    patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[k], label=CLASS_LABELS[k], alpha=0.75)
        for k in CLASS_COLORS
    ]
    ax.legend(
        handles=patches, loc="upper right", fontsize=7.5,
        facecolor="#24283b", edgecolor="#414868",
        labelcolor="#c0caf5", framealpha=0.92,
    )
    ax.set_xlim(0, max(duration, 0.01))
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Time (s)", color="#a9b1d6", fontsize=9)
    ax.set_ylabel("Amplitude",  color="#a9b1d6", fontsize=9)
    ax.tick_params(colors="#a9b1d6", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#414868")
    ax.set_title(
        "Your Reading — Per-Word Stutter Analysis",
        color="#c0caf5", fontsize=10, pad=6,
    )
    fig.tight_layout(pad=0.8)
    return fig


def _dual_plot(
    ref_audio: np.ndarray, ref_sr: int, ref_words: List[Dict],
    user_audio: np.ndarray, user_sr: int, word_results: List[Dict],
) -> plt.Figure:
    """
    Side-by-side dual plot: reference timeline on top, user waveform below.
    Both share the same x-axis type but independent x ranges.
    """
    fig, (ax_ref, ax_usr) = plt.subplots(
        2, 1, figsize=(11, 5.5), facecolor="#1e1e2e"
    )
    fig.subplots_adjust(hspace=0.45)

    # ── Top: reference ──────────────────────────────────────────────────────
    dur_r = len(ref_audio) / ref_sr
    t_r   = np.linspace(0, dur_r, len(ref_audio))
    ax_ref.set_facecolor("#1a1b2e")
    for w in ref_words:
        mid = (w["start"] + w["end"]) / 2
        ax_ref.axvspan(w["start"], w["end"], alpha=0.22, color="#cba6f7", linewidth=0)
        ax_ref.axvline(w["start"], color="#585b70", linewidth=0.8, alpha=0.6)
        ax_ref.text(
            mid, 0.90, w["word"],
            ha="center", va="top", color="#cdd6f4", fontsize=7.5,
            transform=ax_ref.get_xaxis_transform(),
        )
    ax_ref.plot(t_r, ref_audio, color="#89dceb", linewidth=0.7, alpha=0.9)
    ax_ref.set_xlim(0, max(dur_r, 0.01))
    ax_ref.set_ylim(-1.08, 1.08)
    ax_ref.set_xlabel("Time (s)", color="#a9b1d6", fontsize=8)
    ax_ref.set_ylabel("Amplitude",  color="#a9b1d6", fontsize=8)
    ax_ref.tick_params(colors="#a9b1d6", labelsize=7)
    for sp in ax_ref.spines.values():
        sp.set_edgecolor("#414868")
    ax_ref.set_title("Reference (TTS)", color="#89dceb", fontsize=9, pad=5)

    # ── Bottom: user ─────────────────────────────────────────────────────────
    dur_u = len(user_audio) / user_sr
    t_u   = np.linspace(0, dur_u, len(user_audio))
    ax_usr.set_facecolor("#1a1b2e")
    for r in word_results:
        if r["user_start"] is not None:
            color = CLASS_COLORS[r["label"]]
            ax_usr.axvspan(
                r["user_start"], min(r["user_end"], dur_u),
                alpha=0.30, color=color, linewidth=0,
            )
            mid = (r["user_start"] + r["user_end"]) / 2
            ax_usr.text(
                mid, 0.90, r["word"],
                ha="center", va="top", color="#cdd6f4", fontsize=7.5,
                transform=ax_usr.get_xaxis_transform(),
            )
            ax_usr.axvline(r["user_start"], color="#414868", linewidth=0.8, alpha=0.5)
    ax_usr.plot(t_u, user_audio, color="#c0caf5", linewidth=0.7, alpha=0.92)
    patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[k], label=CLASS_LABELS[k], alpha=0.75)
        for k in CLASS_COLORS
    ]
    ax_usr.legend(
        handles=patches, loc="upper right", fontsize=7,
        facecolor="#24283b", edgecolor="#414868",
        labelcolor="#c0caf5", framealpha=0.92,
    )
    ax_usr.set_xlim(0, max(dur_u, 0.01))
    ax_usr.set_ylim(-1.08, 1.08)
    ax_usr.set_xlabel("Time (s)", color="#a9b1d6", fontsize=8)
    ax_usr.set_ylabel("Amplitude",  color="#a9b1d6", fontsize=8)
    ax_usr.tick_params(colors="#a9b1d6", labelsize=7)
    for sp in ax_usr.spines.values():
        sp.set_edgecolor("#414868")
    ax_usr.set_title("Your Reading — colour = stutter type", color="#c0caf5", fontsize=9, pad=5)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Results HTML table
# ─────────────────────────────────────────────────────────────────────────────
def _results_table(word_results: List[Dict]) -> str:
    rows = []
    for r in word_results:
        label = r["label"]
        color = CLASS_COLORS[label]
        badge = (
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:4px;font-size:12px">{CLASS_LABELS[label]}</span>'
        )
        ref_d   = f"{r['ref_dur']:.2f} s"   if r["ref_dur"]   is not None else "—"
        usr_d   = f"{r['user_dur']:.2f} s"  if r["user_dur"]  is not None else "—"
        ratio   = f"{r['ratio']:.2f}×"      if r["ratio"]     is not None else "—"
        gap     = (
            f"+{r['extra_gap']:.2f} s"       if (r["extra_gap"] is not None and r["extra_gap"] > 0.01)
            else "—"
        )
        rows.append(
            f"<tr>"
            f"<td style='padding:5px 12px;font-size:15px;direction:rtl'>{r['word']}</td>"
            f"<td style='padding:5px 12px'>{ref_d}</td>"
            f"<td style='padding:5px 12px'>{usr_d}</td>"
            f"<td style='padding:5px 12px'>{ratio}</td>"
            f"<td style='padding:5px 12px'>{gap}</td>"
            f"<td style='padding:5px 12px'>{badge}</td>"
            f"</tr>"
        )
    html = (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-family:sans-serif'>"
        "<thead><tr style='background:#313244;color:#cdd6f4;text-align:center'>"
        "<th style='padding:7px 12px'>Word / الكلمة</th>"
        "<th style='padding:7px 12px'>Ref Dur</th>"
        "<th style='padding:7px 12px'>User Dur</th>"
        "<th style='padding:7px 12px'>Ratio</th>"
        "<th style='padding:7px 12px'>Extra Silence</th>"
        "<th style='padding:7px 12px'>Label</th>"
        "</tr></thead>"
        "<tbody style='background:#1e1e2e;color:#cdd6f4;text-align:center'>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    # Summary bar
    counts = {k: sum(1 for r in word_results if r["label"] == k) for k in CLASS_COLORS}
    total  = len(word_results)
    summary_parts = []
    for k, cnt in counts.items():
        if cnt:
            pct = 100 * cnt / total if total else 0
            summary_parts.append(
                f'<span style="background:{CLASS_COLORS[k]};color:#fff;'
                f'padding:3px 10px;border-radius:4px;margin:0 4px;font-size:13px">'
                f'{CLASS_LABELS[k]}: {cnt} ({pct:.0f}%)</span>'
            )
    summary = (
        "<div style='margin:10px 0;text-align:center'>"
        + "".join(summary_parts)
        + "</div>"
    )
    return summary + html


def _acoustic_table(word_results: List[Dict]) -> str:
    """HTML results table with additional acoustic feature columns (F0, RMS onset)."""
    rows = []
    for r in word_results:
        label = r["label"]
        color = CLASS_COLORS[label]
        badge = (
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:4px;font-size:12px">{CLASS_LABELS[label]}</span>'
        )
        ref_d  = f"{r['ref_dur']:.2f} s"   if r["ref_dur"]   is not None else "—"
        usr_d  = f"{r['user_dur']:.2f} s"  if r["user_dur"]  is not None else "—"
        ratio  = f"{r['ratio']:.2f}×"      if r["ratio"]     is not None else "—"
        gap    = f"+{r['extra_gap']:.2f} s" if (r.get("extra_gap") and r["extra_gap"] > 0.01) else "—"
        rms_on = f"{r['rms_onset']:.3f}"   if r.get("rms_onset") is not None else "—"
        f0_m   = f"{r['f0_mean']:.0f} Hz"  if r.get("f0_mean") else "—"
        f0_s   = (f"{r['f0_std']:.1f} Hz"
                  if (r.get("f0_std") is not None and r["f0_std"] < 900) else "—")
        rows.append(
            f"<tr>"
            f"<td style='padding:5px 12px;font-size:15px;direction:rtl'>{r['word']}</td>"
            f"<td style='padding:5px 12px'>{ref_d}</td>"
            f"<td style='padding:5px 12px'>{usr_d}</td>"
            f"<td style='padding:5px 12px'>{ratio}</td>"
            f"<td style='padding:5px 12px'>{gap}</td>"
            f"<td style='padding:5px 12px'>{rms_on}</td>"
            f"<td style='padding:5px 12px'>{f0_m}</td>"
            f"<td style='padding:5px 12px'>{f0_s}</td>"
            f"<td style='padding:5px 12px'>{badge}</td>"
            f"</tr>"
        )
    html = (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-family:sans-serif'>"
        "<thead><tr style='background:#313244;color:#cdd6f4;text-align:center'>"
        "<th style='padding:7px 12px'>Word / الكلمة</th>"
        "<th style='padding:7px 12px'>Ref Dur</th>"
        "<th style='padding:7px 12px'>User Dur</th>"
        "<th style='padding:7px 12px'>Ratio</th>"
        "<th style='padding:7px 12px'>Extra Silence</th>"
        "<th style='padding:7px 12px'>RMS Onset</th>"
        "<th style='padding:7px 12px'>F0 Mean</th>"
        "<th style='padding:7px 12px'>F0 Std</th>"
        "<th style='padding:7px 12px'>Label</th>"
        "</tr></thead>"
        "<tbody style='background:#1e1e2e;color:#cdd6f4;text-align:center'>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    counts = {k: sum(1 for r in word_results if r["label"] == k) for k in CLASS_COLORS}
    total  = len(word_results)
    summary_parts = []
    for k, cnt in counts.items():
        if cnt:
            pct = 100 * cnt / total if total else 0
            summary_parts.append(
                f'<span style="background:{CLASS_COLORS[k]};color:#fff;'
                f'padding:3px 10px;border-radius:4px;margin:0 4px;font-size:13px">'
                f'{CLASS_LABELS[k]}: {cnt} ({pct:.0f}%)</span>'
            )
    summary = (
        "<div style='margin:10px 0;text-align:center'>"
        + "".join(summary_parts)
        + "</div>"
    )
    return summary + html


def _plot_acoustic_features(word_results: List[Dict]) -> plt.Figure:
    """Bar charts of per-word F0 mean and RMS onset energy, colour-coded by label."""
    words  = [r["word"] for r in word_results]
    f0s    = [r.get("f0_mean", 0.0) or 0.0 for r in word_results]
    rms_on = [r.get("rms_onset", 0.0) or 0.0 for r in word_results]
    colors = [CLASS_COLORS[r["label"]] for r in word_results]
    n = len(words)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 1.2), 5.0), facecolor="#1e1e2e")
    fig.subplots_adjust(hspace=0.60)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1b2e")
        ax.tick_params(colors="#a9b1d6", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#414868")

    ax1.bar(x, f0s, color=colors, alpha=0.85, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(words, rotation=30, ha="right", fontsize=9, color="#cdd6f4")
    ax1.set_ylabel("F0 mean (Hz)", color="#a9b1d6", fontsize=9)
    ax1.set_title("Fundamental Frequency (F0) per Word — flat = prolongation",
                  color="#c0caf5", fontsize=10, pad=5)
    ax1.axhline(0, color="#585b70", linewidth=0.8)

    ax2.bar(x, rms_on, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(
        ACOUSTIC_BLOCK_ENERGY, color="#F44336", linewidth=1.2, linestyle="--", alpha=0.8,
        label=f"Block threshold  ({ACOUSTIC_BLOCK_ENERGY})",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(words, rotation=30, ha="right", fontsize=9, color="#cdd6f4")
    ax2.set_ylabel("RMS onset energy", color="#a9b1d6", fontsize=9)
    ax2.set_title("Onset Energy per Word — below red line = block",
                  color="#c0caf5", fontsize=10, pad=5)
    ax2.legend(fontsize=8, facecolor="#24283b", edgecolor="#414868", labelcolor="#c0caf5")

    patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[k], label=CLASS_LABELS[k], alpha=0.8)
        for k in CLASS_COLORS
    ]
    fig.legend(
        handles=patches, loc="upper right", fontsize=7.5,
        facecolor="#24283b", edgecolor="#414868", labelcolor="#c0caf5", framealpha=0.92,
        bbox_to_anchor=(0.99, 0.98),
    )
    fig.tight_layout(pad=0.8)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio handler functions
# ─────────────────────────────────────────────────────────────────────────────
def generate_reference(text: str, dialect: str):
    """
    TTS → save wav → Whisper word timestamps → return audio + plot + state.
    """
    text = text.strip()
    if not text:
        return None, None, {}, "⚠️ Please enter Arabic text first."

    try:
        # Resolve dialect token for the Unified model
        d_code = dialect.upper()[:3]  # "MSA", "EGY", "LEV", …
        dialect_id = _dialect_id_map.get(d_code, _dialect_id_map["MSA"])

        # Ensure generated text ends with ". " (required by infer_batch_process)
        gen_text_proc = text.strip().rstrip(".") + ". "

        # Call infer_batch_process directly with a pre-loaded tensor — this
        # avoids torchaudio.load() and the optional torchcodec dependency.
        final_wave, sr, _ = next(_tts_infer_batch(
            (_ref_tensor, _ref_sr),
            _MSA_REF_TEXT_PROC,
            [gen_text_proc],
            _tts_model,
            _vocoder,
            dialect_id=dialect_id,
            device=DEVICE,
        ))

        # Convert to float32 numpy mono
        if isinstance(final_wave, torch.Tensor):
            audio_np = final_wave.cpu().numpy().astype(np.float32)
        else:
            audio_np = np.array(final_wave, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        audio_np = _normalize(audio_np)

        # Ensure 16 kHz for Whisper
        if sr != WHISPER_SR:
            audio_np = librosa.resample(audio_np, orig_sr=int(sr), target_sr=WHISPER_SR)
            sr = WHISPER_SR

        ref_path = _save_wav(audio_np, sr, "guided_ref.wav")
        ref_words, _ = _whisper_words(ref_path)

        fig = _plot_reference(audio_np, sr, ref_words)

        state = {
            "audio":    audio_np,
            "sr":       sr,
            "words":    ref_words,
            "expected": text.split(),   # simple whitespace tokenisation
            "text":     text,
        }
        status = (
            f"✅ Reference ready — {len(ref_words)} words detected by Whisper. "
            "Now record your reading and click Analyse."
        )
        return (sr, audio_np), fig, state, status

    except Exception as exc:
        return None, None, {}, f"🚨 TTS/Whisper error: {exc}"


def analyse_reading(user_audio: Optional[Tuple], ref_state: Dict):
    """
    Whisper on user recording → align to reference → classify → plot + table.
    """
    if user_audio is None:
        return None, "<p style='color:#f38ba8'>⚠️ Record your reading first.</p>"
    if not ref_state:
        return None, "<p style='color:#f38ba8'>⚠️ Generate reference audio first.</p>"

    try:
        sr_u, user_np = user_audio
        user_np = np.array(user_np, dtype=np.float32)
        if user_np.ndim > 1:
            user_np = user_np.mean(axis=1)

        if sr_u != WHISPER_SR:
            user_np = librosa.resample(user_np, orig_sr=sr_u, target_sr=WHISPER_SR)
            sr_u = WHISPER_SR
        user_np = _normalize(user_np)

        user_path = _save_wav(user_np, sr_u, "guided_user.wav")
        user_words, user_text = _whisper_words(user_path)

        ref_words = ref_state["words"]
        expected  = ref_state["expected"]

        word_results = classify_words(expected, ref_words, user_words)

        fig   = _dual_plot(
            ref_state["audio"], ref_state["sr"], ref_words,
            user_np, sr_u, word_results,
        )
        table = _results_table(word_results)
        return fig, table

    except Exception as exc:
        return None, f"<p style='color:#f38ba8'>🚨 Analysis error: {exc}</p>"


def analyse_reading_acoustic(user_audio: Optional[Tuple], ref_state: Dict):
    """
    Acoustic-analysis handler (Tab 2).
    Uses energy-refined boundaries + librosa.pyin F0 + onset RMS to classify words.
    Returns: dual waveform plot, acoustic features plot, HTML table.
    """
    if user_audio is None:
        return None, None, "<p style='color:#f38ba8'>⚠️ Record your reading first.</p>"
    if not ref_state:
        return None, None, "<p style='color:#f38ba8'>⚠️ Generate reference audio first.</p>"

    try:
        sr_u, user_np = user_audio
        user_np = np.array(user_np, dtype=np.float32)
        if user_np.ndim > 1:
            user_np = user_np.mean(axis=1)
        if sr_u != WHISPER_SR:
            user_np = librosa.resample(user_np, orig_sr=sr_u, target_sr=WHISPER_SR)
            sr_u = WHISPER_SR
        user_np = _normalize(user_np)

        user_path = _save_wav(user_np, sr_u, "guided_user_acoustic.wav")
        user_words, _ = _whisper_words(user_path)

        ref_words = ref_state["words"]
        expected  = ref_state["expected"]

        word_results = classify_words_acoustic(
            expected, ref_words, user_words,
            ref_state["audio"], ref_state["sr"],
            user_np, sr_u,
        )

        fig_dual  = _dual_plot(
            ref_state["audio"], ref_state["sr"], ref_words,
            user_np, sr_u, word_results,
        )
        fig_feats = _plot_acoustic_features(word_results)
        table     = _acoustic_table(word_results)
        return fig_dual, fig_feats, table

    except Exception as exc:
        return None, None, f"<p style='color:#f38ba8'>🚨 Analysis error: {exc}</p>"


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Guided Reading — Stutter Detector",
    theme=gr.themes.Soft(),
    css=".rtl-text { direction: rtl; text-align: right; font-size: 20px; }",
) as demo:

    gr.Markdown(
        """
        # 📖 Guided Reading — Stutter Detector  ·  كاشف التلعثم بالقراءة الموجّهة

        | Step | What to do |
        |------|-----------|
        | **1️⃣ Sentence** | Choose a preset or type your own Arabic sentence |
        | **2️⃣ Reference** | Click **Generate Reference** — TTS synthesises the sentence; Whisper extracts per-word timing |
        | **3️⃣ Listen** | Play the reference audio so you know what to say |
        | **4️⃣ Record** | Press the microphone button and read the sentence aloud |
        | **5️⃣ Analyse** | Click **Analyse My Reading** — each word is colour-coded by stutter type |

        ---
        **Detection logic:**  
        🔵 *Prolongation* — your word lasted > 1.8× the reference duration  
        🔴 *Block* — long silence (> 0.4 s extra) before the word  
        🟠 *Repetition* — Whisper found extra repeated words vs. the expected text  
        🟢 *Fluent* — timing matches reference  
        """
    )

    ref_state = gr.State(value={})

    # ── Step 1 & 2: Sentence + Generate ─────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            preset_dd = gr.Dropdown(
                choices=PRESET_SENTENCES,
                value=PRESET_SENTENCES[0],
                label="Preset Sentences",
            )
            text_box = gr.Textbox(
                value=PRESET_SENTENCES[0],
                label="Arabic Sentence  (edit freely)",
                lines=2,
                rtl=True,
            )
        with gr.Column(scale=1):
            dialect_dd = gr.Dropdown(
                choices=["MSA", "EGY", "LEV", "SAU", "UAE"],
                value="MSA",
                label="TTS Dialect",
            )
            gen_btn = gr.Button("🔊 Generate Reference", variant="primary", size="lg")

    gen_status = gr.Textbox(label="Status", interactive=False, lines=1)

    # Selecting a preset fills the text box
    preset_dd.change(fn=lambda x: x, inputs=preset_dd, outputs=text_box)

    # ── Reference output ─────────────────────────────────────────────────────
    with gr.Row():
        ref_audio_out = gr.Audio(
            label="Reference Audio  (TTS synthesised)",
            type="numpy",
            interactive=False,
        )
    ref_plot = gr.Plot(label="Reference Word Timeline")

    # ── Tabbed analysis ───────────────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Tabs():

        # ── Tab 1: Threshold-Based ────────────────────────────────────────────
        with gr.Tab("🔢 Threshold Analysis"):
            gr.Markdown("### 🎙️ Record yourself reading the sentence above")
            with gr.Row():
                user_mic_t = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Your Recording",
                )
                analyse_btn_t = gr.Button("🔍 Analyse My Reading", variant="primary", size="lg")
            result_plot_t  = gr.Plot(label="Comparison — Reference (top) · Your Reading (bottom)")
            result_table_t = gr.HTML(label="Per-Word Results")
            gr.Markdown(
                """
                **Detection rules (fixed thresholds):**  
                Duration ratio > **1.8×** → Prolongation · Extra silence > **0.40 s** → Block · Whisper insertions → Repetition
                """
            )

        # ── Tab 2: Acoustic Analysis ──────────────────────────────────────────
        with gr.Tab("🔬 Acoustic Analysis"):
            gr.Markdown(
                """
                ### 🎙️ Record yourself reading the sentence above
                **Improvements over Threshold tab:**
                - Word onset boundaries are **refined with RMS energy** — finds exact start within Whisper's estimate
                - **F0 (fundamental frequency)** via `librosa.pyin` — flat pitch + long duration → Prolongation
                - **Onset energy** — near-zero RMS at word start → Block
                """
            )
            with gr.Row():
                user_mic_a = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Your Recording",
                )
                analyse_btn_a = gr.Button("🔬 Analyse (Acoustic)", variant="primary", size="lg")
            result_plot_a  = gr.Plot(label="Comparison — Reference (top) · Your Reading (bottom)")
            feat_plot_a    = gr.Plot(label="Acoustic Features — F0 & Onset Energy per Word")
            result_table_a = gr.HTML(label="Per-Word Results (with F0 & RMS columns)")
            gr.Markdown(
                """
                **Detection rules:**  
                Extra silence > **0.35 s** OR onset RMS < **0.02** → Block  
                Duration > **1.6×** AND F0 std < **15 Hz** (flat pitch) → Prolongation  
                Whisper insertions → Repetition
                """
            )

    # ── Wire up ───────────────────────────────────────────────────────────────
    gen_btn.click(
        fn=generate_reference,
        inputs=[text_box, dialect_dd],
        outputs=[ref_audio_out, ref_plot, ref_state, gen_status],
    )
    analyse_btn_t.click(
        fn=analyse_reading,
        inputs=[user_mic_t, ref_state],
        outputs=[result_plot_t, result_table_t],
    )
    analyse_btn_a.click(
        fn=analyse_reading_acoustic,
        inputs=[user_mic_a, ref_state],
        outputs=[result_plot_a, feat_plot_a, result_table_a],
    )

    gr.Markdown(
        """
        ---
        **How the math works**

        | Metric | Threshold tab | Acoustic tab |
        |--------|---------------|--------------|
        | Duration ratio | user_dur ÷ ref_dur > **1.8** → Prolongation | > **1.6** (+ F0 flatness check) |
        | Pre-word silence | extra gap > **0.40 s** → Block | extra gap > **0.35 s** OR onset RMS < **0.02** |
        | Prolongation confidence | duration only | duration **+** F0 std < 15 Hz (librosa.pyin) |
        | Repetition | SequenceMatcher insertions | SequenceMatcher insertions |
        | Boundary accuracy | Raw Whisper timestamps (~100 ms) | Energy-refined onset (~10 ms) |
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
    )
