"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            Stutter Detection Model — Trainer & Evaluator                    ║
║  Strategy: Transfer Learning (Wav2Vec2) + Synthetic Augmentation            ║
║  Classes : Fluent | Repetition | Prolongation | Block                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
# 1. Build a synthetic dataset from clean Arabic speech files:
    python stutter_model_trainer.py --mode build_dataset \
        --clean_dir data/clean \
        --output_dir data/stuttered \
        --samples_per_class 500

# 2. Train the classifier (fine-tune Wav2Vec2 head):
    python stutter_model_trainer.py --mode train \
        --data_dir data/stuttered \
        --output_model models/stutter_classifier \
        --epochs 30 --batch_size 16

# 3. Evaluate on a held-out test set:
    python stutter_model_trainer.py --mode evaluate \
        --data_dir data/stuttered \
        --output_model models/stutter_classifier
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard Library
# ─────────────────────────────────────────────────────────────────────────────
import os
import argparse
import random
import copy
import json
import time
import warnings
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Wav2Vec2 native sample rate
MAX_AUDIO_LEN = 5.0           # seconds — clips longer than this are truncated
LABEL_MAP: Dict[str, int] = {
    "fluent":      0,
    "repetition":  1,
    "prolongation": 2,
    "block":       3,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

# Wav2Vec2 checkpoint — downloads once to HuggingFace cache
# Alternatives: "facebook/wav2vec2-large-xlsr-53"  (heavier, higher ceiling)
#               "facebook/wav2vec2-base"            (lighter, fast iteration)
WAV2VEC2_CHECKPOINT = "facebook/wav2vec2-base"

# ─────────────────────────────────────────────────────────────────────────────
#  1. EVERYAYAH DOWNLOADER  (Clean Arabic Speech)
# ─────────────────────────────────────────────────────────────────────────────

# URL pattern: https://everyayah.com/data/{reciter}/{surah:03d}{ayah:03d}.mp3
EVERYAYAH_BASE = "https://everyayah.com/data"

# Verified reciter folder names on everyayah.com.
# Each entry is (folder_name, display_name).
EVERYAYAH_RECITERS: List[Tuple[str, str]] = [
    ("Alafasy_128kbps",                    "Mishary Alafasy"),
    ("Abdul_Basit_Murattal_192kbps",       "Abdul Basit (Murattal)"),
    ("Husary_128kbps",                     "Mahmoud Khalil Al-Husary"),
    ("Minshawy_Murattal_128kbps",          "Al-Minshawy (Murattal)"),
    ("Mohammad_al_Tablaway_128kbps",       "Mohammad al-Tablaway"),
    ("AbdulSamad_64kbps_QuranExplorer.Com","Abdul Samad"),
]

# Number of ayahs per surah (114 surahs, 1-indexed via QURAN_AYAH_COUNTS[surah-1])
QURAN_AYAH_COUNTS: List[int] = [
    7,  286, 200, 176, 120, 165, 206,  75, 129, 109,
  123,  111,  43,  52,  99, 128, 111, 110,  98, 135,
  112,   78, 118,  64,  77, 227,  93,  88,  69,  60,
   34,   30,  73,  54,  45,  83, 182,  88,  75,  85,
   54,   53,  89,  59,  37,  35,  38,  29,  18,  45,
   60,   49,  62,  55,  78,  96,  29,  22,  24,  13,
   14,   11,  11,  18,  12,  12,  30,  52,  52,  44,
   28,   28,  20,  56,  40,  31,  50,  40,  46,  42,
   29,   19,  36,  25,  22,  17,  19,  26,  30,  20,
   15,   21,  11,   8,   8,  19,   5,   8,   8,  11,
   11,    8,   3,   9,   5,   4,   7,   3,   6,   3,
    5,    4,   5,   6,
]


def _all_ayah_refs() -> List[Tuple[int, int]]:
    """Return every (surah, ayah) pair in Quran order."""
    refs: List[Tuple[int, int]] = []
    for surah, count in enumerate(QURAN_AYAH_COUNTS, start=1):
        for ayah in range(1, count + 1):
            refs.append((surah, ayah))
    return refs   # 6 236 total


def download_clean_from_everyayah(
    output_dir: str,
    num_files: int = 300,
    reciters: Optional[List[str]] = None,
    surahs: Optional[List[int]] = None,
    delay: float = 0.25,
    retries: int = 3,
) -> None:
    """
    Download clean Arabic speech MP3s from https://everyayah.com and save
    them to *output_dir* for use as fluent training data.

    Files are saved as:  output_dir/{reciter_folder}/{surah:03d}{ayah:03d}.mp3

    Args:
        output_dir : Destination directory (created if missing).
        num_files  : Total number of ayah files to download across all
                     selected reciters (spread evenly).
        reciters   : List of reciter folder names from EVERYAYAH_RECITERS.
                     Defaults to all 6 reciters when None.
        surahs     : Restrict downloads to these surah numbers (1-114).
                     Defaults to all surahs when None.
        delay      : Seconds to sleep between HTTP requests (be a good citizen).
        retries    : Number of retry attempts per file on network errors.
    """
    if reciters is None:
        reciters = [r[0] for r in EVERYAYAH_RECITERS]

    # Build the pool of (surah, ayah) pairs to sample from
    all_refs = _all_ayah_refs()
    if surahs:
        all_refs = [(s, a) for s, a in all_refs if s in set(surahs)]
    if not all_refs:
        raise ValueError("No ayah references match the given surah filter.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files_per_reciter = max(1, num_files // len(reciters))
    downloaded = 0
    skipped = 0

    log.info(
        f"Downloading {num_files} ayah MP3s from everyayah.com "
        f"({len(reciters)} reciter(s), ~{files_per_reciter} each) …"
    )

    for reciter_folder in reciters:
        reciter_dir = out_root / reciter_folder
        reciter_dir.mkdir(parents=True, exist_ok=True)

        # Randomly sample unique ayahs for this reciter
        selected = random.sample(all_refs, min(files_per_reciter, len(all_refs)))

        for surah, ayah in selected:
            filename = f"{surah:03d}{ayah:03d}.mp3"
            dest = reciter_dir / filename

            if dest.exists():
                skipped += 1
                continue

            url = f"{EVERYAYAH_BASE}/{reciter_folder}/{filename}"
            success = False

            for attempt in range(1, retries + 1):
                try:
                    # Set a browser-like User-Agent to avoid being blocked
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = resp.read()

                    # Sanity-check: everyayah returns a small HTML error page
                    # (< 2 KB) when a file doesn't exist
                    if len(data) < 2_000:
                        log.warning(
                            f"  Skipping {filename} — response too small "
                            f"({len(data)} B), likely missing on server."
                        )
                        break

                    dest.write_bytes(data)
                    downloaded += 1
                    success = True
                    break

                except urllib.error.HTTPError as e:
                    log.warning(f"  HTTP {e.code} for {url} (attempt {attempt})")
                    if e.code == 404:
                        break   # file genuinely absent — don't retry
                    time.sleep(delay * attempt)

                except (urllib.error.URLError, OSError) as e:
                    log.warning(f"  Network error for {url}: {e} (attempt {attempt})")
                    time.sleep(delay * attempt)

            if success:
                time.sleep(delay)   # polite rate-limit between requests

        log.info(
            f"  [{reciter_folder}] done — "
            f"downloaded so far: {downloaded}, skipped (exist): {skipped}"
        )

    log.info(
        f"\n✅ EveryAyah download complete: "
        f"{downloaded} new files, {skipped} already existed → '{output_dir}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  2. AUDIO AUGMENTATION  (Synthetic Stutter Generation)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *sr* Hz."""
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def trim_audio(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """Remove leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if len(trimmed) > SAMPLE_RATE * 0.1 else audio


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalize so the loudest sample reaches *target_peak*."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * target_peak
    return audio


# ── Augmentation type 1: Sound / Syllable Repetition ──────────────────────
def generate_repetition(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    num_repeats: int = 3,
    chunk_duration: float = 0.25,
) -> np.ndarray:
    """
    Simulate stuttering by repeating the first *chunk_duration* seconds of
    speech *num_repeats* times before the rest of the utterance.

    Example (num_repeats=3, chunk=0.25s):
        [0-0.25s] → [0-0.25s] × 3 + [0.25s-end]
    """
    chunk_samples = int(sr * chunk_duration)
    chunk_samples = min(chunk_samples, len(audio))

    stutter_chunk = audio[:chunk_samples]
    repeated = np.tile(stutter_chunk, num_repeats)

    return np.concatenate([repeated, audio[chunk_samples:]])


# ── Augmentation type 2: Prolongation ─────────────────────────────────────
def generate_prolongation(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    stretch_factor: float = 0.35,
    onset_offset: float = 0.15,
    prolong_duration: float = 0.45,
) -> np.ndarray:
    """
    Slow down a segment of the audio (without pitch shift) to imitate a
    prolonged phoneme.

    *stretch_factor* < 1 → slower playback.
    The segment starts at *onset_offset* seconds from the beginning.
    """
    start = int(sr * onset_offset)
    end = start + int(sr * prolong_duration)
    end = min(end, len(audio))

    if end <= start:
        return audio

    segment = audio[start:end]
    # time_stretch expects rate>0; <1 → slower
    try:
        prolonged = librosa.effects.time_stretch(segment, rate=stretch_factor)
    except Exception:
        prolonged = segment  # fallback: keep original if stretch fails

    return np.concatenate([audio[:start], prolonged, audio[end:]])


# ── Augmentation type 3: Block / Silent Pause ─────────────────────────────
def generate_block(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    onset_offset: float = 0.35,
    block_duration: float = 0.55,
) -> np.ndarray:
    """
    Insert a silent pause at *onset_offset* seconds to simulate a laryngeal
    block (hard onset).
    """
    insert_at = int(sr * onset_offset)
    insert_at = min(insert_at, len(audio))

    silence = np.zeros(int(sr * block_duration))

    return np.concatenate([audio[:insert_at], silence, audio[insert_at:]])


# ── Noise Robustness: White / Ambient Noise ────────────────────────────────
def add_background_noise(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    noise_level: float = 0.07,
    noise_type: str = "white",
    ambient_path: Optional[str] = None,
) -> np.ndarray:
    """
    Add background noise to improve model robustness in real environments.

    noise_type:
        "white"   — Gaussian white noise (default, always available)
        "ambient" — mix a recorded ambient file (requires *ambient_path*)
    noise_level: SNR scale relative to audio RMS (5-15% recommended)
    """
    if noise_type == "ambient" and ambient_path and os.path.exists(ambient_path):
        noise, _ = librosa.load(ambient_path, sr=sr, mono=True)
        # Loop or trim ambient noise to match audio length
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[: len(audio)]
    else:
        noise = np.random.randn(len(audio)).astype(np.float32)

    # Scale noise so its RMS is *noise_level* × audio RMS
    audio_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    noise = noise * (audio_rms / noise_rms) * noise_level

    return np.clip(audio + noise, -1.0, 1.0)


# ── Apply all augmentations with randomisation ────────────────────────────
_AUGMENT_FN = {
    "repetition":   generate_repetition,
    "prolongation": generate_prolongation,
    "block":        generate_block,
}


def augment_audio(
    audio: np.ndarray,
    sr: int,
    label: str,
    add_noise: bool = True,
    noise_level: float = 0.07,
) -> np.ndarray:
    """
    Apply the stutter augmentation corresponding to *label*, then optionally
    add background noise.
    """
    audio = trim_audio(audio)
    audio = normalize_audio(audio)

    if label != "fluent":
        fn = _AUGMENT_FN[label]
        # Randomise parameters slightly for diversity
        if label == "repetition":
            audio = fn(
                audio, sr,
                num_repeats=random.randint(2, 4),
                chunk_duration=random.uniform(0.15, 0.35),
            )
        elif label == "prolongation":
            audio = fn(
                audio, sr,
                stretch_factor=random.uniform(0.25, 0.50),
                onset_offset=random.uniform(0.10, 0.30),
                prolong_duration=random.uniform(0.30, 0.60),
            )
        elif label == "block":
            audio = fn(
                audio, sr,
                onset_offset=random.uniform(0.20, 0.50),
                block_duration=random.uniform(0.40, 0.80),
            )

    if add_noise:
        noise_level_actual = random.uniform(0.03, noise_level)
        audio = add_background_noise(audio, sr, noise_level=noise_level_actual)

    # Enforce maximum clip length
    max_samples = int(MAX_AUDIO_LEN * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return normalize_audio(audio)


# ─────────────────────────────────────────────────────────────────────────────
#  3. DATASET BUILDER  (generate .wav files on disk)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_from_clean_audio(
    clean_dir: str,
    output_dir: str,
    samples_per_class: int = 500,
    noise_level: float = 0.07,
) -> None:
    """
    Given a directory of clean (fluent) speech recordings, synthesise
    *samples_per_class* samples for each of the 4 classes and save them under:

        output_dir/
            fluent/        *.wav
            repetition/    *.wav
            prolongation/  *.wav
            block/         *.wav

    Args:
        clean_dir         : Path to clean speech .wav / .mp3 files.
        output_dir        : Root output directory.
        samples_per_class : Number of generated samples per class.
        noise_level       : Max noise amplitude added to synthetic clips.
    """
    clean_files = sorted(
        [p for p in Path(clean_dir).rglob("*")
         if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
    )
    if not clean_files:
        raise FileNotFoundError(
            f"No audio files found in '{clean_dir}'. "
            "Place clean (fluent) speech recordings there first."
        )

    log.info(f"Found {len(clean_files)} clean audio files in '{clean_dir}'.")

    for label in LABEL_MAP:
        out_path = Path(output_dir) / label
        out_path.mkdir(parents=True, exist_ok=True)

    idx = {label: 0 for label in LABEL_MAP}
    counts = {label: 0 for label in LABEL_MAP}

    while any(counts[label] < samples_per_class for label in LABEL_MAP):
        src = random.choice(clean_files)
        try:
            audio, sr = load_audio(str(src))
        except Exception as e:
            log.warning(f"Could not load {src}: {e}")
            continue

        if len(audio) < sr * 0.5:   # skip clips shorter than 0.5 s
            continue

        for label in LABEL_MAP:
            if counts[label] >= samples_per_class:
                continue

            augmented = augment_audio(audio.copy(), sr, label,
                                      add_noise=True,
                                      noise_level=noise_level)
            out_file = (
                Path(output_dir) / label
                / f"{label}_{idx[label]:05d}.wav"
            )
            sf.write(str(out_file), augmented, sr)
            idx[label] += 1
            counts[label] += 1

        # Progress log every 100 samples
        if sum(counts.values()) % 400 == 0:
            log.info(f"  Generated so far → {counts}")

    log.info(f"✅ Dataset built under '{output_dir}': {counts}")


# ─────────────────────────────────────────────────────────────────────────────
#  4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class StutterDataset(Dataset):
    """
    Loads pre-generated .wav clips from:
        root_dir/<class_name>/*.wav

    Returns (input_values tensor, label int).
    The Wav2Vec2FeatureExtractor normalises the raw waveform to zero mean /
    unit variance, which is what the pre-trained model expects.
    """

    def __init__(
        self,
        root_dir: str,
        feature_extractor: Wav2Vec2FeatureExtractor,
        max_length: float = MAX_AUDIO_LEN,
        file_list: Optional[List[Tuple[str, int]]] = None,
    ):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.max_samples = int(max_length * SAMPLE_RATE)

        if file_list is not None:
            self.samples: List[Tuple[str, int]] = file_list
        else:
            self.samples = self._scan_directory()

    def _scan_directory(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for label_name, label_idx in LABEL_MAP.items():
            class_dir = Path(self.root_dir) / label_name
            if not class_dir.exists():
                log.warning(f"Class directory not found: {class_dir}")
                continue
            files = sorted(class_dir.glob("*.wav"))
            for f in files:
                samples.append((str(f), label_idx))
        random.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_path, label = self.samples[idx]

        audio, sr = load_audio(file_path, sr=SAMPLE_RATE)

        # Truncate / pad to fixed length
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)  # (T,)
        return input_values, torch.tensor(label, dtype=torch.long)


def split_dataset(
    root_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    # remaining 0.15 → test
) -> Tuple[List, List, List]:
    """Return train / val / test file lists from root_dir."""
    all_samples: List[Tuple[str, int]] = []
    for label_name, label_idx in LABEL_MAP.items():
        class_dir = Path(root_dir) / label_name
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.wav"))
        all_samples.extend([(str(f), label_idx) for f in files])

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = all_samples[:n_train]
    val = all_samples[n_train: n_train + n_val]
    test = all_samples[n_train + n_val:]

    log.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  5. MODEL  (Wav2Vec2 + Classifier Head)
# ─────────────────────────────────────────────────────────────────────────────

class StutterClassifier(nn.Module):
    """
    Architecture
    ────────────
    ┌───────────────────────────────┐
    │  Wav2Vec2 Encoder (frozen ×N  │  ← pre-trained feature extractor
    │  epochs, then unfrozen)       │
    └──────────────┬────────────────┘
                   │  (B, T, hidden_size)
            Mean Pooling over T
                   │  (B, hidden_size)
    ┌──────────────▼────────────────┐
    │  Linear → LayerNorm → GELU    │  ← trainable head
    │  Dropout(0.3)                 │
    │  Linear → 4 logits            │
    └───────────────────────────────┘
    """

    def __init__(
        self,
        checkpoint: str = WAV2VEC2_CHECKPOINT,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.30,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(checkpoint)

        hidden = self.encoder.config.hidden_size  # 768 for wav2vec2-base

        # Freeze CNN feature extractor (always recommended)
        self.encoder.feature_extractor._freeze_parameters()

        if freeze_feature_extractor:
            # Also freeze transformer layers initially
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Keep the top-2 transformer layers trainable for fast adaptation
            for layer in self.encoder.encoder.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def unfreeze_encoder(self) -> None:
        """Call this after the warm-up phase to fine-tune all encoder layers."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        # Always keep CNN frozen
        self.encoder.feature_extractor._freeze_parameters()
        log.info("Encoder layers unfrozen for full fine-tuning.")

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_values)
        # Mean-pool across time dimension
        hidden = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_size)
        logits = self.classifier(hidden)                # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  6. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(file_list: List[Tuple[str, int]]) -> torch.Tensor:
    """Inverse-frequency class weights to handle class imbalance."""
    label_counts = np.zeros(NUM_CLASSES)
    for _, label in file_list:
        label_counts[label] += 1
    total = label_counts.sum()
    weights = total / (NUM_CLASSES * label_counts + 1e-9)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(file_list: List[Tuple[str, int]]) -> WeightedRandomSampler:
    """Per-sample weights so each mini-batch is balanced across classes."""
    label_counts = np.zeros(NUM_CLASSES)
    for _, label in file_list:
        label_counts[label] += 1

    sample_weights = []
    for _, label in file_list:
        sample_weights.append(1.0 / (label_counts[label] + 1e-9))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class EarlyStopping:
    """Stop training when val loss stops improving for *patience* epochs."""

    def __init__(self, patience: int = 6, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                log.info(f"Early stopping triggered after {self.patience} epochs "
                         f"without improvement.")
                return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state:
            model.load_state_dict(self.best_state)
            log.info("Restored best model weights.")


# ─────────────────────────────────────────────────────────────────────────────
#  7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    data_dir: str,
    output_model_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    unfreeze_after_epoch: int = 5,   # warm-up then unfreeze encoder
    patience: int = 7,
    noise_level: float = 0.07,
    checkpoint: str = WAV2VEC2_CHECKPOINT,
) -> None:
    """
    Full training pipeline:
      1. Load dataset splits
      2. Build Wav2Vec2 classifier
      3. Warm-up: train head only (encoder frozen)
      4. Fine-tune: unfreeze encoder layers
      5. Early stopping on validation loss
      6. Save best model + metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load feature extractor ──────────────────────────────────────────────
    log.info(f"Loading feature extractor from '{checkpoint}' …")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)

    # ── Build splits ────────────────────────────────────────────────────────
    train_list, val_list, test_list = split_dataset(data_dir)
    if len(train_list) == 0:
        raise RuntimeError(
            f"No samples found in '{data_dir}'. "
            "Run --mode build_dataset first."
        )

    train_ds = StutterDataset(data_dir, feature_extractor, file_list=train_list)
    val_ds   = StutterDataset(data_dir, feature_extractor, file_list=val_list)

    sampler = build_weighted_sampler(train_list)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # ── Class weights for loss ──────────────────────────────────────────────
    class_weights = compute_class_weights(train_list).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ───────────────────────────────────────────────────────────────
    log.info(f"Initialising StutterClassifier (backbone: {checkpoint}) …")
    model = StutterClassifier(
        checkpoint=checkpoint,
        freeze_feature_extractor=True,   # freeze all encoder layers initially
    ).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-3,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)

    best_val_acc = 0.0
    history: List[Dict] = []

    log.info(f"Starting training for {epochs} epoch(s) …")
    log.info("=" * 64)

    for epoch in range(1, epochs + 1):

        # Unfreeze encoder after warm-up
        if epoch == unfreeze_after_epoch + 1:
            model.unfreeze_encoder()
            # Rebuild optimizer with new trainable params at lower LR
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate * 0.1,
                weight_decay=1e-3,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=epochs - unfreeze_after_epoch, eta_min=1e-7
            )

        # ── Training phase ─────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_input, batch_labels in train_loader:
            batch_input  = batch_input.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_input)
            loss = criterion(logits, batch_labels)
            loss.backward()
            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(batch_labels)
            preds          = logits.argmax(dim=-1)
            train_correct += (preds == batch_labels).sum().item()
            train_total   += len(batch_labels)

        scheduler.step()

        # ── Validation phase ───────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_input, batch_labels in val_loader:
                batch_input  = batch_input.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_input)
                loss = criterion(logits, batch_labels)

                val_loss    += loss.item() * len(batch_labels)
                preds        = logits.argmax(dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total   += len(batch_labels)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch_labels.cpu().tolist())

        train_loss_avg = train_loss / train_total
        train_acc      = train_correct / train_total
        val_loss_avg   = val_loss / val_total
        val_acc        = val_correct / val_total
        val_f1         = f1_score(all_labels, all_preds,
                                  average="macro", zero_division=0)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss_avg, 4),
            "train_acc":  round(train_acc,      4),
            "val_loss":   round(val_loss_avg,   4),
            "val_acc":    round(val_acc,         4),
            "val_f1":     round(val_f1,          4),
        }
        history.append(row)

        log.info(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.3%}  "
            f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.3%} "
            f"val_f1={val_f1:.3f}"
        )

        # Track best validation accuracy for checkpoint naming
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if early_stop.step(val_loss_avg, model):
            break

    # ── Restore best weights ───────────────────────────────────────────────
    early_stop.restore_best(model)
    log.info(f"\nBest validation accuracy reached: {best_val_acc:.3%}")

    # ── Save model ─────────────────────────────────────────────────────────
    save_model(model, feature_extractor, output_model_dir,
               history=history, test_list=test_list,
               data_dir=data_dir, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  8. MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model: StutterClassifier,
    feature_extractor: Wav2Vec2FeatureExtractor,
    output_dir: str,
    history: Optional[List[Dict]] = None,
    test_list: Optional[List[Tuple[str, int]]] = None,
    data_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Saves:
        output_dir/
            model.pt            — full model state dict
            feature_extractor/  — HuggingFace processor config
            metadata.json       — label map, accuracy, training history
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = out / "model.pt"
    torch.save(model.state_dict(), str(weights_path))
    log.info(f"Model weights saved → {weights_path}")

    # Save feature extractor
    fe_path = out / "feature_extractor"
    feature_extractor.save_pretrained(str(fe_path))
    log.info(f"Feature extractor saved → {fe_path}")

    # Evaluate on test set if available
    test_metrics: Optional[Dict] = None
    if test_list and data_dir and device is not None:
        test_metrics = evaluate_model(
            model, feature_extractor, data_dir, test_list, device
        )

    # Save metadata
    metadata = {
        "label_map":   LABEL_MAP,
        "label_names": LABEL_NAMES,
        "checkpoint":  WAV2VEC2_CHECKPOINT,
        "num_classes": NUM_CLASSES,
        "sample_rate": SAMPLE_RATE,
        "max_audio_len": MAX_AUDIO_LEN,
        "training_history": history or [],
        "test_metrics": test_metrics,
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    log.info(f"Metadata saved → {meta_path}")
    log.info(f"✅ Model artefacts saved to '{output_dir}'.")


def load_model(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> Tuple[StutterClassifier, Wav2Vec2FeatureExtractor]:
    """
    Load a previously saved StutterClassifier and its feature extractor.

    Returns (model, feature_extractor) ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = Path(model_dir) / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    checkpoint = metadata.get("checkpoint", WAV2VEC2_CHECKPOINT)

    fe_path = Path(model_dir) / "feature_extractor"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(fe_path))

    model = StutterClassifier(checkpoint=checkpoint, freeze_feature_extractor=False)
    weights_path = Path(model_dir) / "model.pt"
    model.load_state_dict(
        torch.load(str(weights_path), map_location=device)
    )
    model.to(device)
    model.eval()

    log.info(f"Model loaded from '{model_dir}' on {device}.")
    return model, feature_extractor


# ─────────────────────────────────────────────────────────────────────────────
#  9. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: StutterClassifier,
    feature_extractor: Wav2Vec2FeatureExtractor,
    data_dir: str,
    file_list: List[Tuple[str, int]],
    device: torch.device,
    batch_size: int = 16,
) -> Dict:
    """
    Runs inference on *file_list*, prints a full classification report, and
    returns a dict with accuracy, macro-F1, and per-class metrics.
    """
    test_ds = StutterDataset(data_dir, feature_extractor, file_list=file_list)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_input, batch_labels in test_loader:
            batch_input = batch_input.to(device)
            logits = model(batch_input)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(
        all_labels, all_preds,
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
    )

    log.info("\n" + "=" * 64)
    log.info(f"TEST RESULTS — Accuracy: {acc:.3%}  Macro-F1: {f1:.3f}")
    log.info("\n" + report)
    log.info("Confusion Matrix:")
    for row in cm:
        log.info("  " + "  ".join(f"{v:4d}" for v in row))
    log.info("=" * 64)

    return {
        "accuracy":         round(acc, 4),
        "macro_f1":         round(f1, 4),
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  10. INFERENCE HELPER  (use in Gradio / production)
# ─────────────────────────────────────────────────────────────────────────────

def predict_stutter(
    audio_path: str,
    model: StutterClassifier,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Single-file inference.

    Returns:
        {
            "label": "repetition",
            "label_ar": "تكرار",
            "confidence": 0.87,
            "all_scores": {"fluent": 0.05, "repetition": 0.87, ...}
        }
    """
    if device is None:
        device = next(model.parameters()).device

    LABEL_AR = {
        "fluent":       "نطق سليم (طليق)",
        "repetition":   "تكرار صوتي",
        "prolongation": "إطالة صوتية",
        "block":        "انسداد / توقف مفاجئ",
    }

    audio, sr = load_audio(audio_path, sr=SAMPLE_RATE)
    audio = trim_audio(audio)
    audio = normalize_audio(audio)

    max_samples = int(MAX_AUDIO_LEN * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))

    inputs = feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
    )
    input_values = inputs.input_values.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_values)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]

    return {
        "label":      pred_label,
        "label_ar":   LABEL_AR.get(pred_label, pred_label),
        "confidence": round(probs[pred_idx], 4),
        "all_scores": {
            LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  11. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stutter Detection Model — Dataset Builder & Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["download_clean", "build_dataset", "train", "evaluate"],
        required=True,
        help="Operation to perform.",
    )
    # download_clean args
    p.add_argument("--everyayah_out", default="mp3/quran",
                   help="Where to save downloaded EveryAyah MP3s.")
    p.add_argument("--num_files",  type=int, default=300,
                   help="Total number of ayah files to download.")
    p.add_argument("--reciters",   nargs="*", default=None,
                   help="Reciter folder names (space-separated). "
                        "Defaults to all 6 built-in reciters.")
    p.add_argument("--surahs",     nargs="*", type=int, default=None,
                   help="Surah numbers to download (1-114). "
                        "Defaults to all surahs.")
    p.add_argument("--delay",      type=float, default=0.25,
                   help="Seconds between HTTP requests.")
    # build_dataset args
    p.add_argument("--clean_dir",   default="data/clean",
                   help="Directory with clean (fluent) audio files.")
    p.add_argument("--output_dir",  default="data/stuttered",
                   help="Output directory for generated synthetic dataset.")
    p.add_argument("--samples_per_class", type=int, default=500,
                   help="How many samples to generate per stutter class.")
    # train / evaluate args
    p.add_argument("--data_dir",    default="data/stuttered",
                   help="Root directory of the (already built) dataset.")
    p.add_argument("--output_model", default="models/stutter_classifier",
                   help="Directory to save the trained model.")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Initial learning rate.")
    p.add_argument("--unfreeze_epoch", type=int, default=5,
                   help="Epoch after which to unfreeze the Wav2Vec2 encoder.")
    p.add_argument("--patience",     type=int,   default=7,
                   help="Early-stopping patience (epochs).")
    p.add_argument("--noise_level",  type=float, default=0.07,
                   help="Max white-noise amplitude for synthetic samples.")
    p.add_argument("--checkpoint",   default=WAV2VEC2_CHECKPOINT,
                   help="HuggingFace Wav2Vec2 checkpoint to use.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "download_clean":
        log.info("Mode: download_clean  (source: everyayah.com)")
        download_clean_from_everyayah(
            output_dir=args.everyayah_out,
            num_files=args.num_files,
            reciters=args.reciters or None,
            surahs=args.surahs or None,
            delay=args.delay,
        )

    elif args.mode == "build_dataset":
        log.info("Mode: build_dataset")
        build_dataset_from_clean_audio(
            clean_dir=args.clean_dir,
            output_dir=args.output_dir,
            samples_per_class=args.samples_per_class,
            noise_level=args.noise_level,
        )

    elif args.mode == "train":
        log.info("Mode: train")
        train_model(
            data_dir=args.data_dir,
            output_model_dir=args.output_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            unfreeze_after_epoch=args.unfreeze_epoch,
            patience=args.patience,
            noise_level=args.noise_level,
            checkpoint=args.checkpoint,
        )

    elif args.mode == "evaluate":
        log.info("Mode: evaluate")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(Path(args.output_model) / "feature_extractor")
        )
        _, _, test_list = split_dataset(args.data_dir)
        model, feature_extractor = load_model(args.output_model, device)
        evaluate_model(model, feature_extractor, args.data_dir,
                       test_list, device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
