import gradio as gr
import whisper
import cv2
import mediapipe as mp
import math
import torch
import numpy as np
import threading
import time
import re
import difflib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque
import warnings

ROOT = Path(__file__).resolve().parents[2]
WHISPER_DIR = ROOT / "models" / "whisper_models"

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass
class AppConfig:
    """Centralized configuration for the speech therapy application."""
    TARGET_WORD: str = "قطار"
    TARGET_WORD_NORMALIZED: str = "قطار"
    
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_LANGUAGE: str = "ar"
    
    MAX_FACES: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    MIN_VERTICAL_OPENING: float = 15.0
    MIN_ASPECT_RATIO: float = 1.8
    SMOOTHING_WINDOW: int = 5
    
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.7
    FONT_THICKNESS: int = 2
    
    SIMILARITY_THRESHOLD: float = 0.8

CONFIG = AppConfig()

LIP_LANDMARKS = {
    'top': 13, 'bottom': 14, 'left': 61, 'right': 291
}

COLORS = {
    'success': (0, 255, 0), 'warning': (0, 165, 255), 'error': (0, 0, 255),
    'info': (255, 255, 255), 'accent': (255, 0, 0), 'neutral': (128, 128, 128)
}

# =============================================================================
# THREAD-SAFE STATE MANAGEMENT
# =============================================================================

class ThreadSafeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._status = "Initializing..."
        self._confidence_history = deque(maxlen=CONFIG.SMOOTHING_WINDOW)
        self._last_update = time.time()
        self._frame_count = 0
        self._fps = 0.0
    
    @property
    def status(self) -> str:
        with self._lock: return self._status
    
    @status.setter
    def status(self, value: str):
        with self._lock:
            self._status = value
            self._last_update = time.time()
    
    @property
    def mouth_correct(self) -> bool:
        with self._lock:
            if len(self._confidence_history) < 3: return False
            return sum(self._confidence_history) / len(self._confidence_history) > 0.6
    
    def update_mouth_state(self, is_correct: bool):
        with self._lock:
            self._confidence_history.append(1.0 if is_correct else 0.0)
    
    def get_fps(self) -> float:
        with self._lock: return self._fps
    
    def update_fps(self):
        with self._lock:
            self._frame_count += 1
            elapsed = time.time() - self._last_update
            if elapsed > 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_update = time.time()

state = ThreadSafeState()

# =============================================================================
# MODEL INITIALIZATION (OPTIMIZED & THREAD-SAFE)
# =============================================================================

class ModelManager:
    def __init__(self):
        self.lock = threading.Lock() # Prevents race conditions in multi-threaded Gradio
        self.whisper_model = None
        self.face_mesh = None
        self._device = None
        self._init_device()
        self._init_whisper()
        self._init_mediapipe()
    
    def _init_device(self):
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        print(f"🔄 Loading AI models on [{self._device.upper()}]...")
    
    def _init_whisper(self):
        try:
            self.whisper_model = whisper.load_model(CONFIG.WHISPER_MODEL_SIZE, device=self._device, download_root=str(WHISPER_DIR))
            print("✅ Whisper ASR model loaded successfully")
        except Exception as e:
            print(f"⚠️ Whisper loading failed: {e}. Falling back to CPU...")
            self._device = "cpu"
            self.whisper_model = whisper.load_model(CONFIG.WHISPER_MODEL_SIZE, device="cpu", download_root=str(WHISPER_DIR))
    
    def _init_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # HUGE SPEEDUP: refine_landmarks=False disables eye/iris tracking which we don't need
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=CONFIG.MAX_FACES,
            refine_landmarks=False, 
            min_detection_confidence=CONFIG.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=CONFIG.MIN_TRACKING_CONFIDENCE
        )
        print("✅ MediaPipe Face Mesh initialized")
    
    def process_frame(self, rgb_frame):
        with self.lock:
            return self.face_mesh.process(rgb_frame)

    def transcribe(self, audio_path: str) -> dict:
        with self.lock:
            if not self.whisper_model:
                raise RuntimeError("Whisper model not initialized")
            return self.whisper_model.transcribe(
                audio_path,
                language=CONFIG.WHISPER_LANGUAGE,
                fp16=(self._device == "cuda"),
                condition_on_previous_text=False, # Prevents hallucination & speeds up
                no_speech_threshold=0.6,
                initial_prompt=CONFIG.TARGET_WORD   # Helps model focus on target word
            )

try:
    models = ModelManager()
except Exception as e:
    print(f"🚨 Critical initialization error: {e}")
    raise

# =============================================================================
# GEOMETRY & CLINICAL ANALYSIS (OPTIMIZED)
# =============================================================================

class MouthAnalyzer:
    @staticmethod
    def extract_landmarks(face_landmarks, w: int, h: int) -> dict:
        # Using tuples instead of dicts for faster access and less memory
        landmarks = {}
        for name, idx in LIP_LANDMARKS.items():
            pt = face_landmarks.landmark[idx]
            landmarks[name] = (int(pt.x * w), int(pt.y * h), pt.z)
        return landmarks
    
    @staticmethod
    def calculate_geometry(landmarks: dict) -> dict:
        top, bottom = landmarks['top'], landmarks['bottom']
        left, right = landmarks['left'], landmarks['right']
        
        # math.hypot is faster and cleaner than manual sqrt(dx**2 + dy**2)
        v_dist = math.hypot(top[0] - bottom[0], top[1] - bottom[1])
        h_dist = math.hypot(left[0] - right[0], left[1] - right[1])
        ratio = h_dist / (v_dist + 1e-6)
        
        return {
            'vertical': v_dist, 'horizontal': h_dist, 'ratio': ratio,
            'is_open': v_dist > CONFIG.MIN_VERTICAL_OPENING,
            'is_proportional': ratio > CONFIG.MIN_ASPECT_RATIO
        }
    
    @staticmethod
    def assess_clinical_state(geometry: dict) -> Tuple[bool, str, float]:
        v_dist, ratio = geometry['vertical'], geometry['ratio']
        
        if v_dist > CONFIG.MIN_VERTICAL_OPENING and ratio > CONFIG.MIN_ASPECT_RATIO:
            confidence = min(1.0, (v_dist / 30.0) * (ratio / 2.5))
            return True, "✓ CORRECT: Mouth properly positioned for 'Qaf'", confidence
        elif v_dist > CONFIG.MIN_VERTICAL_OPENING:
            return False, "⚠ ADJUST: Open mouth wider (horizontal stretch needed)", 0.3
        elif ratio > CONFIG.MIN_ASPECT_RATIO:
            return False, "⚠ ADJUST: Open mouth vertically more", 0.3
        else:
            return False, "✗ INCORRECT: Please open and relax your mouth", 0.0

# =============================================================================
# FAST VIDEO PROCESSING PIPELINE
# =============================================================================

def draw_lips_contour(frame, face_landmarks, w, h, color):
    """Extremely fast manual drawing of lips instead of slow mp_drawing."""
    if not hasattr(draw_lips_contour, "lip_indices"):
        indices = set()
        for start, end in mp.solutions.face_mesh.FACEMESH_LIPS:
            indices.add(start)
            indices.add(end)
        draw_lips_contour.lip_indices = list(indices)
        draw_lips_contour.connections = list(mp.solutions.face_mesh.FACEMESH_LIPS)
        
    points = {}
    for idx in draw_lips_contour.lip_indices:
        pt = face_landmarks.landmark[idx]
        points[idx] = (int(pt.x * w), int(pt.y * h))
        
    for start_idx, end_idx in draw_lips_contour.connections:
        cv2.line(frame, points[start_idx], points[end_idx], color, 2)

def draw_ar_overlay(frame: np.ndarray, landmarks: dict, geometry: dict, color: tuple):
    top, bottom = landmarks['top'], landmarks['bottom']
    left, right = landmarks['left'], landmarks['right']
    
    cv2.circle(frame, (top[0], top[1]), 6, color, -1)
    cv2.circle(frame, (bottom[0], bottom[1]), 6, color, -1)
    cv2.circle(frame, (left[0], left[1]), 5, color, -1)
    cv2.circle(frame, (right[0], right[1]), 5, color, -1)
    
    cv2.line(frame, (top[0], top[1]), (bottom[0], bottom[1]), COLORS['accent'], 2)
    cv2.line(frame, (left[0], left[1]), (right[0], right[1]), COLORS['accent'], 2)
    
    mid_v = ((top[0] + bottom[0]) // 2, (top[1] + bottom[1]) // 2)
    cv2.putText(frame, f"V:{geometry['vertical']:.1f}", (mid_v[0] + 10, mid_v[1]), CONFIG.FONT, 0.5, COLORS['info'], 1)
    
    mid_h = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
    cv2.putText(frame, f"H:{geometry['horizontal']:.1f}", (mid_h[0], mid_h[1] - 10), CONFIG.FONT, 0.5, COLORS['info'], 1)

def draw_status_banner(frame: np.ndarray, text: str, color: tuple):
    h, w = frame.shape[:2]
    # Solid rectangle is much faster than cv2.addWeighted
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, text, (15, 35), CONFIG.FONT, CONFIG.FONT_SCALE, color, 2)

def draw_debug_info(frame: np.ndarray):
    fps = state.get_fps()
    state.update_fps()
    debug_text = f"FPS: {fps:.1f} | Device: {models._device.upper()}"
    h, w = frame.shape[:2]
    text_size = cv2.getTextSize(debug_text, CONFIG.FONT, 0.5, 1)[0]
    cv2.putText(frame, debug_text, (w - text_size[0] - 10, h - 10), CONFIG.FONT, 0.5, COLORS['info'], 1)

def process_video_frame(frame: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    if frame is None:
        return None, "Waiting for camera..."
    
    # 🛠️ FIX: Ensure writable copy (Crucial for Gradio streaming compatibility)
    if not frame.flags.writeable:
        frame = np.copy(frame)
        
    h, w = frame.shape[:2]
    
    # MASSIVE SPEEDUP: Process at lower resolution
    scale = 320 / w
    new_h = int(h * scale)
    small_frame = cv2.resize(frame, (320, new_h))
    
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small.flags.writeable = False
    
    results = models.process_frame(rgb_small)
    rgb_small.flags.writeable = True
    
    color = COLORS['neutral']
    status_text = "👤 No face detected"
    state.update_mouth_state(False)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            draw_lips_contour(frame, face_landmarks, w, h, COLORS['accent'])
            
            landmarks = MouthAnalyzer.extract_landmarks(face_landmarks, w, h)
            geometry = MouthAnalyzer.calculate_geometry(landmarks)
            is_correct, status_text, confidence = MouthAnalyzer.assess_clinical_state(geometry)
            state.update_mouth_state(is_correct)
            
            if is_correct: color = COLORS['success']
            elif confidence > 0: color = COLORS['warning']
            else: color = COLORS['error']
            
            draw_ar_overlay(frame, landmarks, geometry, color)
            
    state.status = status_text
    draw_status_banner(frame, status_text, color)
    draw_debug_info(frame)
    
    return frame, state.status
# =============================================================================
# AUDIO PROCESSING & MULTIMODAL EVALUATION
# =============================================================================

class TextNormalizer:
    SUBSTITUTION_PATTERNS = {
        'طار': 'omission_of_qaf', 'تار': 'omission_of_qaf', 'تتار': 'distortion',
        'كار': 'substitution_qaf_to_kaf', 'قار': 'partial_vowel_error', 'قر': 'truncation'
    }
    
    @staticmethod
    def normalize(text: str) -> str:
        if not text: return ""
        # Fast regex to remove Arabic diacritics (Tashkeel)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        # Normalize Alef, Teh Marbuta, etc.
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي')
        return text.strip()

    @staticmethod
    def calculate_similarity(s1: str, s2: str) -> float:
        if not s1 or not s2: return 0.0
        # difflib is implemented in C and much faster than pure Python Levenshtein
        return difflib.SequenceMatcher(None, s1, s2).ratio()

def format_clinical_report(status: str, audio_result: str, visual_result: str,
                          diagnosis: str, recommendations: List[str], severity: str = "unknown") -> str:
    status_emoji = {"SUCCESS": "🏆", "PARTIAL": "⚠️", "FAILED": "❌", "WARNING": "🔶"}.get(status, "ℹ️")
    severity_indicator = {"none": "🟢", "low": "🟡", "moderate": "🟠", "high": "🔴"}.get(severity, "⚪")
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║  {status_emoji}  MULTIMODAL SPEECH THERAPY EVALUATION REPORT          ║
╠══════════════════════════════════════════════════════════════════╣
║  Target Word: {CONFIG.TARGET_WORD:>45} ║
║  Severity:   {severity_indicator} {severity.upper():<45} ║
╠══════════════════════════════════════════════════════════════════╣
║  📊 AUDIO ANALYSIS                                               ║
║  {audio_result:<64} ║
╠══════════════════════════════════════════════════════════════════╣
║  📹 VISUAL ANALYSIS                                              ║
║  {visual_result:<64} ║
╠══════════════════════════════════════════════════════════════════╣
║  🩺 CLINICAL DIAGNOSIS                                             ║
║  {diagnosis:<64} ║
╠══════════════════════════════════════════════════════════════════╣
║  📋 RECOMMENDATIONS                                                ║
"""
    for i, rec in enumerate(recommendations, 1):
        wrapped = [rec[j:j+60] for j in range(0, len(rec), 60)]
        for line in wrapped:
            report += f"║  {i}. {line:<62} ║\n"
            i = " "
    report += "╚══════════════════════════════════════════════════════════════════╝"
    return report

def evaluate_multimodal_performance(audio_path: Optional[str]) -> Tuple[str, str]:
    if not audio_path:
        return format_clinical_report("ERROR", "No audio provided", state.status, "Please record audio.", ["Record audio using the microphone"]), "Please record audio first."
    
    if "No face" in state.status or "Initializing" in state.status:
        return format_clinical_report("WARNING", "Pending", state.status, "Camera not ready.", ["Position face in camera frame"]), "Camera not ready."
    
    try:
        result = models.transcribe(audio_path)
        transcribed_text = result.get("text", "").strip()
        clean_text = TextNormalizer.normalize(transcribed_text)
        
        similarity = TextNormalizer.calculate_similarity(clean_text, CONFIG.TARGET_WORD_NORMALIZED)
        is_audio_correct = similarity >= CONFIG.SIMILARITY_THRESHOLD
        
        error_type = None
        for pattern, error_name in TextNormalizer.SUBSTITUTION_PATTERNS.items():
            if pattern in clean_text or clean_text == pattern:
                error_type = error_name
                break
        
        mouth_correct = state.mouth_correct
        
        if is_audio_correct and mouth_correct:
            report = format_clinical_report("SUCCESS", f"Correct: '{clean_text}' ({similarity:.0%})", "Mouth positioning correct", "Excellent! Proper motor planning.", ["Proceed to next word"], "none")
        elif is_audio_correct and not mouth_correct:
            report = format_clinical_report("PARTIAL", f"Correct word: '{clean_text}'", "Mouth positioning inconsistent", "Possible speech apraxia indicator.", ["Focus on lip rounding", "Practice with mirror feedback"], "moderate")
        elif error_type == 'omission_of_qaf':
            report = format_clinical_report("FAILED", f"Omission: '{clean_text}' (Missing /q/)", state.status, "Dropping initial stop consonant 'Qaf'.", ["Practice isolated /q/", "Contrast training: Qaf vs. Ta"], "moderate")
        else:
            report = format_clinical_report("FAILED", f"Mispronunciation: '{clean_text}'", state.status, "General articulation error.", ["Break into syllables", "Practice minimal pairs"], "high" if not mouth_correct else "moderate")
            
        return report, "Evaluation complete. See report below."
            
    except Exception as e:
        return format_clinical_report("SYSTEM_ERROR", f"Error: {str(e)}", state.status, "Technical error.", ["Retry recording"]), f"Error: {str(e)}"

# =============================================================================
# GRADIO INTERFACE (STEP-BY-STEP TABS)
# =============================================================================

def create_interface():
    custom_css = """
    .clinical-report { 
        font-family: 'Consolas', 'Courier New', monospace; 
        white-space: pre-wrap;
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #45475a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-size: 14px;
        line-height: 1.5;
    }
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    """
    
    with gr.Blocks(title="AI Speech Therapy", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 AI-Powered Speech Therapy: Multimodal Assessment")
        gr.Markdown(f"### 🎯 Clinical Target: **{CONFIG.TARGET_WORD}** (Arabic emphatic stop /q/)")
        
        with gr.Tabs() as tabs:
            # STEP 1: VISUAL CALIBRATION
            with gr.Tab("📹 Step 1: Visual Calibration", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Instructions")
                        gr.Markdown("1. Position face 30-50cm from camera\n2. Ensure good frontal lighting\n3. Relax jaw and slightly open mouth\n4. Wait for the **green** indicator")
                        status_box = gr.Textbox(label="Current Motor State", value="Waiting for camera...", interactive=False)
                    with gr.Column(scale=2):
                        video_input = gr.Image(sources=["webcam"], streaming=True, label="Live AR Telemetry", height=480)
                        video_output = gr.Image(label="Processed Feed with Lip Tracking", height=480)
                with gr.Row():
                    next_btn_1 = gr.Button("Proceed to Audio Recording ➡️", variant="primary", scale=1)
                    gr.Markdown("") # Spacer
            
            # STEP 2: PHONEMIC PRODUCTION
            with gr.Tab("🎙️ Step 2: Phonemic Production", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Instructions")
                        gr.Markdown("1. Click microphone to record\n2. Say **'قطار'** clearly\n3. Keep recording around 2-3 seconds")
                        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio Sample")
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Audio Status")
                        audio_status = gr.Textbox(label="Recording Status", value="Ready to record...", interactive=False)
                with gr.Row():
                    prev_btn_2 = gr.Button("⬅️ Back to Calibration")
                    next_btn_2 = gr.Button("Proceed to Evaluation ➡️", variant="primary")
            
            # STEP 3: EVALUATION
            with gr.Tab("📊 Step 3: Multimodal Evaluation", id=2):
                with gr.Row():
                    eval_btn = gr.Button("🔬 Run Multimodal Evaluation", variant="primary", size="lg")
                    reset_btn = gr.Button("🔄 Reset Session", variant="secondary")
                clinical_output = gr.Textbox(label="Clinical Assessment Report", lines=25, elem_classes=["clinical-report"])
                with gr.Row():
                    prev_btn_3 = gr.Button("⬅️ Back to Recording")
                    gr.Markdown("") # Spacer
        
        # --- EVENT HANDLERS ---
        
        # Video Stream (Updates both video feed and status box in real-time)
        video_input.stream(fn=process_video_frame, inputs=video_input, outputs=[video_output, status_box])
        
        # Tab Navigation
        next_btn_1.click(fn=lambda: gr.Tabs(selected=1), outputs=tabs)
        prev_btn_2.click(fn=lambda: gr.Tabs(selected=0), outputs=tabs)
        next_btn_2.click(fn=lambda: gr.Tabs(selected=2), outputs=tabs)
        prev_btn_3.click(fn=lambda: gr.Tabs(selected=1), outputs=tabs)
        
        # Evaluation
        eval_btn.click(fn=evaluate_multimodal_performance, inputs=audio_input, outputs=[clinical_output, audio_status])
        
        # Reset Session
        def reset_session():
            state._confidence_history.clear()
            state.status = "Session reset. Position face in frame."
            return None, None, "Session reset. Ready for new evaluation.", gr.Tabs(selected=0)
            
        reset_btn.click(fn=reset_session, inputs=None, outputs=[audio_input, clinical_output, status_box, tabs])
        
    return demo

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AI SPEECH THERAPY - MULTIMODAL CLINICAL SYSTEM")
    print("=" * 60)
    print(f"  Target Word: {CONFIG.TARGET_WORD}")
    print(f"  Device: {models._device.upper()}")
    print(f"  Whisper: {CONFIG.WHISPER_MODEL_SIZE}")
    print("=" * 60)
    
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )