"""
interactive_story.py  —  قصة تفاعلية متعددة الشخصيات لتدريب نطق الأطفال
──────────────────────────────────────────────────────────────────────────
التدفق لكل مشهد:
  1. الشخصية تتكلم  ← الطفل يضغط "استمع" (TTS بـ gTTS — يحتاج إنترنت)
  2. يظهر النص المطلوب قراءته
  3. الطفل يسجّل صوته
  4. يضغط "تحقق" ← Whisper يحوّل الصوت إلى نص ← Levenshtein يحسب الدقة
  5. إن اجتاز العتبة ← يظهر زر "التالي"
"""

import os
import io
import numpy as np
import librosa
import soundfile as sf
import gradio as gr
import whisper
import torch
import Levenshtein
from PIL import Image, ImageDraw

# ═══════════════════════════════════════════════════════════════
# 1. مسارات المشروع
# ═══════════════════════════════════════════════════════════════
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_DIR = os.path.join(ROOT, "models", "whisper_models")
TEMP_DIR    = os.path.join(ROOT, "temp_story")
os.makedirs(TEMP_DIR, exist_ok=True)

# ── تحميل Whisper (محلي) ─────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ تحميل Whisper على ({device})...")
asr_model = whisper.load_model("turbo", device=device, download_root=WHISPER_DIR)
print("✅ Whisper جاهز!")

# ── تحميل gTTS (لا يحتاج نماذج محلية، يتطلب اتصالاً بالإنترنت) ──────
from gtts import gTTS
print("✅ gTTS جاهز لتوليد الصوت العربي!")


# ═══════════════════════════════════════════════════════════════
# 2. بيانات القصة  (محتوى وهمي — Dummy Content)
#    لإضافة قصة حقيقية: عدّل قائمة STORY وأضف مسارات الصور الحقيقية
#    في حقل "image_path" وأصوات الخلفية في "sound_path"
# ═══════════════════════════════════════════════════════════════
STORY = [
    {
        "scene_title":  "🌅 بداية المغامرة",
        "bg_color":     "#5DADE2",          # أزرق سماوي
        "character":    "الراوي",
        "char_color":   "#9B59B6",          # بنفسجي
        "narration":    "في يومٍ مشمسٍ جميل، قرّر الأرنب الصغير أن يغامر ويسافر عبر الغابة الكبيرة.",
        "child_text":   "يومٌ مشمسٌ جميل",
        "threshold":    60,
        "hint":         "💡 يَوْمٌ — مُشْمِسٌ — جَمِيل",
        "image_path":   None,               # ← ضع مسار صورة حقيقية هنا
        "sound_path":   None,               # ← ضع مسار صوت خلفية هنا
    },
    {
        "scene_title":  "🌲 الغابة الخضراء",
        "bg_color":     "#27AE60",          # أخضر
        "character":    "الأرنب",
        "char_color":   "#ECF0F1",          # أبيض
        "narration":    "دخل الأرنب الغابة وصاح بصوت عالٍ: هل هناك أحد يسمعني؟",
        "child_text":   "هل هناك أحد",
        "threshold":    65,
        "hint":         "💡 هَلْ — هُنَاكَ — أَحَد",
        "image_path":   None,
        "sound_path":   None,
    },
    {
        "scene_title":  "🦊 لقاء الثعلب",
        "bg_color":     "#E67E22",          # برتقالي
        "character":    "الثعلب",
        "char_color":   "#D35400",          # بني محروق
        "narration":    "فجأة، خرج الثعلب الذكي من خلف شجرة وقال بصوت ودود: مرحباً يا أرنب، أنا صديقك!",
        "child_text":   "مرحباً يا أرنب",
        "threshold":    65,
        "hint":         "💡 مَرْحَبَاً — يَا — أَرْنَب",
        "image_path":   None,
        "sound_path":   None,
    },
    {
        "scene_title":  "🦁 الأسد الطيب",
        "bg_color":     "#784212",          # بني
        "character":    "الأسد",
        "char_color":   "#F1C40F",          # ذهبي
        "narration":    "سمع الأسد الكبير الأصوات فجاء وقال بهدوء: لا تخافوا، هذه الغابة آمنة لجميع الأصدقاء.",
        "child_text":   "الغابة آمنة للأصدقاء",
        "threshold":    70,
        "hint":         "💡 الغَابَة — آمِنَة — لِلأَصْدِقَاء",
        "image_path":   None,
        "sound_path":   None,
    },
    {
        "scene_title":  "⭐ نهاية سعيدة",
        "bg_color":     "#C0392B",          # أحمر غروب
        "character":    "الراوي",
        "char_color":   "#9B59B6",          # بنفسجي
        "narration":    "وهكذا صار الأرنب والثعلب والأسد أصدقاء حقيقيين وعاشوا معاً بسعادة وهناء.",
        "child_text":   "عاشوا بسعادة وهناء",
        "threshold":    60,
        "hint":         "💡 عَاشُوا — بِسَعَادَة — وَهَنَاء",
        "image_path":   None,
        "sound_path":   None,
    },
]

TOTAL = len(STORY)


# ═══════════════════════════════════════════════════════════════
# 3. دوال توليد الصور الوهمية (Dummy Images)
#    إذا وُجد image_path حقيقي يُستخدم بدلاً منها
# ═══════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_scene_image(scene: dict) -> Image.Image:
    """خلفية المشهد: صورة ملونة بسيطة أو صورة حقيقية إن وُجدت"""
    if scene.get("image_path") and os.path.exists(scene["image_path"]):
        return Image.open(scene["image_path"]).resize((640, 280))

    w, h = 640, 280
    img = Image.new("RGB", (w, h), scene["bg_color"])
    draw = ImageDraw.Draw(img)
    # إطار أبيض
    draw.rectangle([6, 6, w - 6, h - 6], outline="white", width=5)
    # شريط علوي شبه شفاف
    overlay = Image.new("RGB", (w, 50), (0, 0, 0))
    img.paste(overlay, (0, 0))
    draw.text((w // 2, 25), scene["scene_title"], fill="white", anchor="mm")
    return img


def make_char_image(scene: dict) -> Image.Image:
    """أيقونة الشخصية: دائرة ملونة"""
    size = 160
    r, g, b = _hex_to_rgb(scene["char_color"])
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    m = 6
    draw.ellipse([m, m, size - m, size - m], fill=(r, g, b), outline="white", width=5)
    return img


# ═══════════════════════════════════════════════════════════════
# 4. دوال TTS / ASR
# ═══════════════════════════════════════════════════════════════

def generate_narration(scene_idx: int) -> str | None:
    """يولّد صوت الشخصية باستخدام gTTS ويعيد مساره"""
    scene = STORY[scene_idx]
    out_path = os.path.join(TEMP_DIR, f"narration_{scene_idx}.mp3")

    # تجنّب إعادة التوليد إن كان الملف موجوداً
    if os.path.exists(out_path):
        return out_path

    try:
        tts_obj = gTTS(text=scene["narration"], lang="ar", slow=False)
        tts_obj.save(out_path)
        return out_path
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None


def transcribe_audio(audio_path: str) -> str:
    """يحوّل صوت الطفل إلى نص باستخدام Whisper"""
    if not audio_path:
        return ""

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    # تطبيع مستوى الصوت
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # قص الصمت
    trimmed, _ = librosa.effects.trim(audio, top_db=20)
    if len(trimmed) > 16000 * 0.2:
        audio = trimmed

    proc_path = os.path.join(TEMP_DIR, "child_input.wav")
    sf.write(proc_path, audio, 16000)

    result = asr_model.transcribe(
        proc_path,
        language="ar",
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        fp16=(device == "cuda"),
    )
    return result["text"].strip()


def calc_accuracy(spoken: str, target: str) -> float:
    """نسبة الدقة بين النص المنطوق والنص المستهدف (0–100)"""
    spoken = spoken.strip()
    target = target.strip()
    if not spoken or not target:
        return 0.0
    dist = Levenshtein.distance(spoken, target)
    return max(0.0, (1 - dist / max(len(spoken), len(target))) * 100)


# ═══════════════════════════════════════════════════════════════
# 5. دوال تحكم واجهة Gradio
# ═══════════════════════════════════════════════════════════════

def load_scene(idx: int):
    """يُعيد جميع مكوّنات واجهة المشهد idx"""
    scene = STORY[idx]
    is_last = (idx == TOTAL - 1)

    return (
        make_scene_image(scene),                        # scene_img
        make_char_image(scene),                         # char_img
        f"### 🎭 {scene['character']} يقول:",           # char_lbl
        scene["narration"],                             # narration_box
        f"**العبارة المطلوبة:**\n\n> {scene['child_text']}", # prompt_md
        scene["hint"],                                  # hint_lbl
        f"📖 المشهد {idx + 1} من {TOTAL}",              # progress_lbl
        "",                                             # result_box  (مسح)
        gr.update(visible=False),                       # next_btn
        gr.update(visible=not is_last),                 # next_btn (hidden on last)
        None,                                           # narration_audio (مسح)
    )


def on_listen(idx: int):
    """توليد صوت الشخصية وإعادته للمشغّل"""
    return generate_narration(idx)


def on_check(audio_path, idx: int):
    """فحص نطق الطفل وإعطاء النتيجة"""
    if not audio_path:
        return "⚠️ سجّل صوتك أولاً بالضغط على المايك.", gr.update(visible=False)

    scene   = STORY[idx]
    spoken  = transcribe_audio(audio_path)

    if not spoken:
        return "⚠️ لم يُسمع صوت واضح. حاول مجدداً بصوت أعلى.", gr.update(visible=False)

    score     = calc_accuracy(spoken, scene["child_text"])
    threshold = scene["threshold"]
    is_last   = (idx == TOTAL - 1)

    if score >= threshold:
        if is_last:
            msg = (
                f"🏆 أحسنت يا بطل! نطقك ممتاز ({score:.0f}%)!\n\n"
                f"🎉 لقد أنهيت القصة كاملة! أنت رائع!"
            )
        else:
            msg = (
                f"✅ رائع! نطقك صحيح ({score:.0f}%)!\n"
                f"سمعتك تقول: [{spoken}]\n\n"
                f"اضغط «التالي» للانتقال إلى المشهد القادم."
            )
        return msg, gr.update(visible=not is_last)
    else:
        msg = (
            f"💪 تقريباً! ({score:.0f}%) — حاول مرة أخرى.\n"
            f"سمعتك تقول: [{spoken}]\n"
            f"المطلوب: [{scene['child_text']}]\n"
            f"{scene['hint']}"
        )
        return msg, gr.update(visible=False)


def on_next(idx: int):
    """الانتقال للمشهد التالي وتحديث كل مكوّنات الواجهة"""
    new_idx = min(idx + 1, TOTAL - 1)
    return (new_idx,) + load_scene(new_idx)


# ═══════════════════════════════════════════════════════════════
# 6. بناء واجهة Gradio
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(title="القصة التفاعلية") as demo:

    scene_idx = gr.State(0)   # مؤشر المشهد الحالي

    # ── شريط العنوان ─────────────────────────────────────────
    gr.Markdown("# 📚 القصة التفاعلية — تدريب نطق الأطفال")
    progress_lbl = gr.Markdown(f"📖 المشهد 1 من {TOTAL}")

    with gr.Row():
        # ══ العمود الأيسر: الصور + الاستماع ════════════════
        with gr.Column(scale=1):
            scene_img = gr.Image(
                label="مشهد القصة",
                height=280,
                interactive=False,
            )
            with gr.Row():
                char_img = gr.Image(
                    label="الشخصية",
                    height=160,
                    interactive=False,
                )
                with gr.Column():
                    char_lbl = gr.Markdown("### 🎭 الراوي يقول:")
                    listen_btn = gr.Button("🔊 استمع للشخصية", variant="secondary", size="lg")

            narration_audio = gr.Audio(
                label="صوت الشخصية",
                interactive=False,
            )

        # ══ العمود الأيمن: القراءة + التحقق ════════════════
        with gr.Column(scale=1):
            narration_box = gr.Textbox(
                label="💬 ماذا قالت الشخصية",
                lines=3,
                interactive=False,
            )

            gr.Markdown("---")
            prompt_md = gr.Markdown("**العبارة المطلوبة:**")
            hint_lbl  = gr.Markdown("")

            user_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 سجّل صوتك (اضغط على المايك)",
            )

            check_btn = gr.Button("✅ تحقق من نطقي", variant="primary", size="lg")

            result_box = gr.Textbox(
                label="📊 النتيجة",
                lines=5,
                interactive=False,
            )

            next_btn = gr.Button(
                "➡️ المشهد التالي",
                variant="primary",
                size="lg",
                visible=False,
            )

    # ── ربط الأحداث ──────────────────────────────────────────

    _scene_outputs = [
        scene_img, char_img, char_lbl, narration_box,
        prompt_md, hint_lbl, progress_lbl,
        result_box, next_btn, next_btn,   # next_btn appears twice: visibility + visibility reset
        narration_audio,
    ]

    # تحميل المشهد الأول عند فتح الصفحة
    demo.load(
        fn=lambda: load_scene(0),
        outputs=[
            scene_img, char_img, char_lbl, narration_box,
            prompt_md, hint_lbl, progress_lbl,
            result_box, next_btn, next_btn,
            narration_audio,
        ],
    )

    listen_btn.click(
        fn=on_listen,
        inputs=[scene_idx],
        outputs=[narration_audio],
    )

    check_btn.click(
        fn=on_check,
        inputs=[user_audio, scene_idx],
        outputs=[result_box, next_btn],
    )

    next_btn.click(
        fn=on_next,
        inputs=[scene_idx],
        outputs=[
            scene_idx,
            scene_img, char_img, char_lbl, narration_box,
            prompt_md, hint_lbl, progress_lbl,
            result_box, next_btn, next_btn,
            narration_audio,
        ],
    )


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch()
