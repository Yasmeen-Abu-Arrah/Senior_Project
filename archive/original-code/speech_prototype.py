import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import gradio as gr
import whisper
import torch

# Ensure ffmpeg is on PATH (required by Whisper for audio decoding)
_FFMPEG_BIN = r"C:\Users\hussein younis\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
if _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# 1. التحقق من وجود كرت الشاشة وتوجيه النموذج لاستخدامه
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 جاري تحميل نموذج Whisper على الـ ({device})... يرجى الانتظار")

# نقوم بتحميل النسخة base من المجلد المحلي بدلاً من تحميلها من الإنترنت
_WHISPER_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "whisper_models")
# turbo = large-v3-turbo: 6x faster than large-v3, much better for short single-word clips
# large-v3 hallucinates on short audio — turbo is tuned for exactly this use case
model = whisper.load_model("turbo", device=device, download_root=_WHISPER_MODELS_DIR)
print("✅ تم تحميل النموذج بنجاح وهو جاهز للاستخدام اللحظي!")

# 2. تحديد الكلمة المستهدفة (مثال تجريبي لحالة الطفل خالد "قطار")
TARGET_WORD = "قطار"

# ============================================================
# مرحلة معالجة الصوت: تنقية بسيطة وآمنة قبل تمريرها للنموذج
# ============================================================
WHISPER_SR = 16000  # معدل الترددات الذي يتوقعه Whisper

def preprocess_audio(audio_path: str) -> str:
    """
    Pipeline مبسّط وآمن:
    1. تحميل الصوت وإعادة ضبط معدل الترددات إلى 16kHz
    2. تطبيع مستوى الصوت → تضخيم الصوت الخافت لمستوى ثابت
    3. قص الصمت من الأطراف فقط
    ملاحظة: تم إلغاء Pre-emphasis و Noise Reduction لأنهما كانا
    يشوّهان الإشارة للتسجيلات القصيرة ويسببان هلوسة النموذج.
    """
    # 1. تحميل وإعادة ضبط معدل الترددات
    audio, sr = librosa.load(audio_path, sr=WHISPER_SR, mono=True)

    # 2. تطبيع مستوى الصوت (Peak Normalization إلى 95%)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # 3. قص الصمت من الأطراف فقط (عتبة 20dB)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    if len(audio_trimmed) > WHISPER_SR * 0.2:  # تجاهل إذا أصبح أقصر من 0.2 ثانية
        audio = audio_trimmed

    processed_path = audio_path.replace(".wav", "_processed.wav").replace(".webm", "_processed.wav")
    sf.write(processed_path, audio, WHISPER_SR)
    return processed_path


def is_hallucination(text: str) -> bool:
    """يكشف هلوسة Whisper: النصوص المتكررة التي يولّدها النموذج عند إشارة غير واضحة."""
    words = text.split()
    if len(words) < 5:
        return False
    # إذا تكرّرت أي عبارة ثنائية أكثر من مرتين → هلوسة
    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i + 1]
        if text.count(phrase) > 2:
            return True
    return False


def check_arabic_speech(audio_file_path):
    if not audio_file_path:
        return "⚠️ يرجى الضغط على المايك وتسجيل صوتك أولاً يا بطل!"
    
    try:
        # 3. تطبيق مرحلة معالجة الصوت قبل التمرير للنموذج
        processed_path = preprocess_audio(audio_file_path)

        # 4. تشغيل Whisper بإعدادات محسّنة للعربية وكلمات الأطفال المفردة
        result = model.transcribe(
            processed_path,
            language="ar",
            task="transcribe",
            temperature=0,
            beam_size=3,
            best_of=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.4,
            compression_ratio_threshold=1.8,   # خُفِّض من 2.4 لرصد التكرار مبكراً
            logprob_threshold=-0.8,            # يرفض النتائج منخفضة الثقة جداً
            fp16=(device == "cuda"),
        )
        transcribed_text = result["text"].strip()

        # 5. كشف هلوسة النموذج (نصوص متكررة) قبل المقارنة
        if is_hallucination(transcribed_text):
            return "⚠️ لم يتمكن النموذج من التعرف على الكلمة بوضوح. يرجى التسجيل مجدداً بصوت أعلى وأوضح."

        
        # 4. تنظيف النص المستخرج من علامات الترقيم (مثل النقاط وعلامات الاستفهام)
        clean_text = "".join(c for c in transcribed_text if c.isalnum() or c.isspace()).strip()
        
        # 5. منطق التحليل الطبي والبرمجي ومقارنة النتائج
        if clean_text == TARGET_WORD:
            return f"✅ أداء ممتاز ونطق صحيح 100%! \nالكلمة المنطوقة: [{clean_text}]"
        
        # محاكاة ذكية لرصد خطأ الحذف الفونولوجي (إذا قال الطفل طار بدلاً من قطار)
        elif "طار" in clean_text or "تتار" in clean_text:
            return f"❌ تم رصد خطأ (حذف/إبدال الحرف الأول)! \nالكلمة المنطوقة: [{clean_text}] \nالتوجيه العلاجي: حاول إخراج حرف (القاف) بقوة من الحلق قبل نطق بقية الكلمة."
        
        else:
            return f"❌ النطق غير مطابق للكلمة المستهدفة. \nالكلمة المسموعة: [{clean_text}] \nالمطلوب نطق: [{TARGET_WORD}]"
            
    except Exception as e:
        return f"🚨 حدث خطأ أثناء معالجة الإشارة الصوتية: {str(e)}"

# 6. بناء الواجهة التفاعلية الرسومية عبر Gradio
with gr.Blocks(title="Smart Speech Therapy - AI Audio Unit") as demo:
    gr.Markdown("# 🎤 وحدة فحص النطق الصوتي اللحظي بالذكاء الاصطناعي")
    gr.Markdown(f"### 🎯 التحدي الحالي للطفل: انطق كلمة **[{TARGET_WORD}]** بوضوح")
    
    with gr.Row():
        # أداة المايك التفاعلية لالتقاط الصوت من متصفح الهاتف أو الكمبيوتر
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="اضغط على المايك وابدأ الكلام")
        
    with gr.Row():
        check_btn = gr.Button("تحقق من صحة نطق الطفل 🚀", variant="primary")
        
    with gr.Row():
        output_result = gr.Textbox(label="التحليل الطبي والبرمجي للموجة الصوتية", lines=4)
        
    # ربط الزر بالخوارزمية
    check_btn.click(fn=check_arabic_speech, inputs=audio_input, outputs=output_result)

# 7. تشغيل الواجهة وإتاحة رابط فحص خارجي
if __name__ == "__main__":
    # share=True تمنحك رابطاً عاماً مؤقتاً لتجربة المايك من هاتفك الشخصي مباشرة
    demo.launch(share=True)
