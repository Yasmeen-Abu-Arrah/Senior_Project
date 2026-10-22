import gradio as gr
import librosa
import numpy as np
import torch
import whisper
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
import Levenshtein
import os

# المسار المحلي لمجلد نماذج Whisper
WHISPER_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper_models")

# 1. تحميل نموذج Whisper من الملفات المحلية (بدون اتصال بالإنترنت)
print("جاري تحميل نموذج الذكاء الاصطناعي للتعرف على الصوت...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# استخدام large-v3 للحصول على أعلى دقة في اللغة العربية
# يمكن تغييره إلى: small, medium, large-v3-turbo
asr_model = whisper.load_model("large-v3", device=device, download_root=WHISPER_MODELS_DIR)
print("تم تحميل النموذج بنجاح!")

# 2. دالة حساب التطابق الموجي والبصمة الصوتية (MFCC + DTW)
def compare_audio_waves(user_audio_path, reference_audio_path):
    if not user_audio_path or not reference_audio_path:
        return 0.0
    
    # تحميل الملفين الصوتيين وتوحيد معدل العينات (Sampling Rate)
    y_user, sr_user = librosa.load(user_audio_path, sr=16000)
    y_ref, sr_ref = librosa.load(reference_audio_path, sr=16000)
    
    # استخراج الخصائص الصوتية (MFCCs)
    mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_user, n_mfcc=13).T
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13).T
    
    # تطبيق خوارزمية Dynamic Time Warping لمقارنة السلاسل الزمنية المختلفة الطول
    distance, path = fastdtw(mfcc_user, mfcc_ref, dist=cosine)
    
    # تحويل المسافة إلى نسبة مئوية تقريبية للتطابق (كلما قلّت المسافة زاد التطابق)
    # نقسم على طول المسار لتطبيع النتيجة (Normalization)
    normalized_score = distance / len(path)
    
    # تحويل النتيجة المقياسية إلى نسبة مئوية (تعتمد على تجربة العتبة الصوتية)
    wave_similarity = max(0, min(100, (1 - normalized_score) * 100))
    return wave_similarity

# 3. الدالة الرئيسية للتقييم عبر Gradio
def evaluate_pronunciation(reference_text, reference_audio, user_audio):
    if user_audio is None:
        return "الرجاء تسجيل صوتك أولاً.", "N/A", "N/A", "N/A"
    
    # أ) التعرف على الكلام وتحليله نصياً (على مستوى الحروف)
    try:
        print(f"جاري تحليل الصوت المنطوق: {user_audio}")
        result = asr_model.transcribe(
            user_audio,
            language="ar",
            condition_on_previous_text=False,  # منع الحلقة اللانهائية بين المقاطع
            no_speech_threshold=0.6,           # تجاهل المقاطع الصامتة
            compression_ratio_threshold=1.35,  # تجاهل المخرجات المتكررة تلقائياً
            logprob_threshold=-1.0,            # إلغاء المقاطع ذات الثقة المنخفضة
            temperature=0.0,                   # اختيار حتمي (بلا عشوائية)
            fp16=(device == "cuda"),            # سرعة أعلى على GPU
        )
        spoken_text = result["text"].strip()
    except Exception as e:
        return f"حدث خطأ أثناء معالجة الصوت: {str(e)}", "N/A", "N/A", "N/A"
    
    # تنظيف النصوص للمقارنة الحرفية
    ref_clean = reference_text.strip()
    
    # حساب المسافة الحرفية (Levenshtein Distance) على مستوى الحروف والأصوات المنطوقة
    char_distance = Levenshtein.distance(spoken_text, ref_clean)
    max_len = max(len(spoken_text), len(ref_clean))
    
    if max_len == 0:
        text_accuracy = 100.0
    else:
        text_accuracy = ((max_len - char_distance) / max_len) * 100

    # ب) تحليل البصمة الصوتية (MFCC) في حال توفر ملف الشيخ/المحترف المرجعي
    wave_accuracy_score = 0.0
    wave_report = "لم يتم رفع ملف مرجعي للمقارنة الموجية."
    
    if reference_audio is not None:
        wave_accuracy_score = compare_audio_waves(user_audio, reference_audio)
        wave_report = f"{wave_accuracy_score:.2f}%"
    
    # ج) صياغة التقييم النهائي
    final_score = (text_accuracy * 0.6) + (wave_accuracy_score * 0.4) if reference_audio else text_accuracy
    
    if final_score >= 85:
        verdict = "🟢 ممتاز! نطقك سليم ومخارج الحروف ونبرة الصوت مطابقة بشكل كبير جداً."
    elif final_score >= 65:
        verdict = "🟡 جيد. النطق مفهوم ولكن تحتاج للتركيز أكثر على مخارج الحروف الثقيلة ومحاكاة النبرة المرجعية."
    else:
        verdict = "🔴 تحتاج إلى تحسين. هناك اختلاف ملحوظ في نطق الكلمات أو في الإيقاع البصمي للصوت."

    return spoken_text, f"{text_accuracy:.2f}%", wave_report, f"{final_score:.2f}% \n\n {verdict}"

# 4. بناء واجهة مستخدم Gradio اللائقة
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ نظام تقييم النطق ومخارج الحروف الذكي")
    gr.Markdown("يقوم هذا النظام بتحليل نطقك ومقارنته بنص مستهدف وبصمة صوتية مرجعية (لشيخ أو متحدث محترف) باستخدام الذكاء الاصطناعي وخوارزميات معالجة الإشارات الرقمية.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1️⃣ الإعدادات والمرجع")
            input_text = gr.Textbox(
                label="النص المستهدف قراءته", 
                value="الحمد لله رب العالمين", 
                placeholder="اكتب النص الذي سيقرأه المستخدم هنا..."
            )
            ref_audio = gr.Audio(
                label="صوت القارئ المرجعي (اختياري للمقارنة الموجية)", 
                type="filepath"
            )
            
            gr.Markdown("### 2️⃣ تسجيل صوت المستخدم")
            user_audio = gr.Audio(
                label="سجل صوتك هنا أو قم برفع ملف", 
                type="filepath", 
                sources=["microphone", "upload"]
            )
            
            btn_submit = gr.Button("🔍 ابدأ التحليل والتقييم", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 3️⃣ نتائج التحليل")
            output_spoken = gr.Textbox(label="النص المسموع (الذي نطقتَه بالفعْل)", interactive=False)
            
            with gr.Row():
                output_text_acc = gr.Label(label="دقة النطق الحرفي (Text/Phoneme Accuracy)")
                output_wave_acc = gr.Label(label="تطابق البصمة الصوتية (MFCC/DTW Match)")
                
            output_final = gr.Textbox(label="التقييم النهائي والنتيجة المبدئية", interactive=False)

    # ربط العناصر بالدالة المعالجة
    btn_submit.click(
        fn=evaluate_pronunciation,
        inputs=[input_text, ref_audio, user_audio],
        outputs=[output_spoken, output_text_acc, output_wave_acc, output_final]
    )

# تشغيل التطبيق
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())