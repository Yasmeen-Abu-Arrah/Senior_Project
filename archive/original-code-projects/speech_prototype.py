import gradio as gr
import whisper
import torch

# 1. التحقق من وجود كرت الشاشة وتوجيه النموذج لاستخدامه
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 جاري تحميل نموذج Whisper على الـ ({device})... يرجى الانتظار")

# نقوم بتحميل النسخة base لتوازن مثالي بين السرعة ودقة الحروف العربية
model = whisper.load_model("base", device=device)
print("✅ تم تحميل النموذج بنجاح وهو جاهز للاستخدام اللحظي!")

# 2. تحديد الكلمة المستهدفة (مثال تجريبي لحالة الطفل خالد "قطار")
TARGET_WORD = "قطار"

def check_arabic_speech(audio_file_path):
    if not audio_file_path:
        return "⚠️ يرجى الضغط على المايك وتسجيل صوتك أولاً يا بطل!"
    
    try:
        # 3. تشغيل خوارزمية الذكاء الاصطناعي مع إجبارها على فك شفرة اللغة العربية الفصحى واللهجات
        # نحدد الخيار language="ar" لمنع النظام من كتابة النطق بحروف لاتينية (Transliteration)
        result = model.transcribe(audio_file_path, language="ar", task="transcribe")
        transcribed_text = result["text"].strip()
        
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
