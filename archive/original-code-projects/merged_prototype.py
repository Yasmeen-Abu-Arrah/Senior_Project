import os
import re
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_PATH = "./qwen_text_model"

if not os.path.exists(LOCAL_MODEL_PATH) or not os.listdir(LOCAL_MODEL_PATH):
    raise FileNotFoundError("❌ Local text model folder is empty! Please run the download script first.")

print("⏳ Loading Qwen Text Model offline into memory...")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch_dtype,
    device_map=device
)

print(f"✅ Story Generator loaded successfully on {device.upper()}!")

# --- 100% Offline CSS Config pulling from the local file ---
CUSTOM_CSS = """
@font-face {
    font-family: 'LocalAmiri';
    src: url('/file=fonts/Amiri-Regular.ttf') format('truetype');
    font-weight: normal;
    font-style: normal;
}

.arabic-story-container {
    font-family: 'LocalAmiri', serif !important;
    direction: rtl !important;
    text-align: right !important;
    background-color: #fcfbf7;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.storyboard-title {
    font-family: 'LocalAmiri', serif !important;
    font-size: 28px !important;
    font-weight: 700;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 20px;
    border-bottom: 2px solid #f1c40f;
    padding-bottom: 10px;
}

.storyboard-grid {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.storyboard-panel {
    background: #ffffff;
    border-left: 5px solid #4A90E2;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

.panel-header {
    font-size: 14px;
    color: #7f8c8d;
    font-weight: bold;
    margin-bottom: 8px;
}

.panel-text {
    font-family: 'LocalAmiri', serif !important;
    font-size: 24px !important;
    line-height: 2.2 !important;
    color: #2c3e50;
}

.therapy-highlight {
    background-color: #fff2cc;
    border-bottom: 2px dashed #f1c40f;
    color: #d35400;
    font-weight: 700;
    padding: 0 4px;
    border-radius: 4px;
}
"""

# --- RESTORED FUNCTION: Text Generation and Storyboard Mapping ---
def generate_safe_story(target_words, story_theme):
    if not target_words.strip():
        return "<div class='arabic-story-container' style='color:red;'>❌ يرجى إدخال الكلمات المستهدفة أولاً لتضمينها في القصة.</div>"

    # --- ENHANCED SYSTEM PROMPT FOR NATIVE ARABIC SYNTAX & DIACRITICS ---
    system_prompt = (
        "أنت أديب ومؤلف محترف لقصص الأطفال صغار السن، تكتب بلغة عربية فصحى نقية، أصيلة، وخالية تماماً من الركاكة أو الترجمات الحرفية.\n"
        "مهمتك هي كتابة حكاية قصيرة جداً ومبسطة، مقسمة إلى 3 أو 4 فقرات تفصلها أسطر فارغة، وكل فقرة تمثل مشهداً بصرياً واحداً.\n\n"
        "⚠️ قواعد النحو والسياق الصارمة (Strict Linguistic Rules):\n"
        "1. ابدأ الجمل دائماً بتركيب لغوي عربي أصيل (مثل: استيقظَ الطفلُ في الصباحِ الباكرِ، طارَ العصفورُ) وتجنب تماماً الصياغات الركيكة مثل 'في صباح الخير' أو 'هو ذهب'.\n"
        "2. يجب ضبط كافة أواخر الكلمات والحروف بالحركات والتنطيط (التشكيل الكامل) بشكل نحوي سليم مئة بالمئة لتعليم النطق الصحيح.\n"
        "3. استخدم جملًا قصيرة جداً (من 3 إلى 5 كلمات في الجملة الواحدة) لتناسب مهارات النطق والاستيعاب عند الأطفال.\n"
        "4. ركّز حصرياً على دمج الكلمات المستهدفة التي يكتبها المستخدم بشكل طبيعي ومتكرر داخل السرد.\n"
        "5. يجب إحاطة الكلمات المستهدفة المستلمة من المستخدم بقوسين واضحتين في كل مرة تظهر فيها داخل القصة، مثل: (قِطَارٌ)، (عُصْفُورٌ) لتركيز انتباه الطفل بصرياً عليها."
    )

    # We augment the user prompt to give a structural starting clue to the AI
    user_prompt = (
        f"اكتب قصة للأطفال بناءً على المعايير التالية:\n"
        f"الموضوع العام: {story_theme}\n"
        f"الكلمات المستهدفة المطلوب تكرارها ووضعها بين قوسين مع التشكيل: {target_words}\n\n"
        f"توجيه لبداية السرد: ابدأ القصة مباشرة بعبارة عربية بليغة ومبسطة."
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=800,
                temperature=0.3, # 📉 Dropped temperature to 0.3 to maximize grammatical compliance and prevent hallucinations
                top_p=0.85
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Split text generations into paragraph structures
        paragraphs = [p.strip() for p in re.split(r'\n+', response) if p.strip()]
        
        html_output = f"<div class='arabic-story-container'>"
        html_output += f"<div class='storyboard-title'>📖 حكاية البطل لتدريب النطق</div>"
        html_output += f"<div class='storyboard-grid'>"
        
        for idx, para in enumerate(paragraphs, 1):
            # Highlight therapeutic words wrapped in parentheses
            highlighted_para = re.sub(
                r'\((.*?)\)', 
                r"<span class='therapy-highlight'>(\1)</span>", 
                para
            )
            
            html_output += f"""
            <div class='storyboard-panel'>
                <div class='panel-header'>المشهد اللفظي {idx} 🎬</div>
                <div class='panel-text'>{highlighted_para}</div>
            </div>
            """
            
        html_output += "</div></div>"
        return html_output

    except Exception as e:
        return f"<div class='arabic-story-container' style='color:red;'>❌ حدث خطأ أثناء توليد القصة: {str(e)}</div>"

# --- Advanced Professional Interface ---
with gr.Blocks(title="مؤلف قصص الأطفال الذكي", css=CUSTOM_CSS) as demo:
    gr.Markdown("# 📚 مؤلف قصص الأطفال - Offline Mode")
    
    with gr.Row():
        with gr.Column(scale=1):
            words_input = gr.Textbox(label="🔑 الكلمات المستهدفة", placeholder="مثال: قطار")
            theme_input = gr.Dropdown(
                choices=[
                    "يوم جميل وممتع في الحديقة العامة", 
                    "اللعب بالألعاب الجميلة في غرفتي",
                    "ذهاب الطفل إلى المدرسة صباحاً ومشاهدة الطريق",
                    "مساعدة الأم في ترتيب المكعبات الملونة"
                ],
                value="يوم جميل وممتع في الحديقة العامة",
                label="🎭 السيناريو"
            )
            generate_btn = gr.Button("توليد لوحة العمل 🚀", variant="primary")
            
        with gr.Column(scale=2):
            story_output = gr.HTML(
                value="<div class='arabic-story-container' style='text-align:center;'>بانتظار توليد لوحة العمل...</div>"
            )

    generate_btn.click(fn=generate_safe_story, inputs=[words_input, theme_input], outputs=story_output)

# --- 🚀 CRITICAL CONFIGURATION FOR DYNAMIC LOCAL CSS AND PATH SECURITY ---
if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.path.abspath("./fonts")])
