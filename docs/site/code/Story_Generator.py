import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def generate_safe_story(target_words, story_theme):
    if not target_words.strip():
        return "❌ يرجى إدخال الكلمات المستهدفة أولاً لتضمينها في القصة."

    # تعديل الـ System Prompt ليركز فقط على السرد البسيط اليومي للأطفال الصغار جداً
    system_prompt = (
        "أنت كاتب محترف لقصص الأطفال الصغار جداً، وتكتب باللغة العربية الفصحى المبسطة والمشكولة بالكامل (بالحركات).\n"
        "مهمتك هي كتابة قصة قصيرة جداً، بسيطة للغاية، وتدور حول أحداث يومية عادية ومبهجة يفهمها الطفل الصغير (مثل اللعب، التنزّه، أو مشاهدة الأشياء).\n"
        "شروط صارمة:\n"
        "1. تجنب تماماً طرح أي مفاهيم أخلاقية، دينية، أو فلسفية عميقة أو معقدة قد لا تناسب سنهم الصغير.\n"
        "2. يجب تماماً استبعاد أي محتوى للبالغين، أو عنف، أو أفكار مخيفة، أو موضوعات خارجة عن الثقافة العربية والبيئة العامة.\n"
        "3. ركّز فقط على دمج الكلمات المستهدفة التي يكتبها المستخدم بشكل متكرر وطبيعي داخل السرد.\n"
        "4. يجب وضع الكلمات المستهدفة بين قوسين واضحتين مثل: (عصفور)، (قطار) في كل مرة تظهر فيها داخل القصة لتسهيل قراءتها والتركيز عليها."
    )

    user_prompt = f"السياق العام للقصة: {story_theme}\nالكلمات المستهدفة التي يجب وضعها بين قوسين وتكرارها: {target_words}"

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=800,
                temperature=0.6, # تقليل الحرارة لجعل الموديل ملتزماً حرفياً بالبساطة وعدم التفرع
                top_p=0.9
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    except Exception as e:
        return f"❌ حدث خطأ أثناء توليد القصة: {str(e)}"

# --- واجهة Gradio المعدلة بمواضيع يومية بسيطة تناسب الأطفال الصغار ---
with gr.Blocks(title="مؤلف قصص الأطفال المبسطة أوفلاين") as demo:
    gr.Markdown("# 📚 مؤلف قصص الأطفال الصغار لتدريب النطق (Offline)")
    gr.Markdown(
        "يقوم هذا النظام بتوليد قصص يومية بسيطة ومشكولة بالحركات لتناسب الأطفال الصغار، "
        "مع التركيز التام على الكلمات المستهدفة لتدريب النطق وبشكل آمن تماماً وثقافي مألوف."
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🛠️ إعدادات القصة")
            words_input = gr.Textbox(
                label="🔑 الكلمات المستهدفة للنطق (اكتب الكلمات مفصولة بفاصلة)",
                placeholder="مثال: عصفور، قطار",
                lines=2
            )
            
            theme_input = gr.Dropdown(
                choices=[
                    "يوم جميل وممتع في الحديقة العامة",
                    "اللعب بالألعاب الجميلة في غرفتي",
                    "ذهاب الطفل إلى المدرسة صباحاً ومشاهدة الطريق",
                    "مساعدة الأم في ترتيب المكعبات الملونة"
                ],
                value="يوم جميل وممتع في الحديقة العامة",
                label="🎭 سياق القصة اليومي"
            )
            
            generate_btn = gr.Button("توليد القصة البسيطة محلياً", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 📖 نص القصة")
            story_output = gr.Textbox(
                label="نص القصة المشكول والمبسط",
                lines=15,
                placeholder="ستظهر القصة اليومية البسيطة هنا..."
            )

    generate_btn.click(
        fn=generate_safe_story,
        inputs=[words_input, theme_input],
        outputs=story_output
    )

if __name__ == "__main__":
    demo.launch()
