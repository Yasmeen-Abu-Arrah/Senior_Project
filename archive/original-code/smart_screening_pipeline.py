import gradio as gr
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 1. SKLEARN DECISION TREE  (ML Quick Screening)
# =====================================================================
X_train = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 1, 0],
    [2, 0, 0, 1],
    [1, 1, 1, 1],
    [2, 0, 1, 0],
])
y_train = np.array([1, 0, 2, 3, 4, 2])

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

diagnosis_labels = {
    0: "نمو طبيعي / سليم (Normal Development)",
    1: "اشتباه تأخر لغوي (Language Delay)",
    2: "اشتباه اضطراب طلاقة الكلام / تلعثم (Fluency Disorder / Stuttering)",
    3: "اضطراب نطق وفونولوجيا / لدغات (Articulation Disorder)",
    4: "اضطراب لغوي ونطقي مشترك (Mixed Speech-Language Delay)",
}

# =====================================================================
# 2. MEDIAPIPE FACE LANDMARKER PIPELINE  (Lip Tracking)
# =====================================================================
_MODEL_PATH = r"d:\Communication-Disorders\models\face_landmarker.task"
_face_landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(_face_landmarker_options)

UPPER_LIP    = 13
LOWER_LIP    = 14
LEFT_CORNER  = 61
RIGHT_CORNER = 291


def process_video_frame(frame):
    """Activates Computer Vision to track lip distance and symmetry during exercises."""
    if frame is None:
        return None, "لا يوجد إشارة كاميرا"

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = face_landmarker.detect(mp_image)

    status_text = "يرجى التحدث أمام الكاميرا لبدء التتبع حركي..."
    h, w, _     = frame.shape

    if results.face_landmarks:
        face_lms = results.face_landmarks[0]

        p_upper = face_lms[UPPER_LIP]
        p_lower = face_lms[LOWER_LIP]
        p_left  = face_lms[LEFT_CORNER]
        p_right = face_lms[RIGHT_CORNER]

        vertical_gap     = np.abs(p_upper.y - p_lower.y) * h
        horizontal_width = np.abs(p_left.x  - p_right.x) * w

        for idx in [UPPER_LIP, LOWER_LIP, LEFT_CORNER, RIGHT_CORNER]:
            pt = face_lms[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        status_text = (
            f"مسافة فتح الفم الرأسية: {vertical_gap:.1f}px  |  "
            f"العرض الأفقي: {horizontal_width:.1f}px"
        )
        label = "Mouth Open (تحدث/نطق)" if vertical_gap > 25 else "Mouth Closed (ساكن)"
        color = (0, 255, 0)              if vertical_gap > 25 else (0, 0, 255)
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame, status_text


# =====================================================================
# 3. ML QUICK SCREENING LOGIC
# =====================================================================
def run_screening_and_setup_pipeline(age_group, vocab_delay, stutter_repetition, articulation_errors):
    age_idx    = {"سنتين إلى 3 سنوات": 0, "3 إلى 4 سنوات": 1, "4 إلى 6 سنوات": 2}[age_group]
    vocab_val  = 1 if vocab_delay          == "نعم، مفرداته قليلة جداً مقارنة بأقرانه" else 0
    stutter_val = 1 if stutter_repetition  == "نعم، يكرر المقاطع بشكل ملحوظ وبجهد"    else 0
    art_val    = 1 if articulation_errors  == "نعم، يستبدل أو يحذف بعض الحروف"        else 0

    pred_id = clf.predict(np.array([[age_idx, vocab_val, stutter_val, art_val]]))[0]
    final_diagnosis = diagnosis_labels[pred_id]

    recs = {
        1: ["يوصى بـ: مسار بناء الحصيلة اللغوية واللغة الاستقبالية.",
            "تفعيل ألعاب تسمية المجموعات الضمنية."],
        2: ["يوصى بـ: مسار تشكيل الطلاقة اللفظية.",
            "إجراء للنظام: تم حظر مؤقتات الوقت وعناصر الضغط بالألعاب لتقليل التوتر."],
        3: ["يوصى بـ: تمارين تتبع مخارج الحروف الفونولوجية.",
            "إجراء للنظام: توجيه كاميرا MediaPipe للتحقق من زوايا حركة الشفاه."],
    }.get(pred_id, ["المؤشرات طبيعية.", "يُتاح له الوصول لكافة الألعاب التعليمية والقصص."])

    formatted_recs = "\n".join([f"- {r}" for r in recs])
    return (
        f"### النتيجة التشخيصية الذكية:\n**{final_diagnosis}**",
        f"### خطة العمل والتحكم التلقائي بالنظام:\n{formatted_recs}",
    )


# =====================================================================
# 4. CLINICAL SCORING ENGINE  (PediatricSpeechDiagnostics — Rule-Based)
# =====================================================================

def _evaluate_clinical_tree(lq, cap, ssi_score, organic):
    """
    Deterministic rule-based decision tree:
      - Al-Rifai Language Quotient (LQ) thresholds
      - Abu Hasiba CAP threshold (75%)
      - SSI-4 severity bands
      - DSM-5 organic exclusion gate
    """
    if organic == "نعم — يوجد سبب عضوي/بنيوي":
        return {
            "Diagnosis": "اشتباه عجز لغوي-نطقي ثانوي (عضوي)",
            "Findings":  ["سبب عضوي/بنيوي مؤكد — توقف التقييم الآلي فوراً"],
            "Action":    "تحويل إلزامي إلى طب الأذن والأنف والحنجرة أو طب الأطفال العصبي",
            "Status":    "🔴 خطر مرتفع / إحالة طبية عاجلة",
            "organic":   True,
        }

    findings, tracks = [], []

    if lq < 70:
        findings.append("تأخر لغوي حاد (LQ < 70%)")
        tracks.append("تأهيل لغوي مكثف")
    elif lq < 80:
        findings.append("تأخر لغوي خفيف إلى متوسط (LQ 70–79%)")
        tracks.append("تدريب مستهدف على اللغة الاستقبالية والتعبيرية")
    elif lq < 90:
        findings.append("تأخر لغوي حدي (LQ 80–89%)")
        tracks.append("دليل التحفيز المنزلي ومتابعة فعّالة")
    else:
        findings.append("أداء لغوي طبيعي (LQ ≥ 90%)")

    if cap < 75:
        findings.append("اضطراب أصوات الكلام / SSD (CAP < 75%)")
        tracks.append("تمارين نطق مُلعَّبة بالذكاء الاصطناعي — خرائط فونولوجية")
    else:
        findings.append("كفاءة نطقية مثالية (CAP ≥ 75%)")

    if ssi_score >= 25:
        findings.append("تلعثم حاد (SSI-4 ≥ 25)")
        tracks.append("تشكيل الطلاقة وتمارين التنفس المنتظم")
    elif ssi_score >= 11:
        findings.append("تلعثم متوسط (SSI-4: 11–24)")
        tracks.append("مهام الكلام المنضبط وتغذية سمعية راجعة")
    elif ssi_score >= 1:
        findings.append("تلعثم خفيف (SSI-4: 1–10)")
        tracks.append("تدريب على تقليل الحساسية وبناء الثقة التعبيرية")
    else:
        findings.append("أنماط كلام طلقة (SSI-4 = 0)")

    has_issue = len(tracks) > 0
    diag = ("ملف لغوي-نطقي مختلط" if len(tracks) > 1
            else "اضطراب مفرد"     if len(tracks) == 1
            else "ملف نمو طبيعي")

    return {
        "Diagnosis": diag,
        "Findings":  findings,
        "Action":    ("تفعيل بروتوكولات العلاج: " + " | ".join(tracks)
                      if tracks else "تسجيل في متابعة النمو القياسية"),
        "Status":    "🟠 مسار علاجي سريري مُفعَّل" if has_issue else "🟢 مُخلَّص / متابعة معيارية",
        "organic":   False,
    }


# =====================================================================
# 5. VISUALISATION — GAUGES + BAR CHART + RADAR
# =====================================================================

def _draw_gauge(ax, value, max_val, title, color_zones, unit=""):
    """Draw a labelled semicircle gauge on the given Axes."""
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.55, 1.55)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")

    for z_min, z_max, color in color_zones:
        t_start = np.pi * (1.0 - z_min / max_val)
        t_end   = np.pi * (1.0 - z_max / max_val)
        t = np.linspace(t_start, t_end, 120)
        r_out, r_in = 1.15, 0.72
        xs = np.concatenate([r_out * np.cos(t),  r_in * np.cos(t[::-1])])
        ys = np.concatenate([r_out * np.sin(t),  r_in * np.sin(t[::-1])])
        ax.fill(xs, ys, color=color, alpha=0.90, zorder=1)

    t_full = np.linspace(np.pi, 0, 200)
    for r in (1.15, 0.72):
        ax.plot(r * np.cos(t_full), r * np.sin(t_full), color="#2d3561", lw=1.2, zorder=2)
    ax.plot([-1.15, -0.72], [0, 0], color="#2d3561", lw=1.2, zorder=2)
    ax.plot([ 0.72,  1.15], [0, 0], color="#2d3561", lw=1.2, zorder=2)

    norm_val = min(max(value / max_val, 0.0), 1.0)
    angle    = np.pi * (1.0 - norm_val)
    nx, ny   = 0.92 * np.cos(angle), 0.92 * np.sin(angle)
    ax.plot([0, nx], [0, ny], color="white", lw=3.0, zorder=4, solid_capstyle="round")
    ax.plot(0, 0, "o", color="white", markersize=9, zorder=5)

    ax.text(0, -0.28, f"{value:.1f}{unit}", ha="center", va="center",
            fontsize=15, fontweight="bold", color="white", zorder=6)
    ax.text(0, 1.45, title, ha="center", va="top",
            fontsize=9.5, color="#a0c4ff", fontweight="bold", zorder=6,
            multialignment="center")
    ax.text(-1.35, -0.1, "0",           ha="center", fontsize=8, color="#888", zorder=6)
    ax.text( 1.35, -0.1, str(int(max_val)), ha="center", fontsize=8, color="#888", zorder=6)


def create_clinical_plots(lq, cap, ssi_score):
    """Returns a matplotlib Figure: 3 gauges (top) + bar chart + radar (bottom)."""
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── Gauge 1: LQ
    ax_lq = fig.add_subplot(gs[0, 0])
    _draw_gauge(ax_lq, lq, 130,
                "نسبة اللغة LQ\n(الرفاعي)",
                [(0, 70, "#e74c3c"), (70, 80, "#e67e22"),
                 (80, 90, "#f1c40f"), (90, 130, "#2ecc71")],
                unit="%")

    # ── Gauge 2: CAP
    ax_cap = fig.add_subplot(gs[0, 1])
    _draw_gauge(ax_cap, cap, 100,
                "كفاءة النطق CAP\n(أبو حسيبة)",
                [(0, 75, "#e74c3c"), (75, 100, "#2ecc71")],
                unit="%")

    # ── Gauge 3: SSI-4
    ax_ssi = fig.add_subplot(gs[0, 2])
    _draw_gauge(ax_ssi, ssi_score, 52,
                "شدة التلعثم\nSSI-4",
                [(0,  1,  "#2ecc71"),
                 (1,  11, "#f1c40f"),
                 (11, 25, "#e67e22"),
                 (25, 52, "#e74c3c")],
                unit="")

    # ── Bar Chart: scores vs. reference thresholds
    ax_bar = fig.add_subplot(gs[1, :2])
    ax_bar.set_facecolor("#16213e")

    categories = [
        "LQ — الرفاعي\n(طبيعي ≥ 90%)",
        "CAP — أبو حسيبة\n(طبيعي ≥ 75%)",
        "SSI-4\n(خطر ≥ 11)",
    ]
    values     = [lq, cap, ssi_score]
    thresholds = [90, 75, 11]
    bar_colors = [
        "#2ecc71" if lq  >= 90       else "#e74c3c",
        "#2ecc71" if cap >= 75       else "#e74c3c",
        "#2ecc71" if ssi_score < 11  else "#e74c3c",
    ]

    x_pos = np.arange(len(categories))
    bars  = ax_bar.bar(x_pos, values, color=bar_colors, alpha=0.85,
                       width=0.45, edgecolor="white", linewidth=0.6)

    top = max(max(values) * 1.30, 115)
    for bar, thresh in zip(bars, thresholds):
        ax_bar.hlines(thresh,
                      bar.get_x() - 0.08, bar.get_x() + bar.get_width() + 0.08,
                      colors="white", linestyles="--", linewidth=1.5, alpha=0.65)
        ax_bar.text(bar.get_x() + bar.get_width() / 2, thresh + top * 0.025,
                    f"حد = {thresh}", ha="center", color="#ffffffcc", fontsize=9)

    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.015,
                    f"{val:.1f}", ha="center", color="white",
                    fontsize=11, fontweight="bold")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(categories, color="#a0c4ff", fontsize=9)
    ax_bar.set_ylim(0, top)
    ax_bar.tick_params(axis="y", colors="#a0c4ff")
    for spine in ax_bar.spines.values():
        spine.set_color("#2d3561")
    ax_bar.set_title("مقارنة الدرجات بالحدود المرجعية",
                     color="#a0c4ff", fontsize=12, fontweight="bold", pad=10)

    ok_p  = mpatches.Patch(color="#2ecc71", alpha=0.85, label="ضمن المعدل الطبيعي")
    bad_p = mpatches.Patch(color="#e74c3c", alpha=0.85, label="خارج المعدل الطبيعي")
    ax_bar.legend(handles=[ok_p, bad_p], loc="upper right",
                  fontsize=9, facecolor="#1a1a2e", labelcolor="white", framealpha=0.75)

    # ── Radar Chart
    ax_r = fig.add_subplot(gs[1, 2], polar=True)
    ax_r.set_facecolor("#16213e")
    ax_r.patch.set_facecolor("#16213e")

    lq_n  = min(lq / 130, 1.0)
    cap_n = min(cap / 100, 1.0)
    ssi_n = 1.0 - min(ssi_score / 52, 1.0)

    N      = 3
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    pv     = [lq_n, cap_n, ssi_n, lq_n]
    rv     = [0.85, 0.85, 0.85, 0.85]

    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.plot(angles, pv, "o-", lw=2.2, color="#4cc9f0", zorder=3)
    ax_r.fill(angles, pv, alpha=0.28, color="#4cc9f0", zorder=2)
    ax_r.plot(angles, rv, "--", lw=1.4, color="#f1c40f", alpha=0.65, zorder=3)
    ax_r.fill(angles, rv, alpha=0.06, color="#f1c40f", zorder=1)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(["اللغة\n(LQ)", "النطق\n(CAP)", "الطلاقة\n(SSI عكسي)"],
                          color="#a0c4ff", fontsize=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["25%", "50%", "75%", "100%"], color="#555", fontsize=7)
    ax_r.grid(color="#2d3561", linestyle="--", alpha=0.5)
    ax_r.set_title("مخطط الملف الشامل",
                   color="#a0c4ff", fontsize=11, fontweight="bold", pad=14)

    ref_p = mpatches.Patch(color="#f1c40f", alpha=0.6,  label="المستوى المرجعي (85%)")
    act_p = mpatches.Patch(color="#4cc9f0", alpha=0.45, label="أداء الطفل")
    ax_r.legend(handles=[ref_p, act_p], loc="lower right",
                fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.7)

    fig.suptitle(
        "لوحة المؤشرات العيادية الشاملة  |  Clinical Indicators Dashboard",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig


# =====================================================================
# 6. CLINICAL ASSESSMENT ENTRY POINT
# =====================================================================

def run_clinical_assessment(ca_months, la_months, total_sounds, correct_sounds,
                             f_score, d_score, p_score, organic):
    try:
        lq = (float(la_months) / float(ca_months)) * 100 if float(ca_months) > 0 else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        lq = 0.0

    try:
        cap = (float(correct_sounds) / float(total_sounds)) * 100 if float(total_sounds) > 0 else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        cap = 0.0

    try:
        ssi_score = int(f_score) + int(d_score) + int(p_score)
    except (ValueError, TypeError):
        ssi_score = 0

    lq        = max(0.0, min(lq,  200.0))
    cap       = max(0.0, min(cap, 100.0))
    ssi_score = max(0,   min(ssi_score, 52))

    verdict = _evaluate_clinical_tree(lq, cap, ssi_score, organic)

    lq_label  = ("حاد" if lq < 70 else "خفيف–متوسط" if lq < 80 else "حدي" if lq < 90 else "طبيعي")
    cap_label = "اضطراب نطق" if cap < 75 else "طبيعي"
    ssi_label = ("حاد" if ssi_score >= 25 else "متوسط" if ssi_score >= 11
                 else "خفيف" if ssi_score >= 1 else "طبيعي")

    findings_md = "\n".join([f"  \u2022 {f}" for f in verdict.get("Findings", [])])

    report = f"""\
## تقرير التقييم العيادي الشامل

---
### المقاييس الخام المحسوبة

| المقياس | الصيغة | القيمة | التفسير |
|--------|--------|--------|---------|
| نسبة اللغة **LQ** (الرفاعي) | LA ÷ CA × 100 | **{lq:.1f}%** | {lq_label} |
| كفاءة النطق **CAP** (أبو حسيبة) | صحيح ÷ إجمالي × 100 | **{cap:.1f}%** | {cap_label} |
| شدة التلعثم **SSI-4** | تكرار + مدة + مصاحبات | **{ssi_score}** | {ssi_label} |

---
### حالة التصنيف
**{verdict["Status"]}**

### التشخيص الأولي
**{verdict["Diagnosis"]}**

---
### المصفوفة التشخيصية التفصيلية
{findings_md}

---
### بروتوكول العمل المستهدف
> {verdict["Action"]}
"""

    fig = create_clinical_plots(lq, cap, ssi_score)
    return report, fig


# =====================================================================
# 7. INTERACTIVE LANGUAGE AGE TASK WIZARD
# =====================================================================
# Each task: age (months), type, title, instruction, stimulus HTML, criteria
_LANGUAGE_TASKS = [
    {
        "age": 12, "task_type": "naming",
        "title": "المهمة 1 — تسمية الأشخاص المألوفين",
        "instruction": "اعرض الصورة أمام الطفل وانتظر أو اسأله: 'من هذا؟'",
        "stimulus": "<div style='font-size:110px;text-align:center;padding:16px;letter-spacing:24px'>👩 👨</div><div class='task-prompt'>من هذا؟</div>",
        "criteria": "نطق كلمتين هادفتين على الأقل (مثل: ماما / بابا)"
    },
    {
        "age": 18, "task_type": "naming",
        "title": "المهمة 2 — تسمية 10 أشياء شائعة",
        "instruction": "اعرض كل رمز على حدة واسأل: 'ما هذا؟' — احسب عدد الإجابات الصحيحة",
        "stimulus": "<div style='font-size:72px;text-align:center;padding:12px;line-height:1.5'>🍎 🚗 🐱 🐕 🥛<br>🎈 🏠 ✏️ 👟 🌳</div><div class='task-prompt'>ما هذا؟ (لكل صورة)</div>",
        "criteria": "تسمية 6 أشياء صحيحة من أصل 10 على الأقل"
    },
    {
        "age": 24, "task_type": "description",
        "title": "المهمة 3 — جمل من كلمتين + وصف",
        "instruction": "اعرض الرموز. اسأل الطفل: 'ماذا يفعل؟' أو 'صف ما تراه'",
        "stimulus": "<div style='font-size:90px;text-align:center;padding:16px'>🐈&nbsp;🏃</div><div class='task-prompt'>ماذا يفعل القط؟</div><div class='task-hint'>يُتوقع: 'قط يجري' أو 'قطة تتحرك'</div>",
        "criteria": "استخدام جملة من كلمتين على الأقل (فاعل + فعل أو اسم + صفة)"
    },
    {
        "age": 30, "task_type": "comprehension",
        "title": "المهمة 4 — الأسئلة والفهم",
        "instruction": "اسأل الطفل الأسئلة الموضحة. لاحظ هل يُجيب بشكل مناسب.",
        "stimulus": "<div style='font-size:80px;text-align:center;padding:12px'>🎾 &nbsp; 🛏️ &nbsp; 🪑</div><div class='task-prompt'>وين الكرة؟ 🎾</div><div class='task-prompt'>وين السرير؟ 🛏️</div><div class='task-hint'>اسأله أيضاً: 'إيش هذا؟' مشيراً لأشياء حوله</div>",
        "criteria": "الإجابة الصحيحة على سؤالين من ثلاثة على الأقل"
    },
    {
        "age": 36, "task_type": "comprehension",
        "title": "المهمة 5 — تنفيذ تعليمات من خطوتين",
        "instruction": "أعطِ الطفل التعليمة التالية مرة واحدة فقط — لا تكررها",
        "stimulus": "<div style='font-size:80px;text-align:center;padding:12px'>📚 &nbsp; 🪑</div><div class='task-prompt'>'خذ الكتاب وضعه على الكرسي'</div><div class='task-prompt'>'افتح الباب وتعال هنا'</div><div class='task-hint'>لاحظ هل يُنفذ الخطوتين بالترتيب الصحيح</div>",
        "criteria": "تنفيذ تعليمتين من خطوتين بترتيب صحيح"
    },
    {
        "age": 42, "task_type": "story",
        "title": "المهمة 6 — سرد قصة قصيرة من صور",
        "instruction": "اعرض الصور بالترتيب. اسأل الطفل: 'احكِ لي ما حدث'",
        "stimulus": "<div style='font-size:72px;text-align:center;padding:12px'>😴 ➡️ 🌅 ➡️ 🍳 ➡️ 🎒</div><div class='task-prompt'>احكِ لي هذه القصة</div><div class='task-hint'>الولد نائم ← صحى ← أكل الفطور ← ذهب للمدرسة</div>",
        "criteria": "ذكر حدثين متسلسلين مع استخدام ضمائر صحيحة"
    },
    {
        "age": 48, "task_type": "description",
        "title": "المهمة 7 — وصف الأحداث الماضية",
        "instruction": "اسأل الطفل عن يومه أو أنشطته السابقة",
        "stimulus": "<div style='font-size:80px;text-align:center;padding:12px'>📅 &nbsp; 🎒 &nbsp; 🎮</div><div class='task-prompt'>'ماذا فعلتَ أمس؟'</div><div class='task-prompt'>'أخبرني عن آخر مرة لعبتَ فيها'</div><div class='task-hint'>يُتوقع جمل كاملة من 4–5 كلمات تصف حدثاً ماضياً</div>",
        "criteria": "جمل من 4 كلمات فأكثر تصف حدثاً ماضياً بشكل مترابط"
    },
    {
        "age": 54, "task_type": "story",
        "title": "المهمة 8 — ترتيب وسرد قصة متسلسلة",
        "instruction": "اعرض الصور مبعثرة. اطلب من الطفل ترتيبها ثم سرد القصة",
        "stimulus": "<div style='font-size:72px;text-align:center;padding:12px'>🌱 ➡️ 🌿 ➡️ 🌳 ➡️ 🍎</div><div class='task-prompt'>رتّب هذه الصور واحكِ القصة</div><div class='task-hint'>البذرة ← النبتة ← الشجرة ← التفاحة<br>يُتوقع روابط: 'أول / بعدين / في الآخر / لأن'</div>",
        "criteria": "ترتيب صحيح + رابط زمني أو سببي واحد على الأقل"
    },
    {
        "age": 60, "task_type": "comprehension",
        "title": "المهمة 9 — فهم المفاهيم المجردة",
        "instruction": "اسأل الطفل الأسئلة المجردة الموضحة. لاحظ عمق الإجابة.",
        "stimulus": "<div style='font-size:78px;text-align:center;padding:12px'>🌙 &nbsp; 🔥 &nbsp; 🤕</div><div class='task-prompt'>❓ 'ليش نَنام في الليل؟'</div><div class='task-prompt'>❓ 'ماذا تفعل لو جرحتَ إصبعك؟'</div><div class='task-hint'>يُتوقع إجابة منطقية بجملة كاملة</div>",
        "criteria": "الإجابة المنطقية على سؤالين بجملة كاملة لكل منهما"
    },
    {
        "age": 72, "task_type": "conversation",
        "title": "المهمة 10 — المحادثة المتبادلة المعقدة",
        "instruction": "ابدأ محادثة مع الطفل حول موضوع يُحبه. لاحظ دوره في الحوار.",
        "stimulus": "<div style='font-size:78px;text-align:center;padding:12px'>💬 &nbsp; 🎯 &nbsp; 🦸</div><div class='task-prompt'>'أخبرني عن أحلى شيء حصل معك'</div><div class='task-prompt'>'لو كنتَ بطلاً خارقاً — ماذا ستفعل؟'</div><div class='task-hint'>لاحظ: تسلسل الأفكار، الروابط، الاستجابة للأسئلة المتابِعة</div>",
        "criteria": "محادثة متبادلة من 3+ أدوار مع جمل تعبيرية معقدة وتسلسل منطقي"
    },
]

_TASK_TYPE_LABELS = {
    "naming":        "🏷️ تسمية",
    "description":   "🖊️ وصف",
    "comprehension": "🧠 فهم",
    "story":         "📖 سرد",
    "reading":       "📝 قراءة",
    "conversation":  "💬 محادثة",
}


def _get_task_card_html(idx):
    """Return the styled HTML card for the given task index."""
    if idx >= len(_LANGUAGE_TASKS):
        return (
            "<div class='task-card' style='border-color:#2ecc71'>"
            "<div style='font-size:90px;text-align:center'>🎉</div>"
            "<div class='task-prompt' style='color:#2ecc71'>اكتمل التقييم!</div>"
            "<div class='task-hint'>تم تسجيل جميع المهام — يمكنك الآن رؤية العمر اللغوي المحتسَب أدناه</div>"
            "</div>"
        )
    t = _LANGUAGE_TASKS[idx]
    badge = _TASK_TYPE_LABELS.get(t["task_type"], "")
    return (
        f"<div class='task-card'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:10px'>"
        f"<span style='background:#2d3561;color:#a0c4ff;padding:4px 14px;border-radius:20px;font-size:13px'>{badge}</span>"
        f"<span style='color:#888;font-size:13px'>المعيار العمري: {t['age']} شهراً</span>"
        f"</div>"
        f"<h3 style='color:#a0c4ff;margin:6px 0 14px'>{t['title']}</h3>"
        f"{t['stimulus']}"
        f"<hr style='border-color:#2d3561;margin:14px 0'>"
        f"<p style='color:#e0e0e0;font-size:15px;text-align:right'><b>📋 التعليمات:</b> {t['instruction']}</p>"
        f"<p style='color:#f1c40f;font-size:14px;text-align:right;margin-top:6px'>✔ معيار النجاح: {t['criteria']}</p>"
        f"</div>"
    )


def _get_progress_html(idx, results):
    n = len(_LANGUAGE_TASKS)
    done   = len([r for r in results if r is not None])
    passes = sum(1 for r in results if r is True)
    bar = "█" * done + "░" * (n - done)
    if idx >= n:
        return f"✅ **اكتملت جميع المهام** ({done}/{n}) — نجح في: **{passes}** مهمة"
    return f"📊 `{bar}` ({done}/{n}) — المهمة الحالية: **{idx + 1}**"


def compute_la_from_tasks(results):
    """Returns (language_age_months, markdown_message) from task pass/fail list."""
    best_age = 0
    for i, passed in enumerate(results):
        if passed and i < len(_LANGUAGE_TASKS):
            best_age = max(best_age, _LANGUAGE_TASKS[i]["age"])
    msg = f"✅ العمر اللغوي المُحتسَب من المهام: **{best_age} شهراً**"
    return float(best_age), msg


# =====================================================================
# 8. ABU HASIBA ARTICULATION TEST — DIGITIZED ITEMS
# =====================================================================
# Each entry: target sound, position (in Arabic), Arabic word, emoji, elicitation cue
_ABU_HASIBA_ITEMS = [
    # ب
    {"sound": "ب", "pos": "أول الكلمة", "word": "بَيْت", "emoji": "🏠", "cue": "ما هذا؟"},
    {"sound": "ب", "pos": "وسط الكلمة", "word": "كِتَاب", "emoji": "📚", "cue": "ماذا تقرأ؟"},
    {"sound": "ب", "pos": "آخر الكلمة", "word": "كَلْب", "emoji": "🐕", "cue": "ما هذا الحيوان؟"},
    # ت
    {"sound": "ت", "pos": "أول الكلمة", "word": "تُفَّاح", "emoji": "🍎", "cue": "ما هذه الفاكهة؟"},
    {"sound": "ت", "pos": "وسط الكلمة", "word": "قِطَّة", "emoji": "🐱", "cue": "ما هذا الحيوان؟"},
    {"sound": "ت", "pos": "آخر الكلمة", "word": "بَيْت", "emoji": "🏠", "cue": "أين تسكن؟ في..."},
    # ث
    {"sound": "ث", "pos": "أول الكلمة", "word": "ثَعْلَب", "emoji": "🦊", "cue": "ما هذا الحيوان؟"},
    {"sound": "ث", "pos": "آخر الكلمة", "word": "ثُلّث", "emoji": "🔳", "cue": "كم جزءاً تحتوي هذه الشكلة؟"},
    # ج
    {"sound": "ج", "pos": "أول الكلمة", "word": "جَمَل", "emoji": "🐪", "cue": "ما هذا الحيوان؟"},
    {"sound": "ج", "pos": "وسط الكلمة", "word": "دَجَاجَة", "emoji": "🐔", "cue": "ما هذا الطائر؟"},
    # ح
    {"sound": "ح", "pos": "أول الكلمة", "word": "حِصَان", "emoji": "🐴", "cue": "ما هذا الحيوان؟"},
    {"sound": "ح", "pos": "وسط الكلمة", "word": "بُحَيْرَة", "emoji": "🦟", "cue": "ما هذا الحيوان؟"},
    # خ
    {"sound": "خ", "pos": "أول الكلمة", "word": "خُبْز", "emoji": "🍞", "cue": "ما هذا الطعام؟"},
    {"sound": "خ", "pos": "آخر الكلمة", "word": "نُخَّلة", "emoji": "🌴", "cue": "ما هذه الشجرة؟"},
    # د
    {"sound": "د", "pos": "أول الكلمة", "word": "دُبّ", "emoji": "🐻", "cue": "ما هذا الحيوان؟"},
    {"sound": "د", "pos": "آخر الكلمة", "word": "أَسَد", "emoji": "🦁", "cue": "ما هذا الحيوان؟"},
    # ذ
    {"sound": "ذ", "pos": "أول الكلمة", "word": "ذِئْب", "emoji": "🐺", "cue": "ما هذا الحيوان؟"},
    # ر
    {"sound": "ر", "pos": "أول الكلمة", "word": "رَجُل", "emoji": "👨", "cue": "من هذا؟"},
    {"sound": "ر", "pos": "وسط الكلمة", "word": "أَرْنَب", "emoji": "🐇", "cue": "ما هذا الحيوان؟"},
    {"sound": "ر", "pos": "آخر الكلمة", "word": "بَقَر", "emoji": "🐄", "cue": "ما هذا الحيوان؟"},
    # ز
    {"sound": "ز", "pos": "أول الكلمة", "word": "زَهْرَة", "emoji": "🌸", "cue": "ما هذه؟"},
    {"sound": "ز", "pos": "وسط الكلمة", "word": "مَوزَة", "emoji": "🍌", "cue": "ما هذه الفاكهة؟"},
    # س
    {"sound": "س", "pos": "أول الكلمة", "word": "سَمَكَة", "emoji": "🐟", "cue": "ما هذا؟"},
    {"sound": "س", "pos": "آخر الكلمة", "word": "فَرَس", "emoji": "🐴", "cue": "ما هذا الحيوان؟"},
    # ش
    {"sound": "ش", "pos": "أول الكلمة", "word": "شَجَرَة", "emoji": "🌳", "cue": "ما هذه؟"},
    {"sound": "ش", "pos": "وسط الكلمة", "word": "عُصْفُور", "emoji": "🐦", "cue": "ما هذا الطائر؟"},
    # ص
    {"sound": "ص", "pos": "أول الكلمة", "word": "صَابُون", "emoji": "🦼", "cue": "ما هذا؟"},
    {"sound": "ص", "pos": "آخر الكلمة", "word": "قَفَص", "emoji": "🦚", "cue": "ما هذا الطائر؟"},
    # ض
    {"sound": "ض", "pos": "أول الكلمة", "word": "ضِفْدَع", "emoji": "🐸", "cue": "ما هذا الحيوان؟"},
    {"sound": "ض", "pos": "وسط الكلمة", "word": "رَوضَة", "emoji": "🌻", "cue": "ما هذه؟"},
    # ط
    {"sound": "ط", "pos": "أول الكلمة", "word": "طَائِر", "emoji": "🐦", "cue": "ما هذا؟"},
    {"sound": "ط", "pos": "آخر الكلمة", "word": "بُطْط", "emoji": "🦆", "cue": "ما هذا الطائر؟"},
    # ع
    {"sound": "ع", "pos": "أول الكلمة", "word": "عَيْن", "emoji": "👁️", "cue": "ما هذا؟"},
    {"sound": "ع", "pos": "آخر الكلمة", "word": "أُسْبُوع", "emoji": "📅", "cue": "كم يوماً فيه؟"},
    # غ
    {"sound": "غ", "pos": "أول الكلمة", "word": "غَنَم", "emoji": "🐑", "cue": "ما هذا الحيوان؟"},
    # ف
    {"sound": "ف", "pos": "أول الكلمة", "word": "فِيل", "emoji": "🐘", "cue": "ما هذا الحيوان؟"},
    {"sound": "ف", "pos": "آخر الكلمة", "word": "خُرُوف", "emoji": "🐑", "cue": "ما هذا الحيوان؟"},
    # ق
    {"sound": "ق", "pos": "أول الكلمة", "word": "قِرد", "emoji": "🐒", "cue": "ما هذا الحيوان؟"},
    {"sound": "ق", "pos": "آخر الكلمة", "word": "طَرِيق", "emoji": "🛣️", "cue": "ما هذا؟"},
    # ك
    {"sound": "ك", "pos": "أول الكلمة", "word": "كُرَّسِي", "emoji": "🪑", "cue": "ما هذا؟"},
    {"sound": "ك", "pos": "آخر الكلمة", "word": "سَمَك", "emoji": "🐟", "cue": "ما هذا؟"},
    # ل
    {"sound": "ل", "pos": "أول الكلمة", "word": "لَيْمُون", "emoji": "🍋", "cue": "ما هذه الفاكهة؟"},
    {"sound": "ل", "pos": "آخر الكلمة", "word": "جَبَل", "emoji": "⛰️", "cue": "ما هذا؟"},
    # م
    {"sound": "م", "pos": "أول الكلمة", "word": "مَوْزَة", "emoji": "🍌", "cue": "ما هذه الفاكهة؟"},
    {"sound": "م", "pos": "آخر الكلمة", "word": "قَلَم", "emoji": "✏️", "cue": "ما هذا؟"},
    # ن
    {"sound": "ن", "pos": "أول الكلمة", "word": "نَمِر", "emoji": "🐯", "cue": "ما هذا الحيوان؟"},
    {"sound": "ن", "pos": "آخر الكلمة", "word": "حصان", "emoji": "🐴", "cue": "ما هذا الحيوان؟"},
    # ه
    {"sound": "ه", "pos": "أول الكلمة", "word": "هِرَّة", "emoji": "🐱", "cue": "ما هذا الحيوان؟"},
    {"sound": "ه", "pos": "آخر الكلمة", "word": "وَجْه", "emoji": "🙂", "cue": "ما هذا؟"},
    # و
    {"sound": "و", "pos": "أول الكلمة", "word": "وَردَة", "emoji": "🌹", "cue": "ما هذه؟"},
    # ي
    {"sound": "ي", "pos": "أول الكلمة", "word": "يَد", "emoji": "✋", "cue": "ما هذا؟"},
    {"sound": "ي", "pos": "آخر الكلمة", "word": "كُرسِي", "emoji": "🪑", "cue": "ما هذا؟"},
]


def _get_hasiba_card_html(idx):
    """Return styled HTML card for the given Abu Hasiba item index."""
    if idx >= len(_ABU_HASIBA_ITEMS):
        return (
            "<div class='task-card' style='border-color:#2ecc71'>"
            "<div style='font-size:90px;text-align:center'>🎉</div>"
            "<div class='task-prompt' style='color:#2ecc71'>اكتملت جميع بنود الاختبار!</div>"
            "<div class='task-hint'>تم تسجيل جميع النتائج — يمكنك الآن رؤية نسبة كفاءة النطق أدناه</div>"
            "</div>"
        )
    item = _ABU_HASIBA_ITEMS[idx]
    n = len(_ABU_HASIBA_ITEMS)
    return (
        f"<div class='task-card'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:10px'>"
        f"<span style='background:#2d3561;color:#a0c4ff;padding:4px 14px;border-radius:20px;font-size:13px'>🔉 صوت: {item['sound']}</span>"
        f"<span style='color:#888;font-size:13px'>موضعه: {item['pos']} | بند {idx+1}/{n}</span>"
        f"</div>"
        f"<div style='font-size:100px;text-align:center;padding:12px'>{item['emoji']}</div>"
        f"<div class='task-prompt'>قُل: <span style='color:#f1c40f;font-size:32px'>{item['word']}</span></div>"
        f"<hr style='border-color:#2d3561;margin:12px 0'>"
        f"<p style='color:#e0e0e0;font-size:15px;text-align:right'><b>💬 تلميح:</b> {item['cue']}</p>"
        f"<p style='color:#f1c40f;font-size:14px;text-align:right;margin-top:4px'>"
        f"✔ هل نطق الطفل صوت <b style='font-size:20px;color:#fff'>{item['sound']}</b> بشكل صحيح؟"
        f"</p>"
        f"</div>"
    )


def _get_hasiba_progress_html(idx, results):
    n = len(_ABU_HASIBA_ITEMS)
    done    = len([r for r in results if r is not None])
    correct = sum(1 for r in results if r is True)
    bar = "█" * done + "░" * (n - done)
    if idx >= n:
        pct = round(correct / n * 100, 1) if n > 0 else 0
        return f"✅ **اكتمل الاختبار** ({done}/{n}) — صحيح: **{correct}** — CAP = **{pct}%**"
    return f"📊 `{bar}` ({done}/{n}) — صحيح حتى الآن: **{correct}**"


def compute_cap_from_items(results):
    """Returns (total, correct, markdown_message) from Abu Hasiba pass/fail list."""
    total   = len(_ABU_HASIBA_ITEMS)
    correct = sum(1 for r in results if r is True)
    pct     = round(correct / total * 100, 1) if total > 0 else 0.0
    msg     = f"✅ نتيجة الاختبار: **{correct} / {total}** صوتاً صحيحاً — CAP = **{pct}%**"
    return float(total), float(correct), msg


# =====================================================================
# 9. GRADIO UI  (3 Tabs)
# =====================================================================
_RTL_CSS = """
/* ── Global RTL ── */
body, .gradio-container { direction: rtl; text-align: right; }

/* Tabs bar */
.tab-nav { flex-direction: row-reverse; }

/* Labels & markdown */
label, .label-wrap, .block, p, h1, h2, h3, h4, h5, h6,
.prose, .md { direction: rtl; text-align: right; }

/* Inputs */
input, textarea, select { direction: rtl; text-align: right; }

/* Radio / Checkbox groups */
.wrap { flex-direction: row-reverse; justify-content: flex-start; }
.wrap .label-wrap { margin-right: 0; margin-left: 8px; }

/* Sliders */
.gradio-slider { direction: ltr; }   /* keep slider handle L→R */

/* Task wizard card */
.task-card {
    background: #16213e;
    border-radius: 14px;
    padding: 22px;
    border: 2px solid #2d3561;
    direction: rtl;
    text-align: center;
    min-height: 190px;
    margin-bottom: 4px;
}
.task-prompt {
    font-size: 22px;
    color: #ffffff;
    font-weight: bold;
    margin: 12px 0 6px;
    font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
}
.task-hint {
    font-size: 14px;
    color: #a0c4ff;
    margin-top: 6px;
    font-style: italic;
    line-height: 1.6;
}

/* Table inside Markdown */
table { margin-right: auto; margin-left: 0; }
th, td { text-align: right; }

/* Buttons */
button { font-family: "Segoe UI", "Tahoma", "Arial", sans-serif; }
"""

with gr.Blocks(title="منصة علاج وتتبع اضطرابات النطق الذكية",
               theme=gr.themes.Soft(), css=_RTL_CSS) as demo:

    gr.Markdown("# 🎙️ منصة علاج اضطرابات النطق المتكاملة للأطفال")
    gr.Markdown(
        "### نظام فحص ذكي متكامل: أشجار القرار ML "
        "+ التقييم العيادي الكمي (LQ / CAP / SSI-4) "
        "+ تتبع حركي بالرؤية الحاسوبية"
    )

    # ── Tab 1: ML Quick Screening ─────────────────────────────────────
    with gr.Tab("📋 الفحص المبدئي السريع  (ML Decision Tree)"):
        gr.Markdown("#### استبيان أولي مقنن — إجابات ثنائية — يستخدم نموذج شجرة قرار مُدرَّب")
        with gr.Row():
            with gr.Column():
                input_age = gr.Radio(
                    ["سنتين إلى 3 سنوات", "3 إلى 4 سنوات", "4 إلى 6 سنوات"],
                    label="الفئة العمرية للطفل", value="3 إلى 4 سنوات")
                input_vocab = gr.Radio(
                    ["لا، طبيعية", "نعم، مفرداته قليلة جداً مقارنة بأقرانه"],
                    label="هل يعاني من ضعف مفردات أو تأخر لغوي؟  (الرفاعي)",
                    value="لا، طبيعية")
                input_stutter = gr.Radio(
                    ["لا، كلامه مسترسل", "نعم، يكرر المقاطع بشكل ملحوظ وبجهد"],
                    label="هل تلاحظ علامات تكرار كلمات أو حبس صوت؟  (ريلي)",
                    value="لا، كلامه مسترسل")
                input_art = gr.Radio(
                    ["لا، مخارج الحروف سليمة", "نعم، يستبدل أو يحذف بعض الحروف"],
                    label="هل يستبدل الحروف أو يحذف أواخر الكلمات؟  (أبو حسيبة)",
                    value="لا، مخارج الحروف سليمة")
                btn_ml = gr.Button("🚀 تحليل البيانات وإصدار القرار التشخيصي", variant="primary")
            with gr.Column():
                out_diag = gr.Markdown("### النتيجة التشخيصية الذكية:\n*بانتظار الفحص...*")
                out_acts = gr.Markdown("### خطة العمل والتحكم التلقائي بالنظام:\n*بانتظار الفحص...*")

        btn_ml.click(run_screening_and_setup_pipeline,
                     inputs=[input_age, input_vocab, input_stutter, input_art],
                     outputs=[out_diag, out_acts])

    # ── Tab 2: Clinical Quantitative Assessment ───────────────────────
    with gr.Tab("🔬 التقييم العيادي الكمي  (LQ · CAP · SSI-4 · DSM-5)"):
        gr.Markdown(
            "#### تقييم عيادي كمي متكامل: "
            "معادلات الرفاعي (LQ) + أبو حسيبة (CAP) + SSI-4 + بوابة DSM-5"
        )

        with gr.Tabs():

            # ── Sub-tab 1: Al-Rifai LQ ───────────────────────────────
            with gr.Tab("📏 1 — نسبة اللغة (الرفاعي LQ)"):
                gr.Markdown(
                    "##### أدخل العمر الزمني وحدد العمر اللغوي — إما مباشرة أو عبر مهام التقييم التفاعلية"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ca_months = gr.Number(
                            label="العمر الزمني للطفل (بالأشهر)",
                            value=48, minimum=12, maximum=120,
                        )
                        la_mode = gr.Radio(
                            ["✏️ إدخال مباشر", "📋 احسب من معايير النمو اللغوي"],
                            label="طريقة تحديد العمر اللغوي",
                            value="✏️ إدخال مباشر",
                        )
                        gr.Markdown(
                            "> **LQ = (العمر اللغوي ÷ العمر الزمني) × 100**  \n"
                            "> < 70 → حاد | 70–79 → خفيف–متوسط | 80–89 → حدي | ≥ 90 → طبيعي"
                        )

                    with gr.Column(scale=2):
                        # ── Direct input ──────────────────────────────
                        with gr.Group(visible=True) as la_direct_group:
                            la_months = gr.Number(
                                label="العمر اللغوي المُقيَّم (بالأشهر)",
                                value=44, minimum=0, maximum=120,
                            )

                        # ── Task wizard ───────────────────────────────
                        with gr.Group(visible=False) as la_milestone_group:
                            task_idx      = gr.State(0)
                            task_results  = gr.State([])
                            task_card_html = gr.HTML(value=_get_task_card_html(0))
                            task_progress  = gr.Markdown(_get_progress_html(0, []))
                            with gr.Row():
                                btn_restart   = gr.Button("🔄 إعادة التشغيل",       variant="secondary", scale=1)
                                btn_task_fail = gr.Button("❌ لم يُنجح في المهمة",   variant="stop",       scale=2)
                                btn_task_pass = gr.Button("✅ نجح في المهمة",        variant="primary",    scale=2)
                            la_milestone_result = gr.Markdown("")

            # ── Sub-tab 2: Abu Hasiba CAP ────────────────────────────
            with gr.Tab("🗣️ 2 — كفاءة النطق (أبو حسيبة CAP)"):
                gr.Markdown(
                    "##### اختر طريقة الإدخال: إدخال مباشر للأرقام أو اختبار تفاعلي صوت بصوت"
                )

                hasiba_mode = gr.Radio(
                    ["✏️ إدخال مباشر", "🎙️ اختبار تفاعلي (صوت بصوت)"],
                    label="طريقة إدخال نتائج اختبار أبو حسيبة",
                    value="✏️ إدخال مباشر",
                )

                # ── Direct entry group ────────────────────────────────
                with gr.Group(visible=True) as hasiba_direct_group:
                    with gr.Row():
                        with gr.Column():
                            total_sounds = gr.Number(
                                label="إجمالي أصوات الكلام المُختبَرة (مثلاً: 90)",
                                value=90, minimum=1, maximum=500,
                            )
                            correct_sounds = gr.Number(
                                label="الأصوات المنطوقة بشكل صحيح",
                                value=72, minimum=0, maximum=500,
                            )
                            gr.Markdown(
                                "> **CAP = (الصحيحة ÷ الإجمالي) × 100**  \n"
                                "> CAP < 75% → اضطراب أصوات الكلام (SSD)  \n"
                                "> CAP ≥ 75% → كفاءة نطقية طبيعية"
                            )
                        with gr.Column():
                            gr.Markdown(
                                "### دليل تطبيق اختبار أبو حسيبة\n\n"
                                "1. استخدم مواد اختبار أبو حسيبة المعيارية\n"
                                "2. اعرض كل بطاقة صورة أمام الطفل\n"
                                "3. اطلب منه تسمية الصورة أو تكملة الجملة\n"
                                "4. سجّل الإجابات الصحيحة فقط\n"
                                "5. أدخل الإجمالي والصحيح في الخانتين أعلاه"
                            )

                # ── Interactive wizard group ──────────────────────────
                with gr.Group(visible=False) as hasiba_wizard_group:
                    hasiba_idx     = gr.State(0)
                    hasiba_results = gr.State([])

                    hasiba_card_html = gr.HTML(value=_get_hasiba_card_html(0))
                    hasiba_progress  = gr.Markdown(_get_hasiba_progress_html(0, []))

                    with gr.Row():
                        btn_hasiba_restart = gr.Button("🔄 إعادة التشغيل",       variant="secondary", scale=1)
                        btn_hasiba_fail    = gr.Button("❌ لم ينطق الصوت بشكل صحيح", variant="stop",      scale=2)
                        btn_hasiba_pass    = gr.Button("✅ نطق الصوت بشكل صحيح",    variant="primary",    scale=2)

                    hasiba_result_msg = gr.Markdown("")

                # ── Hidden number boxes always used by btn_clin ───────
                # (when in wizard mode these get auto-filled; otherwise user types)
                # We need total_sounds and correct_sounds to exist outside both
                # groups — they were already declared inside hasiba_direct_group,
                # so they remain accessible to btn_clin.click below.

            # ── Sub-tab 3: SSI-4 ─────────────────────────────────────
            with gr.Tab("🌊 3 — شدة التلعثم (SSI-4)"):
                gr.Markdown(
                    "##### حرك كل شريط إلى الدرجة المناسبة بناءً على ملاحظتك للطفل"
                )
                with gr.Row():
                    with gr.Column():
                        f_score = gr.Slider(
                            0, 18, value=0, step=1,
                            label="درجة التكرار  Frequency  (0 – 18 نقطة)",
                        )
                        gr.Markdown(
                            "> كم مرة يتكرر التلعثم؟  \n"
                            "> 0 = لا يوجد | 1–6 = نادر | 7–12 = متكرر | 13–18 = متكرر جداً"
                        )
                        d_score = gr.Slider(
                            0, 14, value=0, step=1,
                            label="درجة المدة  Duration  (0 – 14 نقطة)",
                        )
                        gr.Markdown(
                            "> كم تستغرق نوبة التلعثم؟  \n"
                            "> 0 = لا يوجد | 1–4 = قصيرة | 5–9 = متوسطة | 10–14 = طويلة"
                        )
                        p_score = gr.Slider(
                            0, 20, value=0, step=1,
                            label="درجة المصاحبات الجسدية  Physical  (0 – 20 نقطة)",
                        )
                        gr.Markdown(
                            "> هل يصاحب التلعثم حركات جسدية؟  \n"
                            "> 0 = لا يوجد | 1–6 = خفيفة | 7–13 = متوسطة | 14–20 = شديدة"
                        )
                    with gr.Column():
                        gr.Markdown(
                            "### جدول تفسير إجمالي SSI-4\n\n"
                            "| الإجمالي | الشدة |\n"
                            "|---------|------|\n"
                            "| 0 | طلاقة كاملة |\n"
                            "| 1 – 10 | خفيف |\n"
                            "| 11 – 17 | خفيف–متوسط |\n"
                            "| 18 – 24 | متوسط |\n"
                            "| 25 – 35 | شديد |\n"
                            "| 36 – 52 | شديد جداً |\n\n"
                            "> **المجموع = Frequency + Duration + Physical**"
                        )

            # ── Sub-tab 4: DSM-5 Organic Gate ────────────────────────
            with gr.Tab("🏥 4 — بوابة DSM-5 (الاستبعاد العضوي)"):
                gr.Markdown(
                    "##### هل يوجد سبب عضوي أو طبي يفسّر الاضطراب اللغوي؟"
                )
                with gr.Row():
                    with gr.Column():
                        organic = gr.Radio(
                            ["لا — لا يوجد سبب عضوي/بنيوي معروف",
                             "نعم — يوجد سبب عضوي/بنيوي"],
                            label="هل يعاني الطفل من أسباب عضوية أو بنيوية؟",
                            value="لا — لا يوجد سبب عضوي/بنيوي معروف",
                        )
                        gr.Markdown(
                            "> **في حال نعم:** يتوقف التقييم الآلي فوراً  \n"
                            "> ويُحال الطفل إلزامياً إلى المختص الطبي"
                        )
                    with gr.Column():
                        gr.Markdown(
                            "### أمثلة على الأسباب العضوية\n\n"
                            "- شق الحنك (Cleft Palate)\n"
                            "- فقدان السمع (Hearing Loss)\n"
                            "- شلل دماغي (Cerebral Palsy)\n"
                            "- تلف عصبي (Neurological Damage)\n"
                            "- متلازمة داون أو أي متلازمة وراثية\n"
                            "- إصابة في منطقة النطق / الحنجرة\n\n"
                            "> إذا كان الاضطراب ناتجاً عن أحد هذه الأسباب،  \n"
                            "> فهو **ثانوي** ويتطلب تدخلاً طبياً قبل التأهيل اللغوي."
                        )

            # ── Sub-tab 5: Results ───────────────────────────────────
            with gr.Tab("🔬 5 — النتائج والمؤشرات"):
                gr.Markdown("##### اضغط الزر لحساب جميع الدرجات وعرض التقرير والمخططات")
                btn_clin = gr.Button(
                    "🔬 احسب الدرجات وأصدر التقرير العيادي الشامل",
                    variant="primary", size="lg",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        clinical_report = gr.Markdown("### بانتظار التقييم...")
                    with gr.Column(scale=2):
                        clinical_plot = gr.Plot(label="لوحة المؤشرات العيادية")

        # ── Toggle: show/hide direct vs task-wizard input groups ────
        def _toggle_la_mode(mode):
            direct    = mode == "✏️ إدخال مباشر"
            milestone = not direct
            return gr.update(visible=direct), gr.update(visible=milestone)

        la_mode.change(
            _toggle_la_mode,
            inputs=[la_mode],
            outputs=[la_direct_group, la_milestone_group],
        )

        # ── Task wizard callbacks ────────────────────────────────────
        def _advance_task(idx, results, passed):
            results = list(results)
            while len(results) <= idx:
                results.append(None)
            results[idx] = passed
            next_idx = idx + 1
            card     = _get_task_card_html(next_idx)
            progress = _get_progress_html(next_idx, results)
            if next_idx >= len(_LANGUAGE_TASKS):
                la_val, msg = compute_la_from_tasks(results)
                return next_idx, results, card, progress, gr.update(value=la_val), msg
            return next_idx, results, card, progress, gr.update(), ""

        def _restart_tasks():
            idx, results = 0, []
            return idx, results, _get_task_card_html(idx), _get_progress_html(idx, results), gr.update(), ""

        _wizard_outputs = [task_idx, task_results, task_card_html, task_progress, la_months, la_milestone_result]

        btn_task_pass.click(
            lambda idx, res: _advance_task(idx, res, True),
            inputs=[task_idx, task_results], outputs=_wizard_outputs,
        )
        btn_task_fail.click(
            lambda idx, res: _advance_task(idx, res, False),
            inputs=[task_idx, task_results], outputs=_wizard_outputs,
        )
        btn_restart.click(_restart_tasks, inputs=[], outputs=_wizard_outputs)

        # ── Abu Hasiba wizard callbacks ──────────────────────────────
        def _toggle_hasiba_mode(mode):
            direct  = mode == "✏️ إدخال مباشر"
            return gr.update(visible=direct), gr.update(visible=not direct)

        hasiba_mode.change(
            _toggle_hasiba_mode,
            inputs=[hasiba_mode],
            outputs=[hasiba_direct_group, hasiba_wizard_group],
        )

        def _advance_hasiba(idx, results, passed):
            results = list(results)
            while len(results) <= idx:
                results.append(None)
            results[idx] = passed
            next_idx = idx + 1
            card     = _get_hasiba_card_html(next_idx)
            progress = _get_hasiba_progress_html(next_idx, results)
            if next_idx >= len(_ABU_HASIBA_ITEMS):
                tot, cor, msg = compute_cap_from_items(results)
                return (next_idx, results, card, progress,
                        gr.update(value=tot), gr.update(value=cor), msg)
            return next_idx, results, card, progress, gr.update(), gr.update(), ""

        def _restart_hasiba():
            idx, results = 0, []
            return (idx, results, _get_hasiba_card_html(idx),
                    _get_hasiba_progress_html(idx, results),
                    gr.update(), gr.update(), "")

        _hasiba_outputs = [
            hasiba_idx, hasiba_results,
            hasiba_card_html, hasiba_progress,
            total_sounds, correct_sounds, hasiba_result_msg,
        ]

        btn_hasiba_pass.click(
            lambda idx, res: _advance_hasiba(idx, res, True),
            inputs=[hasiba_idx, hasiba_results], outputs=_hasiba_outputs,
        )
        btn_hasiba_fail.click(
            lambda idx, res: _advance_hasiba(idx, res, False),
            inputs=[hasiba_idx, hasiba_results], outputs=_hasiba_outputs,
        )
        btn_hasiba_restart.click(_restart_hasiba, inputs=[], outputs=_hasiba_outputs)

        btn_clin.click(
            run_clinical_assessment,
            inputs=[ca_months, la_months, total_sounds, correct_sounds,
                    f_score, d_score, p_score, organic],
            outputs=[clinical_report, clinical_plot],
        )

        btn_clin.click(
            run_clinical_assessment,
            inputs=[ca_months, la_months, total_sounds, correct_sounds,
                    f_score, d_score, p_score, organic],
            outputs=[clinical_report, clinical_plot])

    # ── Tab 3: MediaPipe Lip Tracking ─────────────────────────────────
    with gr.Tab("🎥 التتبع الحركي المباشر  (MediaPipe)"):
        gr.Markdown("#### تتبع حركي فوري لملامح الوجه والشفاه أثناء التمارين العلاجية")
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(sources=["webcam"], streaming=True,
                                        label="كاميرا تتبع الطفل")
            with gr.Column():
                cv_analytics = gr.Textbox(
                    label="مؤشرات القياس والتحليل العيادي الفوري (Real-time Analytics)",
                    placeholder="بانتظار بث الكاميرا...")
        webcam_input.stream(process_video_frame,
                            inputs=[webcam_input],
                            outputs=[webcam_input, cv_analytics])


# =====================================================================
# LAUNCH
# =====================================================================
if __name__ == "__main__":
    demo.queue()
    demo.launch()
