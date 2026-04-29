import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
import plotly.graph_objects as go

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paddy Disease Detection",
    page_icon="🌾",
    layout="wide"
)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    disease_model = tf.keras.models.load_model("best_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    try:
        validator        = tf.keras.models.load_model("paddy_validator.keras")
        validator_loaded = True
    except Exception:
        validator        = None
        validator_loaded = False
    return disease_model, class_names, validator, validator_loaded

disease_model, class_names, validator, validator_loaded = load_models()

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
VALIDATOR_THRESHOLD  = 0.60

DISEASE_INFO = {
    "Bacterialblight": {
        "description": "Bacterial Leaf Blight is caused by Xanthomonas oryzae pv. oryzae. It causes wilting, yellowing and drying of rice leaves, severely reducing crop yield.",
        "symptoms": [
            "Water-soaked lesions starting from leaf edges",
            "Yellowing and drying of leaves",
            "Wilting of young seedlings",
            "Milky or pale yellow bacterial ooze on cut stems"
        ],
        "treatment": [
            "Use resistant paddy varieties",
            "Apply copper-based bactericides",
            "Avoid excess nitrogen fertilizer",
            "Drain fields during outbreak period",
            "Remove and destroy infected plant debris"
        ],
        "severity": "High",
        "color": "#e67e22"
    },
    "Blast": {
        "description": "Rice Blast is caused by the fungus Magnaporthe oryzae. It is considered the most devastating rice disease worldwide, capable of destroying entire crops.",
        "symptoms": [
            "Diamond or spindle-shaped lesions on leaves",
            "Gray or white centers with dark brown borders",
            "White to gray colored panicles",
            "Collar rot at leaf-blade junction"
        ],
        "treatment": [
            "Apply Tricyclazole or Propiconazole fungicide",
            "Use blast-resistant certified varieties",
            "Avoid excessive nitrogen fertilization",
            "Ensure proper field drainage",
            "Apply silicon fertilizers to strengthen plant"
        ],
        "severity": "Very High",
        "color": "#e74c3c"
    },
    "Brownspot": {
        "description": "Brown Spot is caused by the fungus Cochliobolus miyabeanus. It mainly affects nutrient-deficient crops and can cause significant yield loss.",
        "symptoms": [
            "Oval or circular brown spots on leaves",
            "Dark brown borders with yellow halo around spots",
            "Spots also visible on leaf sheaths and grains",
            "Severe infection causes leaf drying"
        ],
        "treatment": [
            "Apply Mancozeb or Iprodione fungicide",
            "Use disease-free certified seeds",
            "Maintain proper soil potassium and silicon",
            "Treat seeds with fungicide before planting",
            "Avoid water stress during crop growth"
        ],
        "severity": "Medium",
        "color": "#f39c12"
    },
    "Healthy": {
        "description": "The paddy plant appears healthy with no visible signs of disease. Continue good agricultural practices to maintain plant health throughout the season.",
        "symptoms": [
            "No visible lesions or discoloration",
            "Uniform bright green leaf color",
            "Normal healthy leaf structure and shape",
            "Strong upright plant growth"
        ],
        "treatment": [
            "Continue current good farming practices",
            "Monitor regularly for early disease detection",
            "Maintain proper irrigation schedule",
            "Apply balanced fertilizer as recommended",
            "Keep field free from weeds"
        ],
        "severity": "None",
        "color": "#27ae60"
    },
    "Tungro": {
        "description": "Rice Tungro Disease is caused by two viruses (RTBV and RTSV) transmitted by the green leafhopper insect. It can cause up to 100% yield loss in severe outbreaks.",
        "symptoms": [
            "Yellow to orange leaf discoloration",
            "Stunted and reduced plant growth",
            "Reduced number of tillers",
            "Leaves may show interveinal chlorosis"
        ],
        "treatment": [
            "Control green leafhopper population with insecticides",
            "Plant tungro-resistant or tolerant varieties",
            "Apply carbofuran at transplanting",
            "Remove and destroy infected plants immediately",
            "Avoid late transplanting to escape leafhopper peak"
        ],
        "severity": "Very High",
        "color": "#e74c3c"
    }
}

SEVERITY_COLOR = {
    "None":      "#27ae60",
    "Medium":    "#f39c12",
    "High":      "#e67e22",
    "Very High": "#e74c3c"
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { text-align:center; padding:20px 0 10px 0; }
    .main-header h1 { color:#27ae60; font-size:2.4rem; margin-bottom:5px; }
    .main-header p  { color:#aaa; font-size:1rem; }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    .info-card {
        display: flex;
        align-items: flex-start;
        margin: 8px 0;
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid #333;
        background: #1a1a2e;
    }
    .badge {
        background: #27ae60;
        color: #fff;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: bold;
        flex-shrink: 0;
        margin-right: 12px;
    }
    .badge-warn {
        background: #e74c3c;
        color: #fff;
        font-size: 16px;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-right: 12px;
    }
    .info-text {
        font-size: 14px;
        color: #e0e0e0;
        line-height: 1.6;
    }
    .reject-box {
        background: #2a1a1a;
        border-left: 5px solid #e74c3c;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
        color: #f0c0c0;
        font-size: 15px;
        line-height: 1.6;
    }
    .paddy-hint {
        background: #1a2a1a;
        border-left: 4px solid #27ae60;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 12px;
        color: #b0d4b0;
        font-size: 13px;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ───────────────────────────────────────────────────────────
def check_image_quality(image):
    arr        = np.array(image.convert("RGB"))
    gray       = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    std        = np.std(arr)
    if blur < 50:
        return False, "Image is too blurry. Please upload a clearer, focused image."
    if brightness < 20:
        return False, "Image is too dark. Please use proper lighting."
    if brightness > 240:
        return False, "Image is overexposed. Please avoid direct flash or bright light."
    if std < 15:
        return False, "Image appears blank or single-colored."
    return True, "OK"

def preprocess(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def show_rejection(title, message, show_hint=True):
    st.error(f"🚫 {title}")
    st.markdown(f'<div class="reject-box">{message}</div>', unsafe_allow_html=True)
    if show_hint:
        st.markdown("""
        <div class="paddy-hint">
            <b>📌 What is a paddy leaf?</b><br>
            A long, narrow green leaf from the rice plant (<i>Oryza sativa</i>),
            typically grown in wet or flooded agricultural fields.<br><br>
            This app detects: <b>Bacterialblight · Blast · Brownspot · Healthy · Tungro</b>
        </div>
        """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌾 Paddy Disease Detection</h1>
    <p>Upload a paddy leaf image to instantly detect disease using MobileNetV2 deep learning</p>
</div>
<hr style="margin-bottom:25px; border-color:#333;">
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 About This App")
    st.markdown("""
    This app uses a **MobileNetV2** deep learning model
    trained on **7,220 paddy leaf images** to detect
    5 disease conditions with **99.58% accuracy**.
    """)
    st.markdown("---")
    st.markdown("### 📋 Model Details")
    st.markdown("""
    | Detail | Info |
    |--------|------|
    | Architecture | MobileNetV2 |
    | Training Images | 7,220 |
    | Overall Accuracy | 99.58% |
    | Input Size | 224 × 224 |
    | Min Confidence | 70% |
    """)
    st.markdown("---")
    st.markdown("### 🔬 Detectable Conditions")
    conditions = {
        "🦠 Bacterial Blight": ("99.37%", "#e67e22"),
        "💥 Blast":            ("98.61%", "#e74c3c"),
        "🟤 Brown Spot":       ("100.0%", "#f39c12"),
        "✅ Healthy":          ("100.0%", "#27ae60"),
        "🔴 Tungro":           ("100.0%", "#e74c3c"),
    }
    for name, (acc, col) in conditions.items():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between;
                    padding:6px 10px; margin:4px 0; border-radius:6px;
                    background:#1a1a2e;">
            <span style="font-size:13px; color:#e0e0e0;">{name}</span>
            <span style="font-size:13px; color:{col}; font-weight:bold;">{acc}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    if validator_loaded:
        st.success("✅ Paddy Validator: Active")
    else:
        st.warning("⚠️ Validator: Not loaded (using fallback)")
    st.markdown("---")
    st.warning("""
    ⚠️ **Paddy leaves only!**

    This app will automatically reject:
    - Other plant leaves
    - Human faces / objects
    - Blurry or dark images
    """)
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    for i, step in enumerate([
        "Upload a **clear paddy leaf** image",
        "System checks **image quality**",
        "System verifies it is a **paddy leaf**",
        "**Disease class** is predicted",
        "Follow the **treatment advice**"
    ], 1):
        st.markdown(f"**{i}.** {step}")

# ── Main Layout ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a paddy leaf image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit paddy (rice) leaf image only"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        w, h  = image.size
        st.image(image, caption=f"📁 {uploaded_file.name}", width=480)
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:10px; border-radius:8px;
                    font-size:13px; color:#aaa; margin-top:8px;">
            📐 Size: {w} × {h} px &nbsp;|&nbsp;
            🎨 Mode: {image.mode} &nbsp;|&nbsp;
            📄 {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1a1a2e; border:2px dashed #27ae60;
                    border-radius:12px; padding:50px; text-align:center;">
            <p style="font-size:48px; margin:0;">🌾</p>
            <p style="color:#27ae60; font-weight:bold; margin:12px 0 6px 0; font-size:16px;">
                Upload a Paddy Leaf Image
            </p>
            <p style="color:#888; font-size:13px; margin:0;">
                JPG or PNG format · Clear, well-lit image
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Results Column ─────────────────────────────────────────────────────────────
with col2:
    st.markdown("### 🔍 Analysis Results")

    if not uploaded_file:
        st.markdown("""
        <div style="background:#1a1a2e; border-radius:12px; padding:50px;
                    text-align:center; margin-top:10px;">
            <p style="font-size:42px; margin:0;">🔬</p>
            <p style="margin:12px 0 5px 0; font-size:15px; color:#888;">
                Results will appear here
            </p>
            <p style="margin:0; font-size:13px; color:#555;">
                Upload an image to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("🔬 Analyzing your image..."):
            img_array = preprocess(image)

            # Step 1 — Image Quality Check
            quality_ok, quality_msg = check_image_quality(image)
            if not quality_ok:
                show_rejection("Image Quality Issue", quality_msg, show_hint=False)
                st.stop()

            # Step 2 — Paddy Leaf Validation
            if validator is not None:
                paddy_prob = float(validator.predict(img_array, verbose=0)[0][0])
                if paddy_prob < VALIDATOR_THRESHOLD:
                    show_rejection(
                        "Not a Paddy Leaf Detected!",
                        f"This image does not appear to be a paddy (rice) leaf. "
                        f"Paddy confidence score: <b>{paddy_prob*100:.1f}%</b> "
                        f"(minimum required: {VALIDATOR_THRESHOLD*100:.0f}%). "
                        f"Please upload a clear paddy leaf image and try again."
                    )
                    st.stop()
            else:
                preds_check = disease_model.predict(img_array, verbose=0)
                if float(np.max(preds_check[0])) < 0.60:
                    show_rejection(
                        "Not a Paddy Leaf Detected!",
                        "This does not appear to be a paddy leaf. "
                        "Please upload a clear paddy (rice) leaf image."
                    )
                    st.stop()

            # Step 3 — Disease Prediction
            preds      = disease_model.predict(img_array, verbose=0)
            pred_idx   = int(np.argmax(preds[0]))
            confidence = float(preds[0][pred_idx])
            pred_class = class_names[str(pred_idx)]

            # Step 4 — Confidence Threshold
            if confidence < CONFIDENCE_THRESHOLD:
                show_rejection(
                    "Prediction Confidence Too Low",
                    f"The model is only <b>{confidence*100:.1f}%</b> confident, "
                    f"which is below the minimum threshold of "
                    f"<b>{CONFIDENCE_THRESHOLD*100:.0f}%</b>. "
                    f"Please upload a clearer, well-lit paddy leaf image.",
                    show_hint=False
                )
                st.stop()

        # ── Result Card ────────────────────────────────────────────────────────
        sev     = DISEASE_INFO[pred_class]["severity"]
        sev_col = SEVERITY_COLOR[sev]
        d_col   = DISEASE_INFO[pred_class]["color"]

        st.markdown(f"""
        <div style="background:#1a1a2e; border-left:5px solid {d_col};
                    padding:20px; border-radius:12px; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h2 style="color:{d_col}; margin:0; font-size:1.6rem;">{pred_class}</h2>
                    <p style="margin:5px 0 0 0; color:#aaa; font-size:14px;">
                        Predicted Disease Class
                    </p>
                </div>
                <div style="text-align:right;">
                    <p style="margin:0; font-size:1.9rem; font-weight:bold; color:{d_col};">
                        {confidence*100:.1f}%
                    </p>
                    <p style="margin:0; color:#888; font-size:13px;">Confidence</p>
                </div>
            </div>
            <hr style="border-color:{d_col}; opacity:0.3; margin:12px 0;">
            <div style="display:flex; gap:30px;">
                <div>
                    <span style="font-size:12px; color:#888;">SEVERITY</span><br>
                    <span style="font-weight:bold; color:{sev_col}; font-size:15px;">{sev}</span>
                </div>
                <div>
                    <span style="font-size:12px; color:#888;">VALIDATOR</span><br>
                    <span style="font-weight:bold; color:#27ae60; font-size:15px;">
                        {"Active ✅" if validator_loaded else "Fallback ⚠️"}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence Bar Chart ───────────────────────────────────────────────
        bar_colors = [
            d_col if class_names[str(i)] == pred_class else "#2d2d2d"
            for i in range(len(class_names))
        ]
        fig = go.Figure(go.Bar(
            x            = list(class_names.values()),
            y            = [round(float(preds[0][i]) * 100, 2) for i in range(len(class_names))],
            marker_color = bar_colors,
            text         = [f"{float(preds[0][i])*100:.1f}%" for i in range(len(class_names))],
            textposition = "outside",
            textfont     = dict(color="#e0e0e0")
        ))
        fig.update_layout(
            title         = dict(text="Confidence Score per Class",
                                 font=dict(size=14, color="#e0e0e0")),
            xaxis_title   = "Disease Class",
            yaxis_title   = "Confidence (%)",
            yaxis_range   = [0, 120],
            height        = 300,
            margin        = dict(t=50, b=40, l=40, r=20),
            plot_bgcolor  = "#0e0e0e",
            paper_bgcolor = "#0e0e0e",
            font          = dict(color="#e0e0e0"),
            showlegend    = False
        )
        fig.update_xaxes(showgrid=False, color="#aaa")
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a", color="#aaa")
        st.plotly_chart(fig, use_container_width=True)

# ── Disease Detail Tabs ────────────────────────────────────────────────────────
if uploaded_file and "pred_class" in st.session_state:
    pred_class = st.session_state["pred_class"]
    confidence = st.session_state["confidence"]

    if confidence >= CONFIDENCE_THRESHOLD:
        st.markdown("---")
        st.markdown("### 📚 Detailed Disease Information")

        info    = DISEASE_INFO[pred_class]
        sev     = info["severity"]
        sev_col = SEVERITY_COLOR[sev]
        d_col   = info["color"]

        tab1, tab2, tab3 = st.tabs(["📖 Description", "🔬 Symptoms", "💊 Treatment"])

        with tab1:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"#### {pred_class}")
                st.markdown(f"""
                <p style="font-size:15px; color:#e0e0e0; line-height:1.8; margin-top:8px;">
                    {info["description"]}
                </p>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div style="background:#1a1a2e; border:2px solid {sev_col};
                            border-radius:10px; padding:18px; text-align:center; margin-top:30px;">
                    <p style="margin:0; font-size:12px; color:#888; letter-spacing:1px;">SEVERITY</p>
                    <p style="margin:8px 0 0 0; font-size:20px; font-weight:bold; color:{sev_col};">
                        {sev}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("**Key symptoms to look for:**")
            st.markdown("<br>", unsafe_allow_html=True)
            for symptom in info["symptoms"]:
                st.markdown(f"""
                <div class="info-card">
                    <span class="badge-warn">⚠</span>
                    <span class="info-text">{symptom}</span>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown("**Recommended treatment steps:**")
            st.markdown("<br>", unsafe_allow_html=True)
            for i, step in enumerate(info["treatment"], 1):
                st.markdown(f"""
                <div class="info-card">
                    <span class="badge">{i}</span>
                    <span class="info-text">{step}</span>
                </div>
                """, unsafe_allow_html=True)

# Store results in session state after prediction (place AFTER col2 block)
if uploaded_file and "pred_class" in dir() and isinstance(pred_class, str):
    st.session_state["pred_class"]  = pred_class
    st.session_state["confidence"]  = confidence
elif not uploaded_file:
    st.session_state.pop("pred_class", None)
    st.session_state.pop("confidence", None)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#555; font-size:13px; padding:10px 0;">
    🌾 Paddy Disease Detection &nbsp;|&nbsp;
    MobileNetV2 Transfer Learning &nbsp;|&nbsp;
    Accuracy: 99.58% &nbsp;|&nbsp;
    Built with Streamlit
</p>
""", unsafe_allow_html=True)
