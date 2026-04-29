import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paddy Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme initialisation ──────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "🌑 Dark"

# ── Theme CSS factory ─────────────────────────────────────────────────────────
def get_theme_css(mode: str) -> str:
    # ── colour tokens per mode ────────────────────────────────────────────────
    if mode == "🌑 Dark":
        app_bg          = "linear-gradient(135deg,#0a1628 0%,#0d2137 50%,#0a1a0e 100%)"
        text_color      = "#e8f5e9"
        sidebar_bg      = "linear-gradient(180deg,#0d2137 0%,#0a1a0e 100%)"
        sidebar_border  = "#1b4332"
        sidebar_text    = "#b7e4c7"
        card_bg         = "linear-gradient(135deg,rgba(27,67,50,.6),rgba(13,33,55,.6))"
        card_border     = "#2d6a4f"
        card_h2         = "#52b788"
        card_p          = "#95d5b2"
        hero_grad       = "linear-gradient(90deg,#52b788,#95d5b2,#52b788)"
        hero_sub        = "#95d5b2"
        upload_border   = "#2d6a4f"
        upload_bg       = "rgba(27,67,50,0.15)"
        upload_hover    = "#52b788"
        info_bg         = "rgba(82,183,136,0.12)"
        info_border     = "#52b788"
        warn_bg         = "rgba(231,76,60,0.12)"
        warn_border     = "#e74c3c"
        tab_list_bg     = "rgba(27,67,50,0.3)"
        tab_text        = "#95d5b2"
        tab_sel_bg      = "#2d6a4f"
        tab_sel_text    = "#d8f3dc"
        btn_bg          = "linear-gradient(135deg,#2d6a4f,#52b788)"
        btn_shadow      = "rgba(82,183,136,0.4)"
        spinner_color   = "#52b788"
        hr_color        = "#1b4332"
        footer_color    = "#52b788"
        theme_badge_bg  = "rgba(82,183,136,0.15)"
        theme_badge_txt = "#52b788"

    elif mode == "☀️ Light":
        app_bg          = "linear-gradient(135deg,#f0fdf4 0%,#ecfdf5 50%,#f0fdf4 100%)"
        text_color      = "#14532d"
        sidebar_bg      = "linear-gradient(180deg,#dcfce7 0%,#f0fdf4 100%)"
        sidebar_border  = "#86efac"
        sidebar_text    = "#166534"
        card_bg         = "linear-gradient(135deg,rgba(187,247,208,.7),rgba(209,250,229,.7))"
        card_border     = "#4ade80"
        card_h2         = "#15803d"
        card_p          = "#166534"
        hero_grad       = "linear-gradient(90deg,#15803d,#22c55e,#15803d)"
        hero_sub        = "#166534"
        upload_border   = "#4ade80"
        upload_bg       = "rgba(187,247,208,0.3)"
        upload_hover    = "#15803d"
        info_bg         = "rgba(74,222,128,0.15)"
        info_border     = "#4ade80"
        warn_bg         = "rgba(239,68,68,0.1)"
        warn_border     = "#ef4444"
        tab_list_bg     = "rgba(187,247,208,0.5)"
        tab_text        = "#15803d"
        tab_sel_bg      = "#4ade80"
        tab_sel_text    = "#14532d"
        btn_bg          = "linear-gradient(135deg,#16a34a,#4ade80)"
        btn_shadow      = "rgba(74,222,128,0.45)"
        spinner_color   = "#16a34a"
        hr_color        = "#86efac"
        footer_color    = "#15803d"
        theme_badge_bg  = "rgba(74,222,128,0.2)"
        theme_badge_txt = "#15803d"

    else:  # System default — uses prefers-color-scheme
        return """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
h1 a,h2 a,h3 a,h4 a,h5 a,h6 a,[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,[data-testid="stMarkdownContainer"] h3 a{display:none!important;}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.78rem;font-weight:600;letter-spacing:.5px;}
.result-card{border-radius:16px;padding:20px 24px;margin-bottom:16px;backdrop-filter:blur(12px);border:1px solid;}
.footer{text-align:center;font-size:.78rem;padding:20px 0 8px;opacity:.7;letter-spacing:.5px;}
.js-plotly-plot .plotly{background:transparent!important;}
@media(prefers-color-scheme:dark){
  .stApp{background:linear-gradient(135deg,#0a1628,#0d2137 50%,#0a1a0e);color:#e8f5e9;}
  [data-testid="stSidebar"]{background:linear-gradient(180deg,#0d2137,#0a1a0e);border-right:1px solid #1b4332;}
  [data-testid="stSidebar"] *{color:#b7e4c7!important;}
  .metric-card{background:linear-gradient(135deg,rgba(27,67,50,.6),rgba(13,33,55,.6));border:1px solid #2d6a4f;}
  .metric-card h2{color:#52b788;} .metric-card p{color:#95d5b2;}
  .hero-title h1{background:linear-gradient(90deg,#52b788,#95d5b2,#52b788);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
  .hero-title p{color:#95d5b2;}
  .upload-box{border:2px dashed #2d6a4f;background:rgba(27,67,50,.15);}
  .upload-box:hover{border-color:#52b788;}
  .info-box{background:rgba(82,183,136,.12);border-left:4px solid #52b788;}
  .warn-box{background:rgba(231,76,60,.12);border-left:4px solid #e74c3c;}
  [data-baseweb="tab-list"]{background:rgba(27,67,50,.3)!important;}
  [data-baseweb="tab"]{color:#95d5b2!important;}
  [aria-selected="true"]{background:#2d6a4f!important;color:#d8f3dc!important;}
  .stButton>button{background:linear-gradient(135deg,#2d6a4f,#52b788);color:#fff;}
  hr{border-color:#1b4332!important;} .footer{color:#52b788;}
}
@media(prefers-color-scheme:light){
  .stApp{background:linear-gradient(135deg,#f0fdf4,#ecfdf5 50%,#f0fdf4);color:#14532d;}
  [data-testid="stSidebar"]{background:linear-gradient(180deg,#dcfce7,#f0fdf4);border-right:1px solid #86efac;}
  [data-testid="stSidebar"] *{color:#166534!important;}
  .metric-card{background:linear-gradient(135deg,rgba(187,247,208,.7),rgba(209,250,229,.7));border:1px solid #4ade80;}
  .metric-card h2{color:#15803d;} .metric-card p{color:#166534;}
  .hero-title h1{background:linear-gradient(90deg,#15803d,#22c55e,#15803d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
  .hero-title p{color:#166534;}
  .upload-box{border:2px dashed #4ade80;background:rgba(187,247,208,.3);}
  .upload-box:hover{border-color:#15803d;}
  .info-box{background:rgba(74,222,128,.15);border-left:4px solid #4ade80;}
  .warn-box{background:rgba(239,68,68,.1);border-left:4px solid #ef4444;}
  [data-baseweb="tab-list"]{background:rgba(187,247,208,.5)!important;}
  [data-baseweb="tab"]{color:#15803d!important;}
  [aria-selected="true"]{background:#4ade80!important;color:#14532d!important;}
  .stButton>button{background:linear-gradient(135deg,#16a34a,#4ade80);color:#fff;}
  hr{border-color:#86efac!important;} .footer{color:#15803d;}
}
"""

    # ── shared structural CSS (non-colour) ────────────────────────────────────
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
.stApp{{background:{app_bg};color:{text_color};}}
#MainMenu,footer,header{{visibility:hidden;}}
h1 a,h2 a,h3 a,h4 a,h5 a,h6 a,
.hero-title h1 a,[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,[data-testid="stMarkdownContainer"] h3 a{{display:none!important;}}
[data-testid="stSidebar"]{{background:{sidebar_bg};border-right:1px solid {sidebar_border};}}
[data-testid="stSidebar"] *{{color:{sidebar_text}!important;}}
.metric-card{{background:{card_bg};border:1px solid {card_border};border-radius:16px;
    padding:20px;text-align:center;backdrop-filter:blur(10px);
    transition:transform .2s ease,box-shadow .2s ease;}}
.metric-card:hover{{transform:translateY(-3px);box-shadow:0 8px 32px rgba(45,106,79,.4);}}
.metric-card h2{{font-size:2rem;font-weight:700;color:{card_h2};margin:0;}}
.metric-card p{{font-size:.8rem;color:{card_p};margin:4px 0 0;text-transform:uppercase;letter-spacing:1px;}}
.hero-title{{text-align:center;padding:30px 0 10px;}}
.hero-title h1{{font-size:3rem;font-weight:700;
    background:{hero_grad};-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;}}
.hero-title p{{color:{hero_sub};font-size:1.05rem;margin-top:8px;}}
.upload-box{{border:2px dashed {upload_border};border-radius:20px;padding:30px;
    background:{upload_bg};text-align:center;transition:border-color .3s;}}
.upload-box:hover{{border-color:{upload_hover};}}
.result-card{{border-radius:16px;padding:20px 24px;margin-bottom:16px;backdrop-filter:blur(12px);border:1px solid;}}
.badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.78rem;font-weight:600;letter-spacing:.5px;}}
.info-box{{background:{info_bg};border-left:4px solid {info_border};
    border-radius:0 12px 12px 0;padding:16px 20px;margin:12px 0;}}
.warn-box{{background:{warn_bg};border-left:4px solid {warn_border};
    border-radius:0 12px 12px 0;padding:16px 20px;margin:12px 0;}}
[data-baseweb="tab-list"]{{background:{tab_list_bg}!important;border-radius:10px!important;gap:4px;}}
[data-baseweb="tab"]{{color:{tab_text}!important;border-radius:8px!important;}}
[aria-selected="true"]{{background:{tab_sel_bg}!important;color:{tab_sel_text}!important;}}
.stButton>button{{background:{btn_bg};color:white;border:none;border-radius:10px;
    font-weight:600;padding:10px 28px;transition:all .3s ease;}}
.stButton>button:hover{{transform:translateY(-2px);box-shadow:0 6px 20px {btn_shadow};}}
.stSpinner>div{{border-top-color:{spinner_color}!important;}}
hr{{border-color:{hr_color}!important;opacity:.6;}}
.footer{{text-align:center;color:{footer_color};font-size:.78rem;
    padding:20px 0 8px;opacity:.7;letter-spacing:.5px;}}
.js-plotly-plot .plotly{{background:transparent!important;}}
.theme-badge{{display:inline-flex;align-items:center;gap:6px;
    background:{theme_badge_bg};color:{theme_badge_txt};
    border:1px solid {theme_badge_txt}44;border-radius:20px;
    padding:4px 12px;font-size:.78rem;font-weight:600;letter-spacing:.5px;}}
"""


# ── Apply CSS (re-evaluated every render) ─────────────────────────────────────
st.markdown(
    f"<style>{get_theme_css(st.session_state['theme'])}</style>",
    unsafe_allow_html=True,
)


# ── Constants ─────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Bacterialblight": {
        "description": "Bacterial Leaf Blight is caused by Xanthomonas oryzae pv. oryzae. It causes wilting and yellowing of leaves and can devastate entire fields.",
        "symptoms": [
            "Water-soaked lesions on leaf edges",
            "Yellowing and drying of leaf tips",
            "Wilting of seedlings (kresek symptom)",
        ],
        "treatment": [
            "Use resistant varieties",
            "Apply copper-based bactericides",
            "Avoid excess nitrogen fertilizer",
            "Drain fields during outbreak",
        ],
        "severity": "High",
        "emoji": "🦠",
    },
    "Blast": {
        "description": "Rice Blast is caused by Magnaporthe oryzae fungus. It is one of the most destructive rice diseases worldwide affecting leaves, nodes and panicles.",
        "symptoms": [
            "Diamond-shaped lesions on leaves",
            "Gray centers with brown borders",
            "White to gray panicles at harvest",
        ],
        "treatment": [
            "Apply fungicides like Tricyclazole",
            "Use blast-resistant varieties",
            "Avoid excessive nitrogen",
            "Ensure proper field drainage",
        ],
        "severity": "Very High",
        "emoji": "💥",
    },
    "Brownspot": {
        "description": "Brown Spot is caused by Cochliobolus miyabeanus fungus. It affects leaves, sheaths and grains leading to significant yield loss.",
        "symptoms": [
            "Oval brown spots on leaves",
            "Dark brown borders with yellow halo",
            "Spots on leaf sheaths and grains",
        ],
        "treatment": [
            "Apply Mancozeb or Iprodione fungicide",
            "Use healthy certified seeds",
            "Maintain proper soil nutrition",
            "Treat seeds before planting",
        ],
        "severity": "Medium",
        "emoji": "🟤",
    },
    "Healthy": {
        "description": "The plant appears completely healthy with no visible signs of disease. Continue monitoring regularly for early detection of any future issues.",
        "symptoms": [
            "No visible lesions or discoloration",
            "Normal vibrant green color",
            "Healthy leaf structure and texture",
        ],
        "treatment": [
            "Continue current farming practices",
            "Monitor regularly for early detection",
            "Maintain proper irrigation and nutrition",
        ],
        "severity": "None",
        "emoji": "✅",
    },
    "Tungro": {
        "description": "Rice Tungro Disease is caused by two viruses (RTBV and RTSV) transmitted by green leafhopper insects. It is a serious viral disease.",
        "symptoms": [
            "Yellow to orange discoloration of leaves",
            "Stunted plant growth",
            "Reduced tillering and panicle formation",
        ],
        "treatment": [
            "Control leafhopper population with insecticides",
            "Use tungro-resistant varieties",
            "Apply insecticides early in season",
            "Remove and destroy infected plants",
        ],
        "severity": "Very High",
        "emoji": "🔴",
    },
}

SEVERITY_COLORS = {
    "None":      {"bg": "rgba(82,183,136,0.15)",  "border": "#52b788", "text": "#52b788"},
    "Medium":    {"bg": "rgba(243,156,18,0.15)",   "border": "#f39c12", "text": "#f39c12"},
    "High":      {"bg": "rgba(230,126,34,0.15)",   "border": "#e67e22", "text": "#e67e22"},
    "Very High": {"bg": "rgba(231,76,60,0.15)",    "border": "#e74c3c", "text": "#e74c3c"},
}

CLASS_ACCURACY = {
    "Bacterialblight": "99.37%",
    "Blast":           "98.61%",
    "Brownspot":       "100.0%",
    "Healthy":         "100.0%",
    "Tungro":          "100.0%",
}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def check_leaf_color(image: Image.Image) -> tuple[bool, float]:
    """
    Converts image to HSV and checks if it contains enough
    green/yellow-green pixels to plausibly be a leaf.
    Paddy leaves (healthy or diseased) are mostly green, yellow-green,
    yellow, or brownish-green in HSV space.
    Returns (is_leaf_like, green_ratio).
    """
    import colorsys
    img_rgb = image.convert("RGB").resize((128, 128
    ))
    pixels = list(img_rgb.getdata())
    leaf_pixels = 0
    total = len(pixels)
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h_deg = h * 360
        # Leaf-like hues: green (60-165°) plus yellow (40-60°) plus brown (15-40° with low sat)
        # Also skip very dark or very washed-out pixels
        if v < 0.08 or v > 0.97:          # too dark / too bright (sky, white bg)
            continue
        if s < 0.08:                       # near-grey pixels (not leaf)
            continue
        # Green family: 50° – 170°  (covers healthy + diseased yellowing)
        if 40 <= h_deg <= 170:
            leaf_pixels += 1
        # Brownish-orange (diseased lesions): 15° – 40° with decent saturation
        elif 15 <= h_deg < 40 and s >= 0.25:
            leaf_pixels += 1
    green_ratio = leaf_pixels / max(total, 1)
    return green_ratio >= 0.10, green_ratio   # at least 10 % leaf-like pixels


def is_paddy_leaf(
    image: Image.Image,
    predictions: np.ndarray,
    conf_threshold: float = 80.0,
):
    """
    Multi-factor validation:
      1. Color check  – image must contain enough green/leaf-like pixels.
      2. Confidence   – model must be ≥ conf_threshold% sure of its prediction.
      3. Entropy gate – if the softmax distribution is very flat the model is
                        confused, which usually means the image is not paddy.
    Returns (is_valid: bool, reason: str, confidence: float, green_ratio: float)
    """
    max_conf = float(np.max(predictions[0])) * 100

    # ── 1. Color / structure check ───────────────────────────────────────────
    color_ok, green_ratio = check_leaf_color(image)

    # ── 2. Softmax entropy check ─────────────────────────────────────────────
    probs = predictions[0].astype(np.float64)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(len(probs)))      # worst-case (uniform)
    entropy_ratio = entropy / max_entropy         # 0=certain, 1=fully confused
    entropy_ok = entropy_ratio < 0.85            # reject if model is very unsure

    # ── 3. Confidence threshold ──────────────────────────────────────────────
    conf_ok = max_conf >= conf_threshold

    is_valid = color_ok and conf_ok and entropy_ok

    # Build a human-readable reason for rejection
    if is_valid:
        reason = "OK"
    else:
        parts = []
        if not color_ok:
            parts.append(f"insufficient leaf-like color ({green_ratio*100:.0f}% green pixels, need ≥10%)")
        if not conf_ok:
            parts.append(f"low model confidence ({max_conf:.1f}%, need ≥{conf_threshold}%)")
        if not entropy_ok:
            parts.append(f"model is uncertain (entropy {entropy_ratio*100:.0f}% of max)")
        reason = " · ".join(parts)

    return is_valid, reason, max_conf, green_ratio


def confidence_bar_chart(class_names: dict, predictions: np.ndarray, pred_idx: int, sev_color: str):
    labels = [class_names[str(i)] for i in range(len(class_names))]
    values = [float(p) * 100 for p in predictions[0]]
    colors = [sev_color if i == pred_idx else "rgba(82,183,136,0.25)" for i in range(len(class_names))]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=1,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#b7e4c7", size=11),
    ))
    fig.update_layout(
        title=dict(text="Confidence Score per Class", font=dict(color="#95d5b2", size=14)),
        xaxis=dict(tickfont=dict(color="#95d5b2"), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#95d5b2"), gridcolor="rgba(82,183,136,0.1)",
                   range=[0, 115], zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=10, r=10),
        height=280,
        showlegend=False,
    )
    return fig


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("🌱 Loading model..."):
    model, class_names = load_model()


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    <h1>🌾 Paddy Disease Detection</h1>
    <p>Upload a paddy leaf image · Get instant AI-powered disease diagnosis · Follow treatment advice</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Top metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><h2>99.58%</h2><p>Overall Accuracy</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><h2>5</h2><p>Disease Classes</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><h2>7,220</h2><p>Training Images</p></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><h2>224²</h2><p>Input Resolution</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Paddy Disease AI")
    st.markdown("---")

    # ── Theme switcher ────────────────────────────────────────────────────────
    st.markdown("### 🎨 Appearance")
    _theme_options = ["🌑 Dark", "☀️ Light", "🖥️ System Default"]
    selected_theme = st.radio(
        label="Choose theme",
        options=_theme_options,
        index=_theme_options.index(st.session_state["theme"]),
        horizontal=False,
        label_visibility="collapsed",
    )
    if selected_theme != st.session_state["theme"]:
        st.session_state["theme"] = selected_theme
        st.rerun()

    # Active theme badge
    _badge_icons = {"🌑 Dark": "🌑", "☀️ Light": "☀️", "🖥️ System Default": "🖥️"}
    st.markdown(
        f'<div class="theme-badge">{_badge_icons[st.session_state["theme"]]} '
        f'{st.session_state["theme"].split(" ",1)[1]} mode active</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📋 Model Details")
    for k, v in [("Architecture","MobileNetV2"), ("Backbone","ImageNet"),
                  ("Accuracy","99.58%"), ("Framework","TensorFlow"),
                  ("Input Size","224 × 224")]:
        st.markdown(f"**{k}:** {v}")

    st.markdown("---")
    st.markdown("### 🎯 Class Accuracy")
    for cls, info in DISEASE_INFO.items():
        acc = CLASS_ACCURACY[cls]
        st.markdown(f"{info['emoji']} **{cls}** — `{acc}`")

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    for i, step in enumerate(["Upload a **paddy leaf** image",
                               "Wait for AI analysis",
                               "Review disease diagnosis",
                               "Follow treatment steps"], 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("---")
    st.warning("⚠️ **Paddy leaves only.** Other images will be rejected automatically.")


# ── Main Layout ───────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

# Unified image source ─────────────────────────────────────────────────────────
image        = None   # PIL Image that feeds the model
source_label = ""     # shown in the info box

with col_upload:
    st.markdown("### 📷 Image Input")

    tab_upload, tab_camera = st.tabs(["📤 Upload Photo", "📷 Capture Photo"])

    # ── Tab 1 : File upload ───────────────────────────────────────────────────
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a paddy leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit image of a paddy leaf",
            label_visibility="collapsed",
        )
        if uploaded_file:
            image        = Image.open(uploaded_file)
            source_label = f"📁 {uploaded_file.name} &nbsp;|&nbsp; {uploaded_file.size // 1024} KB"
            st.image(image, caption=f"📷 {uploaded_file.name}", use_column_width=True)
            st.markdown(f"""
            <div class="info-box">
                <b>✅ Image Loaded</b><br>
                <span style="font-size:0.85rem; color:#95d5b2;">
                    {source_label}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="margin-top:12px;">
                <b style="color:#52b788;">⚠️ Important Notice</b><br>
                <span style="font-size:0.85rem;">
                    This app works with <b>paddy (rice) leaf images only</b>.
                    Uploading other plant leaves will show a warning.
                </span>
            </div>
            """, unsafe_allow_html=True)


    # ── Tab 2 : Camera capture ────────────────────────────────────────────────
    with tab_camera:
        st.markdown("""
        <div class="info-box" style="margin-bottom:12px;">
            <b>📷 Live Camera Capture</b><br>
            <span style="font-size:0.85rem;">
                Point your camera at a <b>paddy leaf</b> and click the
                shutter button below to capture and analyse it instantly.
            </span>
        </div>
        """, unsafe_allow_html=True)

        camera_photo = st.camera_input(
            label="Take a photo of the paddy leaf",
            label_visibility="collapsed",
        )

        if camera_photo:
            image        = Image.open(camera_photo)
            source_label = "📷 Live camera capture"
            st.markdown("""
            <div class="info-box">
                <b>✅ Photo Captured</b><br>
                <span style="font-size:0.85rem; color:#95d5b2;">
                    Camera snapshot ready for analysis
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-box" style="margin-top:8px;">
                <div style="font-size:3rem;">📸</div>
                <p style="color:#52b788; font-weight:600; margin:10px 0 4px;">
                    Camera preview appears above
                </p>
                <p style="color:#95d5b2; font-size:0.85rem;">
                    Allow camera access if prompted by your browser
                </p>
            </div>
            """, unsafe_allow_html=True)


# ── Results column ─────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 🔍 Diagnosis Results")

    if image is not None:
        with st.spinner("🧠 Analyzing image with AI..."):
            img_array   = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)
            pred_idx    = int(np.argmax(predictions[0]))
            pred_class  = class_names[str(pred_idx)]
            confidence  = float(predictions[0][pred_idx]) * 100

        is_paddy, reject_reason, max_conf, green_ratio = is_paddy_leaf(image, predictions)

        if not is_paddy:
            st.markdown(f"""
            <div class="warn-box">
                <h4 style="color:#e74c3c; margin:0 0 8px;">🚫 Not a Valid Paddy Leaf</h4>
                <p style="margin:0; font-size:0.9rem;">
                    This image was <b>rejected</b> because: <br>
                    <span style="color:#f39c12;">⚠ {reject_reason}</span>
                </p>
                <p style="margin:8px 0 0; font-size:0.85rem; opacity:0.85;">
                    This app is specifically designed for <b>paddy (rice) leaves only</b>.
                    Please upload a clear photo of a paddy leaf.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📌 What is a Paddy Leaf?")
            for point in [
                "Paddy is also known as **rice plant (Oryza sativa)**",
                "Leaves are **long, narrow and green** (may show yellow/brown patches if diseased)",
                "Grows in **wet or flooded agricultural fields**",
                "This app detects: **BacterialBlight, Blast, Brownspot, Healthy, Tungro**",
                "Ensure the leaf fills most of the frame with a plain background",
            ]:
                st.markdown(f"- {point}")

        else:
            info     = DISEASE_INFO[pred_class]
            severity = info["severity"]
            sc       = SEVERITY_COLORS[severity]

            # Result card
            st.markdown(f"""
            <div class="result-card" style="
                background:{sc['bg']};
                border-color:{sc['border']};">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                    <span style="font-size:2.5rem;">{info['emoji']}</span>
                    <div>
                        <h3 style="color:{sc['text']}; margin:0; font-size:1.6rem;">{pred_class}</h3>
                        <span class="badge" style="
                            background:{sc['border']}22;
                            color:{sc['text']};
                            border:1px solid {sc['border']};">
                            {severity} Severity
                        </span>
                    </div>
                </div>
                <div style="display:flex; gap:24px; margin-top:12px;">
                    <div>
                        <p style="color:#95d5b2; font-size:0.78rem; margin:0; text-transform:uppercase; letter-spacing:1px;">Confidence</p>
                        <p style="color:{sc['text']}; font-size:1.5rem; font-weight:700; margin:2px 0 0;">{confidence:.2f}%</p>
                    </div>
                    <div>
                        <p style="color:#95d5b2; font-size:0.78rem; margin:0; text-transform:uppercase; letter-spacing:1px;">Model Accuracy</p>
                        <p style="color:#52b788; font-size:1.5rem; font-weight:700; margin:2px 0 0;">{CLASS_ACCURACY[pred_class]}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart
            fig = confidence_bar_chart(class_names, predictions, pred_idx, sc["border"])
            st.plotly_chart(fig, use_container_width=True)

            # Disease detail tabs
            st.markdown("---")
            st.markdown("### 📚 Disease Details")
            tab1, tab2, tab3 = st.tabs(["📖 Description", "🔬 Symptoms", "💊 Treatment"])

            with tab1:
                st.markdown(f"**{pred_class}**")
                st.write(info["description"])

            with tab2:
                st.markdown("**Symptoms to look for:**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")

            with tab3:
                st.markdown("**Recommended Treatment:**")
                for t in info["treatment"]:
                    st.markdown(f"✅ {t}")

    else:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; height:350px;
            background:rgba(27,67,50,0.1);
            border:1px dashed #2d6a4f;
            border-radius:20px;
            color:#52b788;">
            <div style="font-size:4rem; margin-bottom:16px;">🔬</div>
            <p style="font-size:1.05rem; font-weight:600; margin:0;">Awaiting Image Input</p>
            <p style="font-size:0.85rem; color:#95d5b2; margin-top:8px;">
                Upload a photo or capture one with your camera
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    🌾 Paddy Disease Detection &nbsp;|&nbsp; MobileNetV2 Transfer Learning &nbsp;|&nbsp;
    Accuracy: 99.58% &nbsp;|&nbsp; Built with Streamlit + TensorFlow
</div>
""", unsafe_allow_html=True)

