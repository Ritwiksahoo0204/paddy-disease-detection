
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import hashlib                       # FIX #8: reliable image hashing
import os
from PIL import Image
from datetime import datetime        # FIX #11: timestamp formatting
import plotly.graph_objects as go
import database

# FIX #13: Single source-of-truth for accuracy string
OVERALL_ACCURACY = "99.58%"

# ── Page config ───────────────────────────────────────────────────────────────
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
        # FIX #10: theme-aware empty state box colours
        await_bg        = "rgba(27,67,50,0.1)"
        await_border    = "#2d6a4f"
        await_color     = "#52b788"
        await_sub       = "#95d5b2"

    elif mode == "☀️ Light":
        app_bg          = "linear-gradient(135deg,#f0fdf4 0%,#ecfdf5 50%,#f0fdf4 100%)"
        text_color      = "#000000"
        sidebar_bg      = "linear-gradient(180deg,#dcfce7 0%,#f0fdf4 100%)"
        sidebar_border  = "#86efac"
        sidebar_text    = "#000000"
        card_bg         = "linear-gradient(135deg,rgba(187,247,208,.7),rgba(209,250,229,.7))"
        card_border     = "#4ade80"
        card_h2         = "#000000"
        card_p          = "#000000"
        hero_grad       = "linear-gradient(90deg,#15803d,#22c55e,#15803d)"
        hero_sub        = "#000000"
        upload_border   = "#4ade80"
        upload_bg       = "rgba(187,247,208,0.3)"
        upload_hover    = "#15803d"
        info_bg         = "rgba(74,222,128,0.15)"
        info_border     = "#4ade80"
        warn_bg         = "rgba(239,68,68,0.1)"
        warn_border     = "#ef4444"
        tab_list_bg     = "rgba(187,247,208,0.5)"
        tab_text        = "#000000"
        tab_sel_bg      = "#4ade80"
        tab_sel_text    = "#000000"
        btn_bg          = "linear-gradient(135deg,#16a34a,#4ade80)"
        btn_shadow      = "rgba(74,222,128,0.45)"
        spinner_color   = "#16a34a"
        hr_color        = "#86efac"
        footer_color    = "#000000"
        theme_badge_bg  = "rgba(74,222,128,0.2)"
        theme_badge_txt = "#000000"
        # FIX #10: theme-aware empty state box colours (light)
        await_bg        = "rgba(187,247,208,0.2)"
        await_border    = "#4ade80"
        await_color     = "#16a34a"
        await_sub       = "#15803d"

    else:  # System default
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
  .await-box{background:rgba(27,67,50,0.1);border:1px dashed #2d6a4f;color:#52b788;}
  .await-box p.sub{color:#95d5b2;}
}
@media(prefers-color-scheme:light){
  .stApp{background:linear-gradient(135deg,#f0fdf4,#ecfdf5 50%,#f0fdf4);color:#000000;}
  [data-testid="stSidebar"]{background:linear-gradient(180deg,#dcfce7,#f0fdf4);border-right:1px solid #86efac;}
  [data-testid="stSidebar"] *{color:#000000!important;}
  .metric-card{background:linear-gradient(135deg,rgba(187,247,208,.7),rgba(209,250,229,.7));border:1px solid #4ade80;}
  .metric-card h2{color:#000000;} .metric-card p{color:#000000;}
  .hero-title h1{background:linear-gradient(90deg,#15803d,#22c55e,#15803d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
  .hero-title p{color:#000000;}
  .upload-box{border:2px dashed #4ade80;background:rgba(187,247,208,.3);}
  .upload-box:hover{border-color:#15803d;}
  .info-box{background:rgba(74,222,128,.15);border-left:4px solid #4ade80;}
  .warn-box{background:rgba(239,68,68,.1);border-left:4px solid #ef4444;}
  [data-baseweb="tab-list"]{background:rgba(187,247,208,.5)!important;}
  [data-baseweb="tab"]{color:#000000!important;}
  [aria-selected="true"]{background:#4ade80!important;color:#000000!important;}
  .stButton>button{background:linear-gradient(135deg,#16a34a,#4ade80);color:#fff;}
  hr{border-color:#86efac!important;} .footer{color:#000000;}
  .await-box{background:rgba(187,247,208,0.2);border:1px dashed #4ade80;color:#16a34a;}
  .await-box p.sub{color:#15803d;}
}
"""

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
.await-box{{background:{await_bg};border:1px dashed {await_border};color:{await_color};}}
.await-box p.sub{{color:{await_sub};}}
"""


# ── Apply CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>{get_theme_css(st.session_state['theme'])}</style>",
    unsafe_allow_html=True,
)

# ── Auth Check ────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None


def auth_ui():
    is_dark = st.session_state["theme"] in ["🌑 Dark", "🖥️ System Default"]

    bg_color  = "linear-gradient(135deg, rgba(13,33,55,0.7), rgba(10,26,14,0.8))" if is_dark else "linear-gradient(135deg, rgba(255,255,255,0.85), rgba(240,253,244,0.9))"
    border    = "rgba(82,183,136,0.4)" if is_dark else "rgba(74,222,128,0.5)"
    shadow    = "rgba(0,0,0,0.5)" if is_dark else "rgba(21,128,61,0.15)"
    text_grad = "linear-gradient(90deg, #52b788, #95d5b2, #52b788)" if is_dark else "linear-gradient(90deg, #15803d, #22c55e, #15803d)"
    sub_color = "#95d5b2" if is_dark else "#000000"
    input_bg  = "rgba(0,0,0,0.2)" if is_dark else "rgba(255,255,255,0.6)"

    st.markdown(f"""
    <style>
    .auth-card {{
        background: {bg_color};
        border: 1px solid {border};
        border-radius: 24px;
        padding: 50px 40px;
        box-shadow: 0 15px 50px {shadow};
        backdrop-filter: blur(20px);
        margin-top: 5vh;
        margin-bottom: 5vh;
    }}
    .auth-title {{
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: {text_grad};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }}
    .auth-subtitle {{
        text-align: center;
        color: {sub_color};
        font-size: 1.05rem;
        margin-bottom: 35px;
        opacity: 0.85;
    }}
    div[data-baseweb="input"] {{
        background-color: {input_bg} !important;
        border-radius: 12px !important;
        border: 1px solid {border} !important;
    }}
    div[data-baseweb="input"]:focus-within {{
        border: 2px solid #52b788 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-bottom:28px;">
            <div style="display:flex; justify-content:center; margin-bottom:16px;">
                <div style="width:82px; height:82px; background:linear-gradient(135deg,#1b4332,#2d6a4f);
                            border-radius:50%; border:2px solid rgba(82,183,136,0.5);
                            display:flex; align-items:center; justify-content:center;">
                    <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <line x1="23" y1="42" x2="23" y2="10" stroke="#52b788" stroke-width="2.2" stroke-linecap="round"/>
                        <path d="M23 32 Q12 28 10 20 Q18 22 23 28" fill="#2d6a4f" stroke="#52b788" stroke-width="1.2"/>
                        <path d="M23 24 Q11 18 10 10 Q19 14 23 20" fill="#40916c" stroke="#74c69d" stroke-width="1"/>
                        <path d="M23 36 Q34 30 36 22 Q28 25 23 31" fill="#2d6a4f" stroke="#52b788" stroke-width="1.2"/>
                        <path d="M23 27 Q35 20 36 11 Q27 16 23 22" fill="#40916c" stroke="#74c69d" stroke-width="1"/>
                        <ellipse cx="23" cy="8" rx="3.5" ry="5" fill="#95d5b2" stroke="#52b788" stroke-width="1"/>
                        <ellipse cx="19" cy="10" rx="2.5" ry="4" fill="#74c69d" stroke="#52b788" stroke-width="0.8"/>
                        <ellipse cx="27" cy="10" rx="2.5" ry="4" fill="#74c69d" stroke="#52b788" stroke-width="0.8"/>
                        <ellipse cx="16" cy="13" rx="2" ry="3.5" fill="#52b788" stroke="#40916c" stroke-width="0.8"/>
                        <ellipse cx="30" cy="13" rx="2" ry="3.5" fill="#52b788" stroke="#40916c" stroke-width="0.8"/>
                    </svg>
                </div>
            </div>
            <div class='auth-title'>Paddy AI</div>
            <div class='auth-subtitle'>Sign in to your intelligent farming portal</div>
        </div>
        """, unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["✨ Log In", "🚀 Sign Up", "🔐 Reset"])

        with t1:
            st.markdown("<br>", unsafe_allow_html=True)
            login_user = st.text_input("Username", key="login_user", placeholder="Enter your username")
            login_pass = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Access Dashboard", use_container_width=True):
                success, uid = database.authenticate_user(login_user, login_pass)
                if success:
                    st.session_state["logged_in"] = True
                    st.session_state["username"]   = login_user
                    st.session_state["user_id"]    = uid
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with t2:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_user      = st.text_input("New Username", key="reg_user", placeholder="Choose a username (3–20 chars)")
            reg_pass      = st.text_input("New Password", type="password", key="reg_pass", placeholder="Min 6 characters")
            reg_pass_conf = st.text_input("Confirm Password", type="password", key="reg_pass_conf", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", use_container_width=True):
                if not reg_user or not reg_pass:
                    st.error("Please fill in all fields.")
                elif reg_pass != reg_pass_conf:
                    st.error("Passwords do not match.")
                else:
                    # FIX #6 & #7: Validation is done inside database.create_user()
                    success, msg = database.create_user(reg_user, reg_pass)
                    if success:
                        _, uid = database.authenticate_user(reg_user, reg_pass)
                        st.session_state["logged_in"] = True
                        st.session_state["username"]   = reg_user
                        st.session_state["user_id"]    = uid
                        st.rerun()
                    else:
                        st.error(msg)

        with t3:
            # FIX #2: Password reset now requires the current/old password
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("ℹ️ You must enter your **current password** to reset it.")
            reset_user     = st.text_input("Your Username", key="reset_user", placeholder="Enter your username")
            reset_old_pass = st.text_input("Current Password", type="password", key="reset_old_pass", placeholder="Your current password")
            reset_pass     = st.text_input("New Password", type="password", key="reset_pass", placeholder="Min 6 characters")
            reset_pass_conf = st.text_input("Confirm New Password", type="password", key="reset_pass_conf", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset Password", use_container_width=True):
                if not reset_user or not reset_old_pass or not reset_pass:
                    st.error("Please fill in all fields.")
                elif reset_pass != reset_pass_conf:
                    st.error("Passwords do not match.")
                else:
                    success, msg = database.update_password(reset_user, reset_old_pass, reset_pass)
                    if success:
                        st.success(msg + " You can now log in with your new password.")
                    else:
                        st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state["logged_in"]:
    auth_ui()
    st.stop()


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

@st.cache_resource(show_spinner=False)
def load_validator():
    if os.path.exists("paddy_validator.keras"):
        try:
            return tf.keras.models.load_model("paddy_validator.keras")
        except Exception as e:
            st.error(f"Error loading validator model: {e}")
            return None
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def is_paddy_leaf(
    image: Image.Image,
    predictions: np.ndarray,
    validator_model,
    conf_threshold: float = 80.0,
):
    max_conf = float(np.max(predictions[0])) * 100

    if validator_model is not None:
        img_array = preprocess_image(image)
        prob = float(validator_model.predict(img_array, verbose=0)[0][0])
        if prob < 0.6:
            return False, f"AI Validator determined image is NOT a paddy leaf (Confidence: {(1-prob)*100:.1f}%)", max_conf, prob

    probs = predictions[0].astype(np.float64)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(len(probs)))
    entropy_ratio = entropy / max_entropy
    entropy_ok = entropy_ratio < 0.85
    conf_ok = max_conf >= conf_threshold
    is_valid = conf_ok and entropy_ok

    if is_valid:
        reason = "OK"
    else:
        parts = []
        if not conf_ok:
            parts.append(f"low disease model confidence ({max_conf:.1f}%, need ≥{conf_threshold}%)")
        if not entropy_ok:
            parts.append(f"disease model is uncertain (entropy {entropy_ratio*100:.0f}% of max)")
        reason = " · ".join(parts)

    return is_valid, reason, max_conf, 1.0


def confidence_bar_chart(class_names: dict, predictions: np.ndarray, pred_idx: int, sev_color: str):
    is_dark = st.session_state["theme"] in ["🌑 Dark", "🖥️ System Default"]
    text_color  = "#b7e4c7" if is_dark else "#000000"
    title_color = "#95d5b2" if is_dark else "#000000"
    grid_color  = "rgba(82,183,136,0.1)" if is_dark else "rgba(21,128,61,0.15)"

    labels = [class_names[str(i)] for i in range(len(class_names))]
    values = [float(p) * 100 for p in predictions[0]]
    colors = [sev_color if i == pred_idx else ("rgba(82,183,136,0.25)" if is_dark else "rgba(74,222,128,0.4)") for i in range(len(class_names))]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)",
        marker_line_width=1,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=text_color, size=11),
    ))
    fig.update_layout(
        title=dict(text="Confidence Score per Class", font=dict(color=title_color, size=14)),
        xaxis=dict(tickfont=dict(color=text_color), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color=text_color), gridcolor=grid_color, range=[0, 115], zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=10, r=10),
        height=280,
        showlegend=False,
    )
    return fig


# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("🌱 Loading AI models..."):
    model, class_names = load_model()
    validator_model = load_validator()

# FIX #14: Warn user if validator is unavailable
if validator_model is None:
    st.sidebar.warning("⚠️ Validator model unavailable. AI leaf check is disabled.")


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
    # FIX #13: Using constant for accuracy
    st.markdown(f'<div class="metric-card"><h2>{OVERALL_ACCURACY}</h2><p>Overall Accuracy</p></div>', unsafe_allow_html=True)
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
    st.markdown(f"**👤 Hello, {st.session_state['username']}**")

    if st.button("Log Out", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"]  = None
        st.session_state["user_id"]   = None
        st.session_state["theme"]     = "🌑 Dark"
        st.rerun()

    st.markdown("---")

    # FIX #9: Confidence threshold slider in sidebar
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=95,
        value=80,
        step=5,
        help="Images predicted below this confidence will be rejected as uncertain.",
    )
    st.markdown("---")

    # FIX #3: Admin check uses env variable, not hardcoded string
    _admin_user = os.environ.get("ADMIN_USERNAME", "admin")
    nav_options = ["🔍 Detect Disease", "📖 My History"]
    if st.session_state.get("username") == _admin_user:
        nav_options.append("👑 Admin Panel")

    app_page = st.radio("Navigation", nav_options, label_visibility="collapsed")
    st.markdown("---")

    # Theme switcher
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

    _badge_icons = {"🌑 Dark": "🌑", "☀️ Light": "☀️", "🖥️ System Default": "🖥️"}
    st.markdown(
        f'<div class="theme-badge">{_badge_icons[st.session_state["theme"]]} '
        f'{st.session_state["theme"].split(" ",1)[1]} mode active</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📋 Model Details")
    for k, v in [("Architecture", "MobileNetV2"), ("Backbone", "ImageNet"),
                 ("Accuracy", OVERALL_ACCURACY), ("Framework", "TensorFlow"),  # FIX #13
                 ("Input Size", "224 × 224")]:
        st.markdown(f"**{k}:** {v}")

    st.markdown("---")
    st.markdown("### 🎯 Class Accuracy")
    for cls, info in DISEASE_INFO.items():
        acc = CLASS_ACCURACY[cls]
        st.markdown(f"{info['emoji']} **{cls}** — **{acc}**")

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    for i, step in enumerate(["Upload a **paddy leaf** image",
                               "Adjust confidence threshold if needed",
                               "Wait for AI analysis",
                               "Review disease diagnosis",
                               "Follow treatment steps"], 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("---")
    st.warning("⚠️ **Paddy leaves only.** Other images will be rejected automatically.")


# ── History Page ──────────────────────────────────────────────────────────────
if app_page == "📖 My History":
    st.markdown("### 📋 Your Recent Scans")
    history = database.get_user_activity(st.session_state["user_id"])
    if not history:
        st.info("You haven't scanned any images yet.")
    else:
        for entry in history:
            # FIX #11: Format timestamp nicely
            try:
                ts = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")
                formatted_ts = ts.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                formatted_ts = entry['timestamp']

            st.markdown(f"""
            <div class="result-card" style="border-color:#52b788; margin-bottom:12px;">
                <h4 style="color:#52b788; margin:0;">{entry['predicted_class']} ({entry['confidence']:.1f}%)</h4>
                <p style="margin:4px 0 0; font-size:0.85rem; color:#95d5b2;">
                    Severity: <b>{entry['severity']}</b> &nbsp;|&nbsp;
                    Date: {formatted_ts} &nbsp;|&nbsp;
                    Image: {entry['image_name']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ── Admin Panel ───────────────────────────────────────────────────────────────
if app_page == "👑 Admin Panel":
    # FIX #3: Single consolidated admin check using env variable
    _admin_user = os.environ.get("ADMIN_USERNAME", "admin")
    if st.session_state["username"] != _admin_user:
        st.error("⛔ Access Denied. You do not have admin privileges.")
        st.stop()

    st.markdown("### 👑 User Database (Admin)")

    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(database.DB_PATH)

    st.markdown("#### 👤 Registered Users")
    users_df = pd.read_sql_query("SELECT id, username, created_at, last_login FROM users", conn)
    st.dataframe(users_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📈 Global Activity Log")
    activity_df = pd.read_sql_query(
        "SELECT user_id, image_name, predicted_class, confidence, timestamp FROM activity ORDER BY timestamp DESC LIMIT 200",
        conn
    )
    st.dataframe(activity_df, use_container_width=True, hide_index=True)
    conn.close()

    if os.path.exists(database.DB_PATH):
        with open(database.DB_PATH, "rb") as file:
            st.download_button(
                label="💾 Download Raw Database File (.db)",
                data=file,
                file_name="paddy_app.db",
                mime="application/octet-stream"
            )

    st.markdown("---")
    st.markdown("#### 🔑 Force Reset a User's Password")
    with st.expander("Admin Password Reset (no old password required)"):
        st.warning("⚠️ Admin-only. This bypasses old-password verification.")
        admin_reset_user = st.text_input("Target Username", key="admin_reset_user")
        admin_reset_pass = st.text_input("New Password", type="password", key="admin_reset_pass")
        if st.button("Force Reset Password", type="primary"):
            if not admin_reset_user or not admin_reset_pass:
                st.error("Please fill in both fields.")
            else:
                # FIX #2: Use the dedicated admin-only reset function
                success, msg = database.admin_force_reset_password(admin_reset_user, admin_reset_pass)
                if success:
                    st.success(f"Successfully changed password for user: {admin_reset_user}")
                else:
                    st.error(msg)

    st.stop()


# ── Main Layout ───────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

image        = None
source_label = ""

with col_upload:
    st.markdown("### 📷 Image Input")

    tab_upload, tab_camera = st.tabs(["📤 Upload Photo", "📷 Capture Photo"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a paddy leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit image of a paddy leaf",
            label_visibility="collapsed",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            source_label = f"📁 {uploaded_file.name} &nbsp;|&nbsp; {uploaded_file.size // 1024} KB"
            # FIX #5: use_container_width replaces deprecated use_column_width
            st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)
            st.markdown(f"""
            <div class="info-box">
                <b>✅ Image Loaded</b><br>
                <span style="font-size:0.85rem; color:#95d5b2;">{source_label}</span>
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

    with tab_camera:
        st.markdown("""
        <div class="info-box" style="margin-bottom:12px;">
            <b>📷 Live Camera Capture</b><br>
            <span style="font-size:0.85rem;">
                Point your camera at a <b>paddy leaf</b> and click the shutter button below.
            </span>
        </div>
        """, unsafe_allow_html=True)

        camera_photo = st.camera_input(
            label="Take a photo of the paddy leaf",
            label_visibility="collapsed",
        )

        if camera_photo:
            image = Image.open(camera_photo)
            source_label = "📷 Live camera capture"
            st.markdown("""
            <div class="info-box">
                <b>✅ Photo Captured</b><br>
                <span style="font-size:0.85rem; color:#95d5b2;">Camera snapshot ready for analysis</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-box" style="margin-top:8px;">
                <div style="font-size:3rem;">📸</div>
                <p style="color:#52b788; font-weight:600; margin:10px 0 4px;">Camera preview appears above</p>
                <p style="color:#95d5b2; font-size:0.85rem;">Allow camera access if prompted by your browser</p>
            </div>
            """, unsafe_allow_html=True)


# ── Results ───────────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 🔍 Diagnosis Results")

    if image is not None:
        with st.spinner("🧠 Analyzing image with AI..."):
            img_array  = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)
            pred_idx   = int(np.argmax(predictions[0]))
            pred_class = class_names[str(pred_idx)]
            confidence = float(predictions[0][pred_idx]) * 100

            is_paddy, reject_reason, max_conf, _ = is_paddy_leaf(
                image, predictions, validator_model, conf_threshold
            )

        if not is_paddy:
            st.markdown(f"""
            <div class="warn-box">
                <h4 style="color:#e74c3c; margin:0 0 8px;">🚫 Not a Valid Paddy Leaf</h4>
                <p style="margin:0; font-size:0.9rem;">
                    This image was <b>rejected</b> because:<br>
                    <span style="color:#f39c12;">⚠ {reject_reason}</span>
                </p>
                <p style="margin:8px 0 0; font-size:0.85rem; opacity:0.85;">
                    This app is specifically designed for <b>paddy (rice) leaves only</b>.
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

            # FIX #8: Use MD5 instead of Python's hash() for reliable deduplication
            current_image_hash = hashlib.md5(image.tobytes()).hexdigest()
            if st.session_state.get("last_recorded_img") != current_image_hash:
                img_name_to_save = (
                    source_label.split(" |")[0].replace("📁 ", "").strip()
                    if "📁" in source_label else "Camera Snapshot"
                )
                database.record_activity(
                    st.session_state["user_id"],
                    img_name_to_save,
                    pred_class,
                    confidence,
                    severity
                )
                st.session_state["last_recorded_img"] = current_image_hash

            st.markdown(f"""
            <div class="result-card" style="background:{sc['bg']};border-color:{sc['border']};">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                    <span style="font-size:2.5rem;">{info['emoji']}</span>
                    <div>
                        <h3 style="color:{sc['text']};margin:0;font-size:1.6rem;">{pred_class}</h3>
                        <span class="badge" style="background:{sc['border']}22;color:{sc['text']};border:1px solid {sc['border']};">
                            {severity} Severity
                        </span>
                    </div>
                </div>
                <div style="display:flex;gap:24px;margin-top:12px;">
                    <div>
                        <p style="color:#95d5b2;font-size:0.78rem;margin:0;text-transform:uppercase;letter-spacing:1px;">Confidence</p>
                        <p style="color:{sc['text']};font-size:1.5rem;font-weight:700;margin:2px 0 0;">{confidence:.2f}%</p>
                    </div>
                    <div>
                        <p style="color:#95d5b2;font-size:0.78rem;margin:0;text-transform:uppercase;letter-spacing:1px;">Model Accuracy</p>
                        <p style="color:#52b788;font-size:1.5rem;font-weight:700;margin:2px 0 0;">{CLASS_ACCURACY[pred_class]}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = confidence_bar_chart(class_names, predictions, pred_idx, sc["border"])
            st.plotly_chart(fig, use_container_width=True)

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
        # FIX #10: Theme-aware empty state using CSS class (not hardcoded dark colours)
        st.markdown("""
        <div class="await-box" style="
            display:flex;flex-direction:column;align-items:center;
            justify-content:center;height:350px;
            border-radius:20px;">
            <div style="font-size:4rem;margin-bottom:16px;">🔬</div>
            <p style="font-size:1.05rem;font-weight:600;margin:0;">Awaiting Image Input</p>
            <p class="sub" style="font-size:0.85rem;margin-top:8px;">
                Upload a photo or capture one with your camera
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div class="footer">
    🌾 Paddy Disease Detection &nbsp;|&nbsp; MobileNetV2 Transfer Learning &nbsp;|&nbsp;
    Accuracy: {OVERALL_ACCURACY} &nbsp;|&nbsp; Built with Streamlit + TensorFlow
</div>
""", unsafe_allow_html=True)

