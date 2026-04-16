
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title = "Paddy Disease Detection",
    page_icon  = "🌾",
    layout     = "wide"
)

@st.cache_resource
def load_models():
    disease_model = tf.keras.models.load_model("best_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    try:
        validator = tf.keras.models.load_model("paddy_validator.keras")
        validator_loaded = True
    except:
        validator = None
        validator_loaded = False
    return disease_model, class_names, validator, validator_loaded

disease_model, class_names, validator, validator_loaded = load_models()

CONFIDENCE_THRESHOLD = 0.70
VALIDATOR_THRESHOLD  = 0.60

DISEASE_INFO = {
    "Bacterialblight": {
        "description" : "Bacterial Leaf Blight caused by Xanthomonas oryzae pv. oryzae. Causes wilting and yellowing of leaves.",
        "symptoms"    : ["Water-soaked lesions on leaf edges", "Yellowing and drying of leaves", "Wilting of seedlings"],
        "treatment"   : ["Use resistant varieties", "Apply copper-based bactericides", "Avoid excess nitrogen fertilizer", "Drain fields during outbreak"],
        "severity"    : "High"
    },
    "Blast": {
        "description" : "Rice Blast caused by Magnaporthe oryzae. One of the most destructive rice diseases worldwide.",
        "symptoms"    : ["Diamond-shaped lesions on leaves", "Gray centers with brown borders", "White to gray panicles"],
        "treatment"   : ["Apply Tricyclazole fungicide", "Use blast-resistant varieties", "Avoid excessive nitrogen", "Proper field drainage"],
        "severity"    : "Very High"
    },
    "Brownspot": {
        "description" : "Brown Spot caused by Cochliobolus miyabeanus fungus. Affects leaves, sheaths and grains.",
        "symptoms"    : ["Oval brown spots on leaves", "Dark brown borders with yellow halo", "Spots on sheaths and grains"],
        "treatment"   : ["Apply Mancozeb or Iprodione", "Use healthy certified seeds", "Maintain soil nutrition", "Treat seeds before planting"],
        "severity"    : "Medium"
    },
    "Healthy": {
        "description" : "No disease detected. The paddy plant appears healthy.",
        "symptoms"    : ["No visible lesions", "Normal green color", "Healthy leaf structure"],
        "treatment"   : ["Continue current practices", "Monitor regularly", "Maintain proper irrigation"],
        "severity"    : "None"
    },
    "Tungro": {
        "description" : "Rice Tungro Disease caused by two viruses transmitted by green leafhopper insects.",
        "symptoms"    : ["Yellow to orange discoloration", "Stunted plant growth", "Reduced tillering"],
        "treatment"   : ["Control leafhopper population", "Use tungro-resistant varieties", "Apply insecticides early", "Remove infected plants"],
        "severity"    : "Very High"
    }
}

SEVERITY_COLOR = {
    "None"     : "#27ae60",
    "Medium"   : "#f39c12",
    "High"     : "#e67e22",
    "Very High": "#e74c3c"
}

def check_image_quality(image):
    arr        = np.array(image.convert("RGB"))
    gray       = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    std        = np.std(arr)
    if blur < 50:
        return False, "Image is too blurry. Please upload a clearer image."
    if brightness < 20:
        return False, "Image is too dark. Please upload a brighter image."
    if brightness > 240:
        return False, "Image is overexposed. Please use proper lighting."
    if std < 15:
        return False, "Image appears blank or single-colored."
    return True, "OK"

def preprocess(image):
    image = image.convert("RGB").resize((224, 224))
    arr   = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def show_rejection(title, message):
    st.error(f"🚫 {title}")
    st.markdown(f"""
    <div style="background:#fde8e8; border-left:5px solid #e74c3c;
                padding:20px; border-radius:8px; margin-top:10px;">
        <p style="margin:0; font-size:15px; color:#333;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ---
    ### 📌 Reminder
    This app is designed **only for paddy (rice) leaf images**.
    - Paddy leaves are **long, narrow and green**
    - From the rice plant (*Oryza sativa*)
    - Grown in **wet or flooded** agricultural fields
    - This app detects: **Bacterialblight, Blast, Brownspot, Healthy, Tungro**
    """)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<h1 style="text-align:center; color:#27ae60;">🌾 Paddy Disease Detection</h1>
<p style="text-align:center; color:gray; font-size:16px;">
    Upload a paddy leaf image to detect disease using deep learning
</p><hr>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Model Info")
    st.markdown("""
    | Detail | Info |
    |--------|------|
    | Model | MobileNetV2 |
    | Accuracy | 99.58% |
    | Classes | 5 |
    | Min Confidence | 70% |
    """)
    if validator_loaded:
        st.success("✅ Paddy validator active")
    else:
        st.warning("⚠️ Running without validator")
    st.markdown("---")
    st.markdown("### 🌿 Detectable Classes")
    for cls, acc in {
        "🦠 Bacterialblight": "99.37%",
        "💥 Blast"          : "98.61%",
        "🟤 Brownspot"      : "100.0%",
        "✅ Healthy"        : "100.0%",
        "🔴 Tungro"         : "100.0%"
    }.items():
        st.markdown(f"**{cls}** — {acc}")
    st.markdown("---")
    st.warning("⚠️ **Paddy leaves only.**\nOther images will be rejected automatically.")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Upload a **clear paddy leaf** image
    2. System checks image quality
    3. System verifies it is a paddy leaf
    4. Disease prediction is shown
    5. Follow the treatment advice
    """)

# ── Main layout ──────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Paddy Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a paddy leaf image",
        type = ["jpg", "jpeg", "png"],
        help = "Upload a clear paddy (rice) leaf image only"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"✅ Image uploaded: {uploaded_file.name}")

with col2:
    st.markdown("### 🔍 Prediction Results")

    if uploaded_file:
        with st.spinner("Analyzing image... please wait"):
            img_array = preprocess(image)

            # ── Step 1: Image quality check ──────────────
            quality_ok, quality_msg = check_image_quality(image)
            if not quality_ok:
                show_rejection("Image Quality Issue", quality_msg)
                st.stop()

            # ── Step 2: Paddy leaf validation ─────────────
            if validator is not None:
                paddy_prob = float(
                    validator.predict(img_array, verbose=0)[0][0]
                )
                if paddy_prob < VALIDATOR_THRESHOLD:
                    show_rejection(
                        "Not a Paddy Leaf Detected!",
                        f"This does not appear to be a paddy leaf "
                        f"(paddy confidence: {paddy_prob*100:.1f}%). "
                        f"This app only works with paddy (rice) leaf images. "
                        f"Please upload the correct image and try again."
                    )
                    st.stop()
            else:
                # Fallback if validator not loaded
                preds_check = disease_model.predict(img_array, verbose=0)
                if float(np.max(preds_check[0])) < 0.60:
                    show_rejection(
                        "Not a Paddy Leaf Detected!",
                        "This does not appear to be a paddy leaf. "
                        "Please upload a clear paddy (rice) leaf image."
                    )
                    st.stop()

            # ── Step 3: Disease prediction ────────────────
            preds      = disease_model.predict(img_array, verbose=0)
            pred_idx   = int(np.argmax(preds[0]))
            confidence = float(preds[0][pred_idx])
            pred_class = class_names[str(pred_idx)]

            # ── Step 4: Confidence threshold ─────────────
            if confidence < CONFIDENCE_THRESHOLD:
                show_rejection(
                    "Low Confidence — Cannot Predict",
                    f"The model is only {confidence*100:.1f}% confident, "
                    f"which is below the minimum threshold of "
                    f"{CONFIDENCE_THRESHOLD*100:.0f}%. "
                    f"Please upload a clearer, well-lit paddy leaf image."
                )
                st.stop()

        # ── Show result ───────────────────────────────────
        sev      = DISEASE_INFO[pred_class]["severity"]
        sev_col  = SEVERITY_COLOR[sev]

        st.markdown(f"""
        <div style="background:#f8f9fa; border-left:5px solid {sev_col};
                    padding:18px; border-radius:8px; margin-bottom:15px;">
            <h3 style="color:{sev_col}; margin:0;">{pred_class}</h3>
            <p style="margin:8px 0; font-size:18px;">
                Confidence : <strong>{confidence*100:.2f}%</strong>
            </p>
            <p style="margin:0;">
                Severity : <strong style="color:{sev_col};">{sev}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar chart
        fig = go.Figure(go.Bar(
            x = list(class_names.values()),
            y = [float(p)*100 for p in preds[0]],
            marker_color = [
                sev_col if class_names[str(i)] == pred_class
                else "#bdc3c7"
                for i in range(len(class_names))
            ]
        ))
        fig.update_layout(
            title       = "Confidence Score per Class (%)",
            xaxis_title = "Disease Class",
            yaxis_title = "Confidence (%)",
            height      = 280,
            margin      = dict(t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Upload a paddy leaf image to see prediction results")
        st.markdown("""
        <div style="background:#eafaf1; border-left:5px solid #27ae60;
                    padding:15px; border-radius:5px;">
            <p style="margin:0; font-weight:bold; color:#27ae60;">⚠️ Important Notice</p>
            <p style="margin:8px 0; font-size:14px;">
                This app works with <strong>paddy (rice) leaf images only</strong>.
                Uploading other plant leaves, faces, or unrelated images
                will be automatically rejected.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Disease details section ───────────────────────────────────
if uploaded_file and "pred_class" in dir():
    st.markdown("---")
    st.markdown("### 📚 Disease Details")
    info = DISEASE_INFO[pred_class]
    tab1, tab2, tab3 = st.tabs(["📖 Description", "🔬 Symptoms", "💊 Treatment"])
    with tab1:
        st.write(info["description"])
    with tab2:
        for s in info["symptoms"]:
            st.markdown(f"- {s}")
    with tab3:
        for t in info["treatment"]:
            st.markdown(f"✅ {t}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:gray; font-size:13px;">
Paddy Disease Detection | MobileNetV2 | Accuracy: 99.58% | Built with Streamlit
</p>
""", unsafe_allow_html=True)
