
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title = "Paddy Disease Detection",
    page_icon  = "🌾",
    layout     = "wide"
)

# Load model & class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# Valid paddy leaf keywords
PADDY_KEYWORDS = [
    "paddy", "rice", "oryza", "leaf", "padi",
    "bacterialblight", "blast", "brownspot",
    "healthy", "tungro"
]

# Disease info
DISEASE_INFO = {
    "Bacterialblight": {
        "description" : "Bacterial Leaf Blight is caused by Xanthomonas oryzae pv. oryzae. It causes wilting and yellowing of leaves.",
        "symptoms"    : [
            "Water-soaked lesions on leaf edges",
            "Yellowing and drying of leaves",
            "Wilting of seedlings"
        ],
        "treatment"   : [
            "Use resistant varieties",
            "Apply copper-based bactericides",
            "Avoid excess nitrogen fertilizer",
            "Drain fields during outbreak"
        ],
        "severity"    : "High"
    },
    "Blast": {
        "description" : "Rice Blast is caused by Magnaporthe oryzae fungus. It is one of the most destructive rice diseases worldwide.",
        "symptoms"    : [
            "Diamond-shaped lesions on leaves",
            "Gray centers with brown borders",
            "White to gray panicles"
        ],
        "treatment"   : [
            "Apply fungicides like Tricyclazole",
            "Use blast-resistant varieties",
            "Avoid excessive nitrogen",
            "Ensure proper field drainage"
        ],
        "severity"    : "Very High"
    },
    "Brownspot": {
        "description" : "Brown Spot is caused by Cochliobolus miyabeanus fungus. It affects leaves, sheaths and grains.",
        "symptoms"    : [
            "Oval brown spots on leaves",
            "Dark brown borders with yellow halo",
            "Spots on leaf sheaths and grains"
        ],
        "treatment"   : [
            "Apply Mancozeb or Iprodione fungicide",
            "Use healthy certified seeds",
            "Maintain proper soil nutrition",
            "Treat seeds before planting"
        ],
        "severity"    : "Medium"
    },
    "Healthy": {
        "description" : "The plant appears healthy with no visible signs of disease.",
        "symptoms"    : [
            "No visible lesions",
            "Normal green color",
            "Healthy leaf structure"
        ],
        "treatment"   : [
            "Continue current farming practices",
            "Monitor regularly for early detection",
            "Maintain proper irrigation and nutrition"
        ],
        "severity"    : "None"
    },
    "Tungro": {
        "description" : "Rice Tungro Disease is caused by two viruses transmitted by green leafhopper insects.",
        "symptoms"    : [
            "Yellow to orange discoloration of leaves",
            "Stunted plant growth",
            "Reduced tillering"
        ],
        "treatment"   : [
            "Control leafhopper population",
            "Use tungro-resistant varieties",
            "Apply insecticides early",
            "Remove and destroy infected plants"
        ],
        "severity"    : "Very High"
    }
}

SEVERITY_COLOR = {
    "None"      : "#27ae60",
    "Medium"    : "#f39c12",
    "High"      : "#e67e22",
    "Very High" : "#e74c3c"
}

# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Check if image is likely a paddy leaf using model confidence
def is_paddy_leaf(predictions):
    max_confidence = float(np.max(predictions[0])) * 100
    # If max confidence is below 60% it is likely not a paddy leaf
    return max_confidence >= 60.0, max_confidence

# Header
st.markdown("""
    <h1 style="text-align:center; color:#27ae60;">
        🌾 Paddy Disease Detection
    </h1>
    <p style="text-align:center; color:gray; font-size:16px;">
        Upload a paddy leaf image to detect disease using MobileNetV2 deep learning model
    </p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Model Info")
    st.markdown("""
    | Detail | Info |
    |--------|------|
    | Architecture | MobileNetV2 |
    | Accuracy | 99.58% |
    | Classes | 5 |
    | Input Size | 224x224 |
    | Framework | TensorFlow |
    """)

    st.markdown("---")
    st.markdown("### 🌿 Detectable Disease Classes")
    classes_display = {
        "🦠 Bacterialblight" : "99.37%",
        "💥 Blast"           : "98.61%",
        "🟤 Brownspot"       : "100.0%",
        "✅ Healthy"         : "100.0%",
        "🔴 Tungro"          : "100.0%"
    }
    for cls, acc in classes_display.items():
        st.markdown(f"**{cls}** - {acc}")

    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Upload a **paddy leaf** image only
    2. Wait for prediction
    3. View disease details
    4. Follow treatment advice
    """)

    st.markdown("---")
    st.warning("⚠️ This app is designed for **paddy leaves only**. Other plant leaves will not be recognized.")

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Paddy Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a paddy leaf image",
        type = ["jpg", "jpeg", "png"],
        help = "Upload a clear image of a paddy leaf only"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Image uploaded : {uploaded_file.name}")

with col2:
    st.markdown("### 🔍 Prediction Results")

    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            img_array   = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)
            pred_idx    = int(np.argmax(predictions[0]))
            pred_class  = class_names[str(pred_idx)]
            confidence  = float(predictions[0][pred_idx]) * 100

        # Check if it is a paddy leaf
        is_paddy, max_conf = is_paddy_leaf(predictions)

        if not is_paddy:
            # Not a paddy leaf — show warning
            st.error("🚫 Non-Paddy Leaf Detected!")
            st.markdown("""
            <div style="
                background-color: #fde8e8;
                border-left: 5px solid #e74c3c;
                padding: 20px;
                border-radius: 8px;">
                <h4 style="color:#e74c3c; margin:0;">
                    ⚠️ This does not appear to be a paddy leaf!
                </h4>
                <br>
                <p style="margin:0; font-size:15px;">
                    This app is specifically designed to detect diseases
                    in <strong>paddy (rice) leaves only</strong>.
                </p>
                <br>
                <p style="margin:0; font-size:14px; color:gray;">
                    Please upload a clear image of a paddy leaf and try again.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📌 What is a Paddy Leaf?")
            st.markdown("""
            - Paddy is also known as **rice plant (Oryza sativa)**
            - Leaves are **long, narrow and green**
            - They grow in **wet/flooded agricultural fields**
            - This app detects 5 conditions: **Bacterialblight, Blast, Brownspot, Healthy, Tungro**
            """)

        else:
            # Valid paddy leaf — show results
            severity     = DISEASE_INFO[pred_class]["severity"]
            severity_col = SEVERITY_COLOR[severity]

            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border-left: 5px solid {severity_col};
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;">
                <h3 style="color:{severity_col}; margin:0;">
                    {pred_class}
                </h3>
                <p style="margin:5px 0; font-size:18px;">
                    Confidence : <strong>{confidence:.2f}%</strong>
                </p>
                <p style="margin:5px 0;">
                    Severity : <strong style="color:{severity_col};">{severity}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar chart
            fig = go.Figure(go.Bar(
                x = [class_names[str(i)] for i in range(len(class_names))],
                y = [float(p) * 100 for p in predictions[0]],
                marker_color = [
                    severity_col if i == pred_idx else "#bdc3c7"
                    for i in range(len(class_names))
                ]
            ))
            fig.update_layout(
                title       = "Confidence Score per Class (%)",
                xaxis_title = "Disease Class",
                yaxis_title = "Confidence (%)",
                height      = 300,
                margin      = dict(t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Disease details
            st.markdown("---")
            st.markdown("### 📚 Disease Details")
            info = DISEASE_INFO[pred_class]
            tab1, tab2, tab3 = st.tabs(["📖 Description", "🔬 Symptoms", "💊 Treatment"])

            with tab1:
                st.markdown(f"**{pred_class}**")
                st.write(info["description"])

            with tab2:
                st.markdown("**Symptoms to look for:**")
                for symptom in info["symptoms"]:
                    st.markdown(f"- {symptom}")

            with tab3:
                st.markdown("**Recommended Treatment:**")
                for step in info["treatment"]:
                    st.markdown(f"✅ {step}")

    else:
        st.info("👆 Upload a paddy leaf image to see prediction results")
        st.markdown("""
        <div style="
            background-color: #eafaf1;
            border-left: 5px solid #27ae60;
            padding: 15px;
            border-radius: 5px;">
            <p style="margin:0; color:#27ae60; font-weight:bold;">
                ⚠️ Important Notice
            </p>
            <p style="margin:5px 0; font-size:14px;">
                This app only works with <strong>paddy (rice) leaf images</strong>.
                Uploading other plant leaves will show a warning message.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:gray; font-size:13px;">
    Paddy Disease Detection | MobileNetV2 | Accuracy: 99.58% | Built with Streamlit
</p>
""", unsafe_allow_html=True)
