import os
import io
import json
import base64
import sqlite3
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image
import cv2
import tensorflow as tf
from dotenv import load_dotenv
import secrets

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rate Limiter ────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Paddy Doctor API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    os.getenv("FRONTEND_URL", "https://paddy-disease-detection.vercel.app"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH    = BASE_DIR / "paddy_doctor.db"

# ── Admin Auth ───────────────────────────────────────────────────────────────
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ── Load Models (Lazy) ──────────────────────────────────────────────────────
main_model  = None
validator   = None
class_names = {}

def get_models():
    global main_model, validator, class_names
    if main_model is None:
        import gc
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        main_model = tf.keras.models.load_model(
            str(MODELS_DIR / "best_model.keras"), compile=False
        )
        validator = tf.keras.models.load_model(
            str(MODELS_DIR / "paddy_validator.keras"), compile=False
        )
        with open(MODELS_DIR / "class_names.json") as f:
            class_names = json.load(f)
        gc.collect()
        logger.info("Models loaded successfully!")
    return main_model, validator, class_names
# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            disease       TEXT,
            confidence    REAL,
            severity      TEXT,
            validator_prob REAL,
            was_correct   TEXT,
            timestamp     TEXT,
            image_base64  TEXT,
            heatmap_base64 TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE              = (224, 224)
VALIDATOR_THRESHOLD   = 0.6
CONFIDENCE_THRESHOLD  = 0.70

# ── Treatment Database ────────────────────────────────────────────────────────
TREATMENTS = {
    "Bacterialblight": {
        "description": "Bacterial Leaf Blight is caused by Xanthomonas oryzae. It causes yellowing and wilting of leaves from the tips and edges.",
        "symptoms": [
            "Yellow to white lesions along leaf margins",
            "Wilting and drying of leaves from tip to base",
            "Milky or opaque bacterial ooze on cut ends",
            "Leaves turn grayish green then yellow"
        ],
        "treatment": [
            "Apply Copper Oxychloride (3g/L) spray immediately",
            "Apply Streptocycline (0.5g/L) + Copper Oxychloride mixture",
            "Drain fields during active tillering stage",
            "Avoid excess nitrogen fertilizer application",
            "Remove and destroy infected plant debris"
        ],
        "prevention": [
            "Use resistant varieties like IR-64, BR-23",
            "Use certified disease-free seeds only",
            "Maintain proper field drainage at all times",
            "Avoid clipping seedling tips during transplanting",
            "Practice crop rotation with non-host crops"
        ],
        "severity_note": "Can cause 20-30% yield loss if untreated",
        "risk_level": "High"
    },
    "Blast": {
        "description": "Rice Blast is caused by Magnaporthe oryzae fungus. It is one of the most destructive rice diseases worldwide affecting leaves, nodes and panicles.",
        "symptoms": [
            "Diamond-shaped lesions with gray center and brown border",
            "White to gray spindle shaped spots on leaves",
            "Neck rot causing panicle to fall over",
            "Infected nodes turn blackish and break easily"
        ],
        "treatment": [
            "Apply Tricyclazole (0.6g/L) at early infection stage",
            "Apply Isoprothiolane (1.5ml/L) as foliar spray",
            "Use Carbendazim (1g/L) for severe infection",
            "Spray at booting and heading stage",
            "Repeat spray every 7-10 days during outbreak"
        ],
        "prevention": [
            "Use blast resistant varieties like Swarna, MTU-7029",
            "Avoid excess nitrogen — increases susceptibility",
            "Maintain proper plant spacing for air circulation",
            "Remove weeds that harbor the fungus",
            "Treat seeds with Thiram or Captan before sowing"
        ],
        "severity_note": "Can cause complete crop failure if neck blast occurs",
        "risk_level": "High"
    },
    "Brownspot": {
        "description": "Brown Spot is caused by Cochliobolus miyabeanus fungus. Common in nutrient-deficient soils and can cause significant grain discoloration.",
        "symptoms": [
            "Circular to oval brown spots with yellow halo on leaves",
            "Dark brown spots on leaf sheaths and glumes",
            "Infected seeds show dark brown discoloration",
            "Spots enlarge and coalesce in severe cases"
        ],
        "treatment": [
            "Apply Mancozeb (2.5g/L) as foliar spray",
            "Apply Iprodione or Propiconazole fungicide",
            "Use Edifenphos (1ml/L) spray at early stage",
            "Apply potassium and phosphorus fertilizers",
            "Spray at tillering and booting stages"
        ],
        "prevention": [
            "Improve soil fertility especially potassium",
            "Use balanced NPK fertilization",
            "Treat seeds with Thiram (2g/kg seed)",
            "Use resistant varieties where available",
            "Maintain proper water management"
        ],
        "severity_note": "Can cause 5-45% yield loss depending on severity",
        "risk_level": "Moderate"
    },
    "Healthy": {
        "description": "Your paddy plant appears healthy with no visible disease symptoms detected. Continue with regular care and monitoring practices.",
        "symptoms": [
            "No disease symptoms detected",
            "Leaves appear green and healthy",
            "Normal growth pattern observed",
            "No lesions or discoloration visible"
        ],
        "treatment": [
            "No treatment needed at this time",
            "Continue regular irrigation schedule",
            "Maintain balanced NPK fertilization",
            "Keep monitoring weekly for early detection",
            "Ensure proper drainage is maintained"
        ],
        "prevention": [
            "Maintain proper field drainage always",
            "Use balanced NPK fertilization",
            "Regular field inspection every 7-10 days",
            "Keep field free of weeds at all times",
            "Use certified disease-free seeds for next season"
        ],
        "severity_note": "Plant is healthy — maintain regular monitoring",
        "risk_level": "Low"
    },
    "Tungro": {
        "description": "Rice Tungro Disease is caused by two viruses transmitted by green leafhopper insects. It causes severe stunting and yellowing.",
        "symptoms": [
            "Yellow to orange discoloration of leaves",
            "Stunted plant growth and reduced tillering",
            "Interveinal chlorosis on younger leaves",
            "Reduced and partially sterile panicles"
        ],
        "treatment": [
            "No direct cure — focus on vector control immediately",
            "Apply Imidacloprid (0.5ml/L) to control leafhoppers",
            "Apply Thiamethoxam to reduce vector population",
            "Remove and destroy infected plants immediately",
            "Apply Carbofuran granules at transplanting"
        ],
        "prevention": [
            "Use Tungro resistant varieties like TN1, IR-36",
            "Synchronize planting with neighbors to reduce vectors",
            "Avoid planting near already infected fields",
            "Use yellow sticky traps to monitor leafhoppers",
            "Treat seeds with systemic insecticides before sowing"
        ],
        "severity_note": "Can cause 20-100% yield loss — act immediately",
        "risk_level": "High"
    }
}

DISCLAIMER = {
    "en": "This AI tool provides preliminary disease screening ONLY. It is NOT a substitute for professional agricultural advice. NEVER apply pesticides based solely on this result. Always verify with a certified agricultural officer.",
    "hi": "यह AI उपकरण केवल प्रारंभिक रोग जांच के लिए है। यह पेशेवर कृषि सलाह का विकल्प नहीं है।",
    "bn": "এই AI টুল শুধুমাত্র প্রাথমিক রোগ নির্ণয়ের জন্য। এটি পেশাদার কৃষি পরামর্শের বিকল্প নয়।"
}

EXPERT_HELPLINE = "Kisan Call Centre: 1800-180-1551 (Toll Free, 24/7)"

# ── Image Quality Check ──────────────────────────────────────────────────────
def check_image_quality(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = img_array.mean()
    if blur_score < 50:
        return False, "Image is too blurry. Please retake with a steady hand."
    if mean_brightness < 30:
        return False, "Image is too dark. Please improve lighting."
    if mean_brightness > 220:
        return False, "Image is overexposed. Please reduce lighting."
    return True, "OK"

# ── Severity Estimation ──────────────────────────────────────────────────────
def get_severity(confidence, disease):
    if disease == "Healthy":
        return "None"
    if confidence >= 90:
        return "Severe"
    elif confidence >= 75:
        return "Moderate"
    else:
        return "Mild"

# ── Warning ──────────────────────────────────────────────────────────────────
def get_warning(disease, confidence):
    if disease == "Blast" and confidence < 92:
        return "Blast can resemble Brownspot. Please verify with an agricultural expert."
    if disease == "Tungro":
        return "Tungro spreads rapidly. Isolate affected area and contact expert immediately."
    return None

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def generate_gradcam(img_array, model, class_idx):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv_layer).output, model.output]
        )

        img_tensor = tf.cast(np.expand_dims(img_array, axis=0), tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]

        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # ── Return ONLY the superimposed heatmap ──
        original     = (img_array * 255).astype(np.uint8)
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        buf = io.BytesIO()
        Image.fromarray(superimposed).save(buf, format='PNG')
        buf.seek(0)

        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return None

# ── Preprocess ───────────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array   = np.array(img_resized) / 255.0
    return img_array

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "message": "Paddy Doctor API running"}

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    main_model, validator, class_names = get_models()
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted.")

    image_bytes = await file.read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    # Image quality check
    img_uint8 = (img_array * 255).astype(np.uint8)
    quality_ok, quality_msg = check_image_quality(img_uint8)
    if not quality_ok:
        return JSONResponse({
            "status": "quality_error",
            "message": quality_msg,
            "tip": "Take photo in natural daylight, avoid shadows, hold camera steady."
        })

    img_input = np.expand_dims(img_array, axis=0)

    # Validator
    val_prob = float(validator.predict(img_input, verbose=0)[0][0])
    if val_prob < VALIDATOR_THRESHOLD:
        return JSONResponse({
            "status": "not_paddy",
            "validator_prob": round(val_prob, 3),
            "message": "This does not appear to be a paddy leaf image.",
            "tip": "Please upload a clear photo of a single paddy leaf."
        })

    # Prediction
    pred      = main_model.predict(img_input, verbose=0)[0]
    cls_idx   = int(np.argmax(pred))
    cls_name  = class_names[str(cls_idx)]
    confidence = float(np.max(pred) * 100)

    # Low confidence
    if confidence < CONFIDENCE_THRESHOLD * 100:
        return JSONResponse({
            "status": "uncertain",
            "disease": cls_name,
            "confidence": round(confidence, 2),
            "message": f"Confidence too low ({confidence:.1f}%) to safely recommend treatment.",
            "tip": "Retake photo in better lighting with leaf filling most of frame.",
            "disclaimer": DISCLAIMER,
            "expert_helpline": EXPERT_HELPLINE
        })

    severity  = get_severity(confidence, cls_name)
    warning   = get_warning(cls_name, confidence)
    info      = TREATMENTS[cls_name]
    all_probs = {class_names[str(i)]: round(float(pred[i]) * 100, 2) for i in range(len(pred))}

    # Top 3 predictions
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = [{"disease": k, "probability": v} for k, v in sorted_probs[:3]]

    # Grad-CAM
    # heatmap_b64 = generate_gradcam(img_array, main_model, cls_idx)
    heatmap_b64 = None  #Grad-CAM — disabled on free tier to save memory

    # Image base64
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (disease, confidence, severity, validator_prob, timestamp, image_base64, heatmap_base64)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (cls_name, round(confidence, 2), severity, round(val_prob, 3),
         datetime.now().isoformat(), img_b64, heatmap_b64)
    )
    conn.commit()
    conn.close()

    return JSONResponse({
        "status": "success",
        "disease": cls_name,
        "confidence": round(confidence, 2),
        "severity": severity,
        "risk_level": info["risk_level"],
        "validator_prob": round(val_prob, 3),
        "all_probabilities": all_probs,
        "top3": top3,
        "warning": warning,
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "severity_note": info["severity_note"],
        "heatmap": heatmap_b64,
        "image": img_b64,
        "disclaimer": DISCLAIMER,
        "expert_helpline": EXPERT_HELPLINE
    })

@app.post("/feedback/{prediction_id}")
async def submit_feedback(prediction_id: int, request: Request):
    data = await request.json()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE predictions SET was_correct = ? WHERE id = ?",
        (data.get("was_correct"), prediction_id)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Thank you for your feedback!"}

@app.get("/history")
async def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT id, disease, confidence, severity, validator_prob,
                  was_correct, timestamp, image_base64
           FROM predictions ORDER BY timestamp DESC LIMIT 20"""
    )
    rows = cursor.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({
            "id":            row[0],
            "disease":       row[1],
            "confidence":    row[2],
            "severity":      row[3],
            "validator_prob": row[4],
            "was_correct":   row[5],
            "timestamp":     row[6],
            "image_base64":  row[7]
        })
    return {"history": history}

@app.get("/dashboard/stats")
async def dashboard_stats(username: str = Depends(verify_admin)):
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    diseased = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE disease != 'Healthy'"
    ).fetchone()[0]
    healthy = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE disease = 'Healthy'"
    ).fetchone()[0]

    disease_dist = conn.execute(
        "SELECT disease, COUNT(*) FROM predictions GROUP BY disease"
    ).fetchall()

    daily_scans = conn.execute(
        """SELECT DATE(timestamp) as date, COUNT(*) as count
           FROM predictions
           GROUP BY DATE(timestamp)
           ORDER BY date DESC LIMIT 7"""
    ).fetchall()

    accuracy_data = conn.execute(
        """SELECT was_correct, COUNT(*) FROM predictions
           WHERE was_correct IS NOT NULL GROUP BY was_correct"""
    ).fetchall()

    conn.close()

    return {
        "total_scans":    total,
        "diseased_leaves": diseased,
        "healthy_leaves":  healthy,
        "model_accuracy":  94.26,
        "disease_distribution": [{"name": r[0], "value": r[1]} for r in disease_dist],
        "daily_scans": [{"date": r[0], "count": r[1]} for r in daily_scans],
        "accuracy_feedback": {r[0]: r[1] for r in accuracy_data}
    }

# ── Serve React Frontend ──────────────────────────────────────────────────────
FRONTEND_BUILD = BASE_DIR.parent / "frontend" / "dist"
if FRONTEND_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_BUILD / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        return FileResponse(str(FRONTEND_BUILD / "index.html"))
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)