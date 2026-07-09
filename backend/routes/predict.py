# /health and /predict routes.

import sqlite3
import secrets
import base64
import numpy as np
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse

from config import DB_PATH, UPLOADS_DIR, VALIDATOR_THRESHOLD, CONFIDENCE_THRESHOLD, limiter
from treatments_data import TREATMENTS, DISCLAIMER, EXPERT_HELPLINE
from ml_utils import (
    main_model, validator, class_names,
    check_image_quality, get_severity, get_warning,
    generate_gradcam, preprocess_image,
)

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "message": "Paddy Doctor API running"}


@router.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...), lang: str = "en"):
    session_id = request.headers.get("x-session-id") or "anonymous"
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

    # Resolve language-specific info
    lang = lang if lang in ["en", "hi", "bn"] else "en"
    disease_data = TREATMENTS[cls_name]
    info = disease_data[lang]
    risk_level = disease_data["risk_level"]

    all_probs = {class_names[str(i)]: round(float(pred[i]) * 100, 2) for i in range(len(pred))}

    # Top 3 predictions
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = [{"disease": k, "probability": v} for k, v in sorted_probs[:3]]

    # Grad-CAM
    try:
        heatmap_b64 = generate_gradcam(img_array, cls_idx)
    except Exception:
        heatmap_b64 = None

    # Save files to disk instead of database
    scan_id = secrets.token_hex(8)
    original_filename = f"original_{scan_id}.jpg"
    original_path = UPLOADS_DIR / original_filename
    with open(original_path, "wb") as f:
        f.write(image_bytes)

    heatmap_filename = None
    if heatmap_b64:
        heatmap_filename = f"heatmap_{scan_id}.jpg"
        heatmap_path = UPLOADS_DIR / heatmap_filename
        with open(heatmap_path, "wb") as f:
            f.write(base64.b64decode(heatmap_b64))

    # Save to DB (store only filenames)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO predictions
           (disease, confidence, severity, validator_prob, timestamp, image_path, heatmap_path, session_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (cls_name, round(confidence, 2), severity, round(val_prob, 3),
         datetime.now().isoformat(), original_filename, heatmap_filename, session_id)
    )
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()

    base_url = str(request.base_url).rstrip('/')
    original_url = f"{base_url}/uploads/{original_filename}"
    heatmap_url = f"{base_url}/uploads/{heatmap_filename}" if heatmap_filename else None

    return JSONResponse({
        "status": "success",
        "prediction_id": prediction_id,
        "disease": cls_name,
        "confidence": round(confidence, 2),
        "severity": severity,
        "risk_level": risk_level,
        "validator_prob": round(val_prob, 3),
        "all_probabilities": all_probs,
        "top3": top3,
        "warning": warning,
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "severity_note": info["severity_note"],
        "heatmap": heatmap_url,
        "image": original_url,
        "disclaimer": DISCLAIMER,
        "expert_helpline": EXPERT_HELPLINE
    })