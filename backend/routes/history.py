# /feedback/{id}, /prediction/{id}, and /history routes.

import sqlite3
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from config import DB_PATH
from treatments_data import TREATMENTS, DISCLAIMER, EXPERT_HELPLINE
from ml_utils import get_warning

router = APIRouter()


@router.post("/feedback/{prediction_id}")
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


@router.get("/prediction/{prediction_id}")
async def get_prediction(prediction_id: int, request: Request, lang: str = "en"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT id, disease, confidence, severity, validator_prob,
                  was_correct, timestamp, image_path, heatmap_path
           FROM predictions WHERE id = ?""",
        (prediction_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    cls_name = row[1]
    confidence = row[2]
    severity = row[3]
    val_prob = row[4]
    was_correct = row[5]
    timestamp = row[6]
    image_filename = row[7]
    heatmap_filename = row[8]

    lang = lang if lang in ["en", "hi", "bn"] else "en"
    disease_data = TREATMENTS.get(cls_name)
    if not disease_data:
        raise HTTPException(status_code=404, detail="Disease info not found")

    info = disease_data[lang]
    risk_level = disease_data["risk_level"]
    warning = get_warning(cls_name, confidence)

    base_url = str(request.base_url).rstrip('/')
    original_url = f"{base_url}/uploads/{image_filename}" if image_filename else None
    heatmap_url = f"{base_url}/uploads/{heatmap_filename}" if heatmap_filename else None

    # Reconstruct top3 and all_probabilities for compatible frontend client review
    top3 = [{"disease": cls_name, "probability": confidence}]
    all_probs = {cls_name: confidence}

    return JSONResponse({
        "status": "success",
        "prediction_id": prediction_id,
        "disease": cls_name,
        "confidence": confidence,
        "severity": severity,
        "risk_level": risk_level,
        "validator_prob": val_prob,
        "was_correct": was_correct,
        "timestamp": timestamp,
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


@router.get("/history")
async def get_history(request: Request):
    session_id = request.headers.get("x-session-id") or "anonymous"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT id, disease, confidence, severity, validator_prob,
                  was_correct, timestamp, image_path
           FROM predictions WHERE session_id = ? ORDER BY timestamp DESC LIMIT 20""",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    base_url = str(request.base_url).rstrip('/')

    history = []
    for row in rows:
        filename = row[7]
        url = f"{base_url}/uploads/{filename}" if filename else None
        history.append({
            "id":            row[0],
            "disease":       row[1],
            "confidence":    row[2],
            "severity":      row[3],
            "validator_prob": row[4],
            "was_correct":   row[5],
            "timestamp":     row[6],
            "image_url":     url
        })
    return {"history": history}