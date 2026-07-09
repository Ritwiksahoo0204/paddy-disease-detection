# /dashboard/stats route (admin-only).

import sqlite3
from fastapi import APIRouter, Depends

from config import DB_PATH
from auth import verify_admin

router = APIRouter()


@router.get("/dashboard/stats")
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