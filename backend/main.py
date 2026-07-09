import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import BASE_DIR, UPLOADS_DIR, limiter
from database import init_db
import ml_utils  # noqa: F401  (loads models at import time; must happen before routes are used)
from routes import predict, history, dashboard

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Paddy Doctor API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
_frontend_urls = os.getenv("FRONTEND_URL", "")
if _frontend_urls:
    ALLOWED_ORIGINS += [origin.strip() for origin in _frontend_urls.split(",") if origin.strip()]
else:
    logger.warning(
        "FRONTEND_URL env var not set — CORS will only allow localhost origins. "
        "Set FRONTEND_URL to your deployed frontend's URL (comma-separate multiple "
        "URLs if you have more than one) or cross-origin requests from your real "
        "frontend will be blocked."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# ── Database ───────────────────────────────────────────────────────────────────
init_db()

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(dashboard.router)

# ── Serve React Frontend ────────────────────────────────────────────────────────
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