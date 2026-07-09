# Shared paths, constants, and the rate limiter instance.
# Kept separate from main.py so route modules can import these without
# creating a circular import back to main.py.

from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
DB_PATH     = BASE_DIR / "paddy_doctor.db"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model / prediction constants ─────────────────────────────────────────────
IMG_SIZE             = (224, 224)
VALIDATOR_THRESHOLD  = 0.6
CONFIDENCE_THRESHOLD = 0.70

# ── Rate limiter (shared singleton used by both main.py and routes) ─────────
limiter = Limiter(key_func=get_remote_address)