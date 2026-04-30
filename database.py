
import sqlite3
import bcrypt
import re
import os
from datetime import datetime

# ── DB Path ───────────────────────────────────────────────────────────────────
# FIX #4: Use /data/ for persistent storage on HuggingFace Spaces.
# Go to HF Space Settings → Variables and set ADMIN_USERNAME and ADMIN_PASSWORD.
if os.path.exists('/content/drive/MyDrive/Project_2k26/'):
    DB_PATH = '/content/drive/MyDrive/Project_2k26/paddy_app.db'
elif os.path.exists('/data'):
    DB_PATH = '/data/paddy_app.db'   # persistent on HuggingFace Spaces
else:
    DB_PATH = 'paddy_app.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)

    # Create Activity History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            severity TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Create Analytics/Visits Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            visit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


# FIX #7: Username validation — alphanumeric + underscore, 3–20 chars only
def is_valid_username(username: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', username))


def create_user(username, password):
    # FIX #7: Validate username format
    if not is_valid_username(username):
        return False, "Username must be 3–20 characters, letters, numbers, or underscores only."

    # FIX #6: Enforce minimum password length
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "Username already exists."

        hashed = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                       (username, hashed))
        conn.commit()
        return True, "User created successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user:
        user_id, hashed = user
        if verify_password(password, hashed):
            # Update last login
            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))

            # Record visit
            cursor.execute("INSERT INTO visits (user_id) VALUES (?)", (user_id,))

            conn.commit()
            conn.close()
            return True, user_id

    conn.close()
    return False, None


# FIX #2: Password reset now requires the OLD password for verification
def update_password(username, old_password, new_password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return False, "Username not found. Please check spelling."

        user_id, hashed = row
        # Verify old password before allowing reset
        if not verify_password(old_password, hashed):
            return False, "Current password is incorrect."

        # FIX #6: Enforce minimum password length on reset too
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters long."

        new_hashed = hash_password(new_password)
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hashed, username))
        conn.commit()
        return True, "Password reset successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


# Admin-only force reset (no old password needed) — only called from admin panel
def admin_force_reset_password(username, new_password):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not cursor.fetchone():
            return False, "Username not found."

        hashed = hash_password(new_password)
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hashed, username))
        conn.commit()
        return True, "Password reset successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def record_activity(user_id, image_name, predicted_class, confidence, severity):
    if not user_id:
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO activity (user_id, image_name, predicted_class, confidence, severity)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, image_name, predicted_class, confidence, severity))
        conn.commit()
        return True
    except Exception as e:
        # FIX #15: Log the actual error instead of swallowing it silently
        print(f"[record_activity error] {e}")
        return False
    finally:
        conn.close()


def get_user_activity(user_id):
    if not user_id:
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # FIX #12: Limit to last 50 records to prevent slow rendering
    cursor.execute("""
        SELECT image_name, predicted_class, confidence, severity, timestamp
        FROM activity
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 50
    """, (user_id,))
    results = cursor.fetchall()
    conn.close()

    return [
        {
            "image_name": r[0],
            "predicted_class": r[1],
            "confidence": r[2],
            "severity": r[3],
            "timestamp": r[4]
        }
        for r in results
    ]


# Initialize on import
init_db()

# FIX #1: Admin credentials loaded from environment variables — NOT hardcoded.
# Set ADMIN_USERNAME and ADMIN_PASSWORD in HuggingFace Space Secrets.
_admin_user = os.environ.get("ADMIN_USERNAME", "admin")
_admin_pass = os.environ.get("ADMIN_PASSWORD", "changeme123")
create_user(_admin_user, _admin_pass)


