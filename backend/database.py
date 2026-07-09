# SQLite schema setup and migrations.

import sqlite3
from config import DB_PATH


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
            image_path    TEXT,
            heatmap_path  TEXT
        )
    """)
    # Migration: add session_id column if it doesn't already exist (safe on existing DBs)
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "session_id" not in existing_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN session_id TEXT")
    conn.commit()
    conn.close()