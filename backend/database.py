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
    # Migration: add any expected column that's missing (safe on existing DBs
    # of any prior schema version — checks each column individually rather
    # than assuming only session_id could ever be missing)
    expected_columns = {
        "image_path":   "TEXT",
        "heatmap_path": "TEXT",
        "session_id":   "TEXT",
    }
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    for col_name, col_type in expected_columns.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
    conn.commit()
    conn.close()