import sqlite3
import bcrypt
from datetime import datetime
import os

if os.path.exists('/content/drive/MyDrive/Project_2k26/'):
    DB_PATH = '/content/drive/MyDrive/Project_2k26/paddy_app.db'
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

def create_user(username, password):
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

def update_password(username, new_password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not cursor.fetchone():
            return False, "Username not found. Please check spelling."
            
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
    except:
        return False
    finally:
        conn.close()

def get_user_activity(user_id):
    if not user_id:
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT image_name, predicted_class, confidence, severity, timestamp 
        FROM activity 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    """, (user_id,))
    results = cursor.fetchall()
    conn.close()
    
    # Format as list of dicts
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

# Automatically create the default admin account if it doesn't exist
create_user("Ritwiksahoo0204", "0204")
