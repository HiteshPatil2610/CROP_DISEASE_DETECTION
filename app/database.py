"""
SQLite Database Handler
Stores all detection results with full history
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "app/detections.db"
Path("app").mkdir(exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # Access columns by name
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS detections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            image_filename  TEXT NOT NULL,
            crop_type       TEXT NOT NULL,
            disease_name    TEXT NOT NULL,
            confidence      REAL NOT NULL,
            severity        TEXT NOT NULL,
            description     TEXT,
            treatment       TEXT,
            yolo_boxes      INTEGER DEFAULT 0,
            inference_ms    REAL,
            result_image    TEXT,
            top5_json       TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            model_name  TEXT NOT NULL,
            accuracy    REAL,
            precision   REAL,
            recall      REAL,
            f1_score    REAL,
            map50       REAL,
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS crop_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_type   TEXT UNIQUE,
            total_scans INTEGER DEFAULT 0,
            disease_count INTEGER DEFAULT 0,
            healthy_count INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized:", DB_PATH)

def save_detection(result: dict) -> int:
    """Save a prediction result to DB. Returns new row ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections
        (timestamp, image_filename, crop_type, disease_name, confidence,
         severity, description, treatment, yolo_boxes, inference_ms, result_image, top5_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result.get("image_path", ""),
        result.get("crop_type", ""),
        result.get("disease_name", ""),
        result.get("confidence", 0),
        result.get("severity", ""),
        result.get("description", ""),
        result.get("treatment", ""),
        result.get("yolo_boxes", 0),
        result.get("inference_ms", 0),
        result.get("result_image", ""),
        json.dumps(result.get("top5", []))
    ))

    row_id = cursor.lastrowid
    _update_crop_stats(cursor, result)
    conn.commit()
    conn.close()
    return row_id

def _update_crop_stats(cursor, result):
    crop = result.get("crop_type", "Unknown")
    is_healthy = "healthy" in result.get("disease_name", "").lower()
    cursor.execute("""
        INSERT INTO crop_stats (crop_type, total_scans, disease_count, healthy_count)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(crop_type) DO UPDATE SET
            total_scans   = total_scans + 1,
            disease_count = disease_count + ?,
            healthy_count = healthy_count + ?
    """, (crop, 0 if is_healthy else 1, 1 if is_healthy else 0,
              0 if is_healthy else 1, 1 if is_healthy else 0))

def get_all_detections(limit: int = 100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_detection_by_id(detection_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM detections WHERE id = ?", (detection_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_stats():
    conn = get_connection()
    total    = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    by_crop  = conn.execute("SELECT * FROM crop_stats").fetchall()
    by_disease = conn.execute("""
        SELECT disease_name, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM detections GROUP BY disease_name ORDER BY count DESC
    """).fetchall()
    conn.close()
    return {
        "total_scans": total,
        "by_crop"    : [dict(r) for r in by_crop],
        "by_disease" : [dict(r) for r in by_disease]
    }

def delete_detection(detection_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    conn.commit()
    conn.close()

# Initialize on import
init_db()