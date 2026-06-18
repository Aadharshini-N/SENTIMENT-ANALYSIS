import sqlite3
from datetime import datetime

DB_PATH = "database/reviews.db"

# ── Create table if it doesn't exist ─────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT    NOT NULL,
            sentiment   TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            created_at  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database ready.")

# ── Save one prediction ───────────────────────────────────────
def save_review(review_text, sentiment, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reviews (review_text, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (review_text, sentiment, round(confidence, 4), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ── Fetch last 10 predictions ─────────────────────────────────
def get_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT review_text, sentiment, confidence, created_at
        FROM reviews
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "review_text": row[0],
            "sentiment":   row[1],
            "confidence":  row[2],
            "created_at":  row[3]
        }
        for row in rows
    ]

# ── Run this file directly to test ───────────────────────────
if __name__ == "__main__":
    init_db()
    save_review("Great product!", "Positive", 0.95)
    print(get_history())