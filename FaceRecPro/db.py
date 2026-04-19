import sqlite3
import json
import os
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "database", "face_db.sqlite")

_BACKEND = None


def _use_sqlite_explicit():
    v = os.environ.get("USE_SQLITE", "").lower()
    return v in ("1", "true", "yes")


def _detect_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if _use_sqlite_explicit():
        _BACKEND = "sqlite"
        print(f"Using SQLite (USE_SQLITE): {SQLITE_DB_PATH}")
        return _BACKEND
    try:
        import mysql.connector

        conn = mysql.connector.connect(**config.DB_CONFIG)
        conn.close()
        _BACKEND = "mysql"
        print("Using MySQL for FaceRec Pro.")
    except Exception as err:
        print(f"MySQL unavailable ({err}); falling back to SQLite: {SQLITE_DB_PATH}")
        _BACKEND = "sqlite"
    return _BACKEND


def get_db_connection():
    backend = _detect_backend()
    if backend == "sqlite":
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    import mysql.connector

    try:
        return mysql.connector.connect(**config.DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"MySQL connection error: {err}")
        raise err


def init_db():
    backend = _detect_backend()
    if backend == "sqlite":
        _init_sqlite()
    else:
        _init_mysql()


def _init_sqlite():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_seen TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER,
            name TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            emotion TEXT,
            ear REAL,
            gaze_x REAL,
            gaze_y REAL,
            engagement REAL,
            metrics_json TEXT
        )
        """
    )
    cursor.execute("PRAGMA table_info(history)")
    columns = {row[1] for row in cursor.fetchall()}
    for col, ddl in [
        ("emotion", "ALTER TABLE history ADD COLUMN emotion TEXT"),
        ("ear", "ALTER TABLE history ADD COLUMN ear REAL"),
        ("gaze_x", "ALTER TABLE history ADD COLUMN gaze_x REAL"),
        ("gaze_y", "ALTER TABLE history ADD COLUMN gaze_y REAL"),
        ("engagement", "ALTER TABLE history ADD COLUMN engagement REAL"),
        ("metrics_json", "ALTER TABLE history ADD COLUMN metrics_json TEXT"),
    ]:
        if col not in columns:
            try:
                cursor.execute(ddl)
                print(f"SQLite migration: added column [{col}]")
            except sqlite3.OperationalError:
                pass
    conn.commit()
    conn.close()
    print("SQLite database ready.")


def _init_mysql():
    conn = get_db_connection()
    cursor = conn.cursor()
    id_type = "INT AUTO_INCREMENT PRIMARY KEY"
    text_type = "LONGTEXT"
    ts_type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"

    cursor.execute(
        f"""
    CREATE TABLE IF NOT EXISTS faces (
        id {id_type},
        name VARCHAR(255) NOT NULL,
        encoding {text_type} NOT NULL,
        created_at {ts_type},
        last_seen {ts_type} NULL
    )
    """
    )

    cursor.execute(
        f"""
    CREATE TABLE IF NOT EXISTS history (
        id {id_type},
        face_id INT,
        name VARCHAR(255) NOT NULL,
        confidence FLOAT,
        timestamp {ts_type},
        image_path VARCHAR(255)
    )
    """
    )

    cursor.execute("SHOW COLUMNS FROM history")
    columns = [col[0] for col in cursor.fetchall()]

    required_cols = {
        "emotion": "VARCHAR(50)",
        "ear": "FLOAT",
        "gaze_x": "FLOAT",
        "gaze_y": "FLOAT",
        "engagement": "FLOAT",
        "metrics_json": "LONGTEXT",
    }

    for col, dtype in required_cols.items():
        if col not in columns:
            print(f"MIGRATION: Adding column [{col}] to history table...")
            cursor.execute(f"ALTER TABLE history ADD COLUMN {col} {dtype}")

    conn.commit()
    conn.close()
    print("MySQL tables initialized and migrated.")


def add_face(name, encoding):
    backend = _detect_backend()
    encoding_json = json.dumps(encoding.tolist())
    conn = get_db_connection()
    cursor = conn.cursor()
    if backend == "sqlite":
        cursor.execute(
            "INSERT INTO faces (name, encoding) VALUES (?, ?)",
            (name, encoding_json),
        )
        face_id = cursor.lastrowid
    else:
        cursor.execute(
            "INSERT INTO faces (name, encoding) VALUES (%s, %s)",
            (name, encoding_json),
        )
        face_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return face_id


def get_all_faces():
    backend = _detect_backend()
    conn = get_db_connection()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces")
        rows = [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]
    else:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM faces")
        rows = cursor.fetchall()
    conn.close()
    return rows


def add_history_entry(
    name,
    confidence,
    face_id=None,
    image_path=None,
    emotion=None,
    ear=None,
    gaze=None,
    engagement=None,
    metrics_json=None,
):
    backend = _detect_backend()
    gx = gaze.get("gaze_x", 0) if gaze else 0
    gy = gaze.get("gaze_y", 0) if gaze else 0
    mj = json.dumps(metrics_json) if metrics_json is not None else None

    conn = get_db_connection()
    cursor = conn.cursor()
    if backend == "sqlite":
        cursor.execute(
            """
            INSERT INTO history
            (name, confidence, face_id, image_path, emotion, ear, gaze_x, gaze_y, engagement, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, confidence, face_id, image_path, emotion, ear, gx, gy, engagement, mj),
        )
        if face_id:
            cursor.execute(
                "UPDATE faces SET last_seen = CURRENT_TIMESTAMP WHERE id = ?",
                (face_id,),
            )
    else:
        cursor.execute(
            """
            INSERT INTO history
            (name, confidence, face_id, image_path, emotion, ear, gaze_x, gaze_y, engagement, metrics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (name, confidence, face_id, image_path, emotion, ear, gx, gy, engagement, mj),
        )
        if face_id:
            cursor.execute(
                "UPDATE faces SET last_seen = NOW() WHERE id = %s",
                (face_id,),
            )
    conn.commit()
    conn.close()


def get_history(limit=50):
    backend = _detect_backend()
    conn = get_db_connection()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]
    else:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM history ORDER BY timestamp DESC LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()
    conn.close()
    return rows
