import sqlite3
import mysql.connector
from mysql.connector import errorcode
import json
import datetime
import os
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "database", "face_db.sqlite")

def get_db_connection():
    try:
        conn = mysql.connector.connect(**config.DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"CRITICAL: MySQL Connection Error: {err}")
        raise err

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Strictly MySQL syntax
    id_type = "INT AUTO_INCREMENT PRIMARY KEY"
    text_type = "LONGTEXT"
    ts_type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    
    # Faces table
    print("Executing: CREATE TABLE faces (MySQL ONLY)")
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS faces (
        id {id_type},
        name VARCHAR(255) NOT NULL,
        encoding {text_type} NOT NULL,
        created_at {ts_type},
        last_seen {ts_type} NULL
    )
    ''')
    
    # History table
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS history (
        id {id_type},
        face_id INT,
        name VARCHAR(255) NOT NULL,
        confidence FLOAT,
        timestamp {ts_type},
        image_path VARCHAR(255)
    )
    ''')
    
    conn.commit()
    print("Database tables initialized.")
    conn.close()

def add_face(name, encoding):
    conn = get_db_connection()
    cursor = conn.cursor()
    encoding_json = json.dumps(encoding.tolist())
    
    query = "INSERT INTO faces (name, encoding) VALUES (%s, %s)"
    cursor.execute(query, (name, encoding_json))
    face_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return face_id

def get_all_faces():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM faces")
    faces = cursor.fetchall()
    conn.close()
    return faces

def add_history_entry(name, confidence, face_id=None, image_path=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "INSERT INTO history (name, confidence, face_id, image_path) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (name, confidence, face_id, image_path))
    
    if face_id:
        update_query = "UPDATE faces SET last_seen = NOW() WHERE id = %s"
        cursor.execute(update_query, (face_id,))
    
    conn.commit()
    conn.close()

def get_history(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows
