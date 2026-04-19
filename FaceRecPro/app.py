from fastapi import FastAPI, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
import face_recognition
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import io
import os
import db
from face_engine import FaceEngine
import asyncio
import json
import time
import csv
from io import StringIO

app = FastAPI(title="FaceRec Pro Elite")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()
db.init_db()
engine = FaceEngine()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# View Options
view_options = {"show_landmarks": False}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(index_path, "r") as f:
        return f.read()

@app.get("/faces")
async def get_faces():
    try:
        faces = db.get_all_faces()
        return {"faces": faces}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/enroll")
async def enroll_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    success, msg = engine.enroll(contents, name)
    if success: return {"status": "success", "message": msg}
    return JSONResponse(status_code=400, content={"status": "error", "message": msg})

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        c1 = await file1.read(); c2 = await file2.read()
        img1 = face_recognition.load_image_file(io.BytesIO(c1))
        img2 = face_recognition.load_image_file(io.BytesIO(c2))
        enc1 = face_recognition.face_encodings(img1); enc2 = face_recognition.face_encodings(img2)
        if not enc1 or not enc2: return {"status": "error", "message": "Face not detected."}
        dist = face_recognition.face_distance([enc1[0]], enc2[0])[0]
        tol = getattr(engine, "tolerance", 0.52)
        match = bool(dist <= tol)
        return {"status": "success", "match": match, "distance": round(float(dist), 3), "confidence": round(float((1-dist)*100), 1)}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/history")
async def get_history():
    try:
        history = db.get_history()
        return {"history": [dict(h) for h in history]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/export_csv")
async def export_csv():
    history = db.get_history(limit=1000)
    output = StringIO()
    fieldnames = [
        "id",
        "name",
        "confidence",
        "emotion",
        "ear",
        "engagement",
        "gaze_x",
        "gaze_y",
        "attention_score",
        "fatigue_index",
        "blink_bpm",
        "head_yaw_deg",
        "timestamp",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in history:
        m = row.get("metrics_json")
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                m = {}
        elif m is None:
            m = {}
        hp = m.get("head_pose") or {}
        flat = {
            "id": row.get("id", ""),
            "name": row.get("name", ""),
            "confidence": row.get("confidence", ""),
            "emotion": row.get("emotion", ""),
            "ear": row.get("ear", ""),
            "engagement": row.get("engagement", ""),
            "gaze_x": row.get("gaze_x", ""),
            "gaze_y": row.get("gaze_y", ""),
            "attention_score": m.get("attention_score", ""),
            "fatigue_index": m.get("fatigue_index", ""),
            "blink_bpm": m.get("blink_bpm", ""),
            "head_yaw_deg": hp.get("yaw_deg", ""),
            "timestamp": row.get("timestamp", ""),
        }
        writer.writerow(flat)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=forensic_audit.csv"})

@app.post("/toggle_landmarks")
async def toggle_landmarks(state: str = Form(...)):
    view_options["show_landmarks"] = str(state).lower() in ("true", "1", "on", "yes")
    return {"status": "success", "show_landmarks": view_options["show_landmarks"]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)

import threading

class VideoEngine:
    def __init__(self, engine):
        self.engine = engine
        self.cap = cv2.VideoCapture(0)
        self.raw_frame = None
        self.display_frame = None
        self.results = []
        self.ai_latency = 0
        self.running = True
        self.lock = threading.Lock()
        
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._ai_worker, daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock: self.raw_frame = frame
            time.sleep(0.03)

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.raw_frame.copy() if self.raw_frame is not None else None
            if frame is not None:
                start = time.time()
                try:
                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    out = self.engine.recognize(small, scale=0.5)
                    detections = out.get("detections", [])
                    frame_size = out.get("frame_size") or [
                        frame.shape[0],
                        frame.shape[1],
                    ]
                    with self.lock:
                        self.results = detections
                        self.ai_latency = int((time.time() - start) * 1000)
                    
                    if asyncio_loop:
                        payload = {
                            "type": "biometric_update",
                            "detections": detections,
                            "ai_latency": self.ai_latency,
                            "frame_size": frame_size,
                        }
                        asyncio_loop.call_soon_threadsafe(lambda: asyncio.create_task(manager.broadcast(payload)))
                except Exception as e:
                    print(f"AI Worker Error: {e}")
            time.sleep(0.01)

    def _render_loop(self):
        while self.running:
            with self.lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None
                results = self.results
            if frame is not None:
                for res in results:
                    top, right, bottom, left = res['bbox']
                    color = (0, 255, 0) if res['name'] != "Unknown" else (0, 165, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    tid = res.get('track_id')
                    label = res['name'] if tid is None else f"{res['name']}  #{tid}"
                    cv2.putText(
                        frame, label, (left, max(top - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
                    if tid is not None:
                        cv2.putText(
                            frame, f"track {tid}", (left, min(bottom + 22, frame.shape[0] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1,
                        )
                    
                    if view_options["show_landmarks"]:
                        for part, pts in res.get("landmarks", {}).items():
                            for (px, py) in pts:
                                cv2.circle(frame, (px, py), 1, (0, 255, 255), -1)
                
                with self.lock: self.display_frame = frame
            time.sleep(0.03)

    def get_frame(self):
        with self.lock: return self.display_frame

video_engine = None
asyncio_loop = None

def gen_frames():
    while True:
        frame = video_engine.get_frame() if video_engine else None
        if frame is None:
            time.sleep(0.1); continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("startup")
async def startup_event():
    global video_engine, asyncio_loop
    asyncio_loop = asyncio.get_running_loop()
    video_engine = VideoEngine(engine)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
