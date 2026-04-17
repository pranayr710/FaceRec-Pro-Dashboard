from fastapi import FastAPI, UploadFile, File, Form, Request
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

app = FastAPI(title="FaceRec Pro API")
# Ensure database is initialized
db.init_db()

engine = FaceEngine()

# Mount static files
app.mount("/static", StaticFiles(directory="FaceRecPro/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("FaceRecPro/static/index.html", "r") as f:
        return f.read()

@app.post("/enroll")
async def enroll_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    success, message = engine.enroll(contents, name)
    if success:
        return {"status": "success", "message": message}
    return JSONResponse(status_code=400, content={"status": "error", "message": message})

@app.get("/faces")
async def get_faces():
    faces = db.get_all_faces()
    return [{"id": f['id'], "name": f['name'], "created_at": f['created_at'], "last_seen": f['last_seen']} for f in faces]

@app.get("/history")
async def get_history():
    history = db.get_history()
    return [dict(h) for h in history]

@app.post("/recognize")
async def recognize_image(file: UploadFile = File(...), tolerance: float = 0.6):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = engine.recognize(frame, tolerance=tolerance)
    
    # Log results to history
    for res in results:
        db.add_history_entry(res['name'], res['confidence'], res['face_id'])
        
    return {"status": "success", "detections": results}

@app.post("/set_tolerance")
async def set_tolerance(value: float = Form(...)):
    engine.tolerance = value
    return {"status": "success", "tolerance": value}

@app.get("/stats")
async def get_stats():
    # We can use a global or just check the last results
    # For now, we'll return some aggregate data
    faces = db.get_all_faces()
    history = db.get_history(limit=50)
    
    # Calculate unique recognitions today
    today_count = len(history) # Simplified
    
    return {
        "enrolled": len(faces),
        "recent_activity": today_count,
        "status": "Healthy"
    }

import threading
import time

# Video Engine (Multi-threaded)
class VideoEngine:
    def __init__(self, engine):
        self.engine = engine
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.results = []
        self.running = True
        self.lock = threading.Lock()
        
        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print("Video Engine started in background thread.")

    def _update(self):
        print("DEBUG: Thread loop starting...")
        try:
            frame_count = 0
            while self.running:
                success, raw_frame = self.cap.read()
                if not success:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "CAMERA ERROR: Read Failed", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    with self.lock:
                        self.frame = error_frame
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                if frame_count % 4 == 0 or not self.results:
                    small_frame = cv2.resize(raw_frame, (0, 0), fx=0.5, fy=0.5)
                    new_results = self.engine.recognize(small_frame)
                    
                    if not new_results:
                        new_results = self.engine.recognize(raw_frame, scale=1.0)

                    with self.lock:
                        self.results = new_results

                with self.lock:
                    current_results = self.results
                
                display_frame = raw_frame.copy()
                for res in current_results:
                    top, right, bottom, left = res['bbox']
                    is_known = res['name'] != "Unknown"
                    color = (0, 255, 0) if is_known else (0, 165, 255)
                    
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)
                    cv2.rectangle(display_frame, (left, top - 35), (right, top), color, cv2.FILLED)
                    cv2.putText(display_frame, f"{res['name']}", (left + 6, top - 8), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
                
                with self.lock:
                    self.frame = display_frame
                
                time.sleep(0.01)
        except Exception as e:
            print(f"CRITICAL THREAD ERROR: {e}")
            import traceback
            traceback.print_exc()

    def get_frame(self):
        with self.lock:
            return self.frame

video_engine = VideoEngine(engine)

def gen_frames():
    while True:
        frame = video_engine.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03) # ~30 FPS

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/capture")
async def capture_current_frame():
    frame = video_engine.get_frame()
    if frame is None:
        return JSONResponse(status_code=500, content={"message": "No frame"})
    
    ret, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

# Mock endpoint for "Upload Video" demo
@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    # This would process a video file and return detection data
    # For simplicity, we save it and then simulate processing
    return {"status": "success", "message": "Video processing started (Demo)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
