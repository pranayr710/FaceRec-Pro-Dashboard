from fastapi import FastAPI, UploadFile, File, Form, Request
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

app = FastAPI(title="FaceRec Pro API")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure database is initialized
db.init_db()

engine = FaceEngine()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Configuration
show_landmarks = False

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

@app.post("/toggle_landmarks")
async def toggle_landmarks(active: bool = Form(...)):
    global show_landmarks
    show_landmarks = active
    return {"status": "success", "show_landmarks": show_landmarks}

@app.post("/api/compare")
async def api_compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        res, err = engine.compare_two_faces(await file1.read(), await file2.read())
        if err:
            return JSONResponse(status_code=400, content={"message": err})
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/landmarks")
async def api_landmarks(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))
        # Convert to BGR for consistent processing if needed, but engine takes BGR
        # Wait, engine.get_landmarks takes BGR (OpenCV format)
        # face_recognition.load_image_file returns RGB
        bgr = image[:,:,::-1]
        res = engine.get_landmarks(bgr)
        return {"landmarks": res}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

import threading
import time

# Video Engine (Multi-threaded)
class VideoEngine:
    def __init__(self, engine):
        self.engine = engine
        self.cap = None
        self.active_index = -1
        
        # Camera Auto-Scanner
        for i in [0, 1, 2, 3, 4]:
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                success, _ = test_cap.read()
                if success:
                    self.cap = test_cap
                    self.active_index = i
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if not self.cap:
             self.cap = cv2.VideoCapture(0)

        self.raw_frame = None
        self.display_frame = None
        self.results = []
        self.landmarks = []
        self.running = True
        self.lock = threading.Lock()
        
        # 1. Capture Thread (Highest Priority)
        self.cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        # 2. AI Worker Thread (Lower Priority)
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        # 3. Render Thread (Maintains Smoothness)
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        
        # Start background threads
        self.cap_thread.start()
        self.ai_thread.start()
        self.render_thread.start()

    def _capture_loop(self):
        """Continuously pulls raw frames from camera"""
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.raw_frame = frame
            time.sleep(0.01)

    def _ai_worker(self):
        """Asynchronously processes AI onto the latest frame"""
        while self.running:
            with self.lock:
                frame_to_process = self.raw_frame.copy() if self.raw_frame is not None else None
            
            if frame_to_process is not None:
                try:
                    # 1. Recognition Scan
                    # Use a smaller frame for faster response if possible
                    small = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
                    new_res = self.engine.recognize(small)
                    if not new_res:
                        new_res = self.engine.recognize(frame_to_process, scale=1.0)
                        
                    # 2. Landmark Scan (Only if enabled)
                    new_landmarks = []
                    if show_landmarks:
                        new_landmarks = self.engine.get_landmarks(frame_to_process)

                    with self.lock:
                        self.results = new_res
                        self.landmarks = new_landmarks
                except Exception:
                    pass
            
            time.sleep(0.1) # Throttle AI to ~10 FPS to save CPU

    def _render_loop(self):
        """Merges latest raw frame and latest AI results at 30 FPS"""
        while self.running:
            with self.lock:
                base_frame = self.raw_frame.copy() if self.raw_frame is not None else None
                current_results = self.results
                current_landmarks = self.landmarks
            
            if base_frame is not None:
                # 1. Draw Recognition Boxes
                for res in current_results:
                    top, right, bottom, left = res['bbox']
                    is_known = res['name'] != "Unknown"
                    color = (0, 255, 0) if is_known else (0, 165, 255)
                    cv2.rectangle(base_frame, (left, top), (right, bottom), color, 3)
                    cv2.rectangle(base_frame, (left, top - 35), (right, top), color, cv2.FILLED)
                    cv2.putText(base_frame, f"{res['name']}", (left + 6, top - 8), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
                
                # 2. Draw Pre-computed Landmarks
                if show_landmarks:
                    for face_landmarks in current_landmarks:
                        for feature_name, points in face_landmarks.items():
                            for point in points:
                                cv2.circle(base_frame, point, 2, (0, 255, 255), -1)

                with self.lock:
                    self.display_frame = base_frame
            
            time.sleep(0.03) # ~33 FPS

    def get_frame(self):
        with self.lock:
            return self.display_frame

video_engine = VideoEngine(engine)

def gen_frames():
    while True:
        frame = video_engine.get_frame()
        if frame is None:
            # Fallback loading frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "BIOMETRIC ENGINE INITIALIZING...", (60, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5) 
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
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
