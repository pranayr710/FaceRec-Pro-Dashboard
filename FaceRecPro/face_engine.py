import face_recognition
import cv2
import numpy as np
import json
import os
from PIL import Image
import io
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class FaceEngine:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.tolerance = 0.6 # Default
        self.load_known_faces()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        faces = db.get_all_faces()
        for face in faces:
            self.known_face_ids.append(face['id'])
            self.known_face_names.append(face['name'])
            encoding = np.array(json.loads(face['encoding']))
            self.known_face_encodings.append(encoding)
        print(f"Loaded {len(self.known_face_names)} faces from database.")

    def enroll(self, image_bytes, name):
        # Convert bytes to image
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        
        # Get encoding
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            return False, "No face detected in image."
        
        encoding = encodings[0]
        
        # Save to DB
        face_id = db.add_face(name, encoding)
        
        # Save image file for reference
        upload_dir = os.path.join(BASE_DIR, "uploads", "known")
        os.makedirs(upload_dir, exist_ok=True)
        img = Image.fromarray(image)
        img.save(os.path.join(upload_dir, f"{face_id}_{name}.jpg"))
        
        # Refresh local cache
        self.load_known_faces()
        return True, "Enrolled successfully."

    def recognize(self, frame_np, tolerance=None, scale=0.5):
        """
        frame_np: Opencv BGR frame
        scale: The scale factor applied to this frame relative to original
        """
        if tolerance is None:
            tolerance = self.tolerance
            
        # Surgical Contrast Enhancement (CLAHE)
        # Convert to LAB to boost lightness channel only (prevents color distortion)
        lab = cv2.cvtColor(frame_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = enhanced_bgr[:, :, ::-1]
        
        # Multi-scale detection for maximum sensitivity
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)
        if not face_locations:
            # Deep scan if first pass fails
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
            
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0
            face_id = None
            
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                min_dist = face_distances[best_match_index]
                
                print(f"DEBUG: Face distance to {self.known_face_names[best_match_index]} is {min_dist:.4f}")

                if min_dist <= self.tolerance:
                    name = self.known_face_names[best_match_index]
                    face_id = self.known_face_ids[best_match_index]
                    # Map distance 0.0->1.0 to confidence 100%->0% with tolerance cutoff
                    # If dist = 0.0 -> 100%. If dist = tolerance -> 60%
                    confidence = (1.0 - min_dist) * 100
                else:
                    confidence = max(0, 1.0 - min_dist) * 100
            
            results.append({
                "name": name,
                "face_id": face_id,
                "confidence": round(confidence, 1),
                "bbox": [int(v / scale) for v in [top, right, bottom, left]]
            })
            
            # Log to history (throttled in app.py generally, but we can do it here simple)
            # Actually, we should probably only log to DB if it's a "significant" detection to avoid DB bloat
            
        return results
