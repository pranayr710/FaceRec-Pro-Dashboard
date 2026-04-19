import cv2
import numpy as np
import os
import sys
import time

# Add app directory to path
sys.path.append(os.path.join(os.getcwd(), 'FaceRecPro'))

try:
    from face_engine import FaceEngine
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_diagnostic():
    print("=== Elite Edition System Diagnostic ===")
    engine = FaceEngine()
    
    test_image_path = "FaceRecPro/uploads/known/2_meghapranay.jpg"
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return

    frame = cv2.imread(test_image_path)
    if frame is None:
        print("Error: Could not read test image.")
        return

    print(f"Processing frame: {frame.shape}")
    
    start_time = time.time()
    # Run recognition (Full scale 1.0 for maximum accuracy in diagnostic)
    engine_out = engine.recognize(frame, scale=1.0)
    latency = (time.time() - start_time) * 1000

    print(f"\n[AI Engine Health]")
    print(f"Latency: {latency:.2f}ms")
    print(f"Brightness: {engine_out.get('brightness', 'N/A')}")
    
    detections = engine_out.get('detections', [])
    print(f"Detections Found: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"\n--- Detection #{i+1} ---")
        print(f"Identity: {det.get('name', 'Unknown')}")
        print(f"Confidence: {det.get('confidence', 0)}%")
        
        # Pose
        pose = det.get('pose', {})
        print(f"Pose (P/Y/R): {pose.get('pitch', 0)} / {pose.get('yaw', 0)} / {pose.get('roll', 0)}")
        
        # Liveness/EAR
        ear = det.get('ear', 0)
        print(f"EAR (Blink Data): {ear}")
        
        # Emotion
        emotion = det.get('emotion', 'Unknown')
        print(f"Dominant Emotion: {emotion}")
        
    print("\n[Forensic Verification]")
    # Check landmarks consistency
    test_landmarks = engine.get_landmarks(frame)
    if test_landmarks and len(test_landmarks) > 0:
        print(f"Landmark Tracking: OK ({len(test_landmarks[0])} keypoints)")
    else:
        print("Landmark Tracking: FAILED")

if __name__ == "__main__":
    run_diagnostic()
