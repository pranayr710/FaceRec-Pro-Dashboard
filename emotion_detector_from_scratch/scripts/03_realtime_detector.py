import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'emotion_cnn_model.h5')

# Ensure the model file exists
if not os.path.exists(MODEL_SAVE_PATH):
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
    print("Please ensure '02_train_model.py' was run successfully and the model was saved.")
    exit()

# Load the trained emotion detection model
print(f"Loading emotion detection model from {MODEL_SAVE_PATH}...")
try:
    model = load_model(MODEL_SAVE_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have the correct TensorFlow version and the model file is not corrupted.")
    exit()

# Load the Haar cascade classifier for face detection
# This XML file is usually included with OpenCV.
# You might need to provide the full path if it's not found automatically.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading Haar cascade: {e}")
    print("Please ensure 'haarcascade_frontalface_default.xml' is in your OpenCV data directory or provide its full path.")
    exit()

# Define emotion labels (must match the order used during training)
emotion_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Please check if your webcam is connected and not in use by another application.")
    exit()

print("\nStarting real-time emotion detection using your custom model.")
print("Press 'q' to quit the application.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the frame to grayscale for face detection and model input
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region of interest (ROI)
        face_roi = gray_frame[y:y+h, x:x+w]

        # Resize the face ROI to 48x48 pixels (model input size)
        resized_face = cv2.resize(face_roi, (48, 48))

        # Normalize the pixels to [0, 1] and add batch/channel dimensions
        # Model expects shape (batch_size, 48, 48, 1)
        normalized_face = resized_face / 255.0
        model_input = np.expand_dims(np.expand_dims(normalized_face, axis=-1), axis=0)

        # Predict the emotion
        try:
            predictions = model.predict(model_input, verbose=0)[0]
            predicted_emotion_index = np.argmax(predictions)
            dominant_emotion = emotion_labels[predicted_emotion_index]
            confidence = predictions[predicted_emotion_index] * 100

            # Display the dominant emotion and its confidence
            text = f"{dominant_emotion} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            # Handle prediction errors (e.g., if model input shape is wrong)
            cv2.putText(frame, "Prediction Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(f"Prediction error: {e}") # Uncomment for debugging

    # Display the frame with detected faces and emotions
    cv2.imshow('Real-time Emotion Detector (Custom Model)', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("\nApplication stopped.")