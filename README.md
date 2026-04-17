# FaceRec Pro Dashboard 🚀

A professional-grade, real-time face recognition system featuring a web-based dashboard, SQL logging, and advanced image enhancement. Built on the foundation of the world's simplest face recognition library.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen)
![Tech Stack](https://img.shields.io/badge/Tech-FastAPI%20|%20OpenCV%20|%20dlib-blue)

## 🌟 Key Features

*   **Real-Time Analytics**: High-performance multi-threaded video engine delivering a smooth 30 FPS experience.
*   **"Surgical" Image Processing**: Advanced **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and multi-scale detection for maximum sensitivity in challenging lighting.
*   **Face Management**: Easy-to-use web interface for enrolling new faces and managing profiles.
*   **Security Logs**: Persistent SQL-backed activity history tracking recognized faces and confidence levels.
*   **Modern Web UI**: Responsive dashboard built with FastAPI and vanilla JS/CSS for speed and reliability.

## 🛠️ Technical Highlights

### Contrast Limited Adaptive Histogram Equalization (CLAHE)
The system uses professional-grade image pre-processing to handle backlighting and deep shadows. By converting frames to LAB color space and applying CLAHE to the lightness channel, we boost feature visibility without distorting colors.

### High-Performance Video Engine
To prevent UI lag and camera conflicts, the recognition logic runs in a dedicated background thread. This ensures the live stream remains fluid while the recognition model performs deep scans of every few frames.

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   A webcam or video source

### Installation

1. **Clone and Setup Environment**:
   ```bash
   git clone https://github.com/pranayr710/FaceRec-Pro-Dashboard.git
   cd FaceRec-Pro-Dashboard
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn opencv-python-headless dlib face_recognition numpy pillow
   ```

### Running the Dashboard

Launch the application using the integrated starter:
```bash
python FaceRecPro/app.py
```
After starting, open your browser and navigate to:
**`http://localhost:8000`**

---

## 📚 Core Library Background

The recognition engine is powered by [dlib](http://dlib.net/)'s state-of-the-art face recognition built with deep learning.

*   **Accuracy**: 99.38% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.
*   **Features**:
    *   Find faces in pictures
    *   Manipulate facial features (eyes, nose, mouth)
    *   Identify faces across different angles and lighting

### Command Line Usage
The library also provides a CLI tool for batch processing:
```bash
face_recognition ./known_people/ ./unknown_pictures/
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Thanks
*   **Davis King** (@nulhom) for creating dlib and the trained models.
*   **Adam Geitgey** for the original `face_recognition` library.
