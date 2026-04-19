# FaceRec Pro Dashboard: Elite Forensic Analytics 🚀

[![Status](https://img.shields.io/badge/Status-Industrial--Grade-brightgreen)](https://github.com/pranayr710/FaceRec-Pro-Dashboard)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue)](https://github.com/pranayr710/FaceRec-Pro-Dashboard)
[![Intelligence](https://img.shields.io/badge/AI-Computer%20Vision%20%2B%20Psychometrics-blueviolet)](https://github.com/pranayr710/FaceRec-Pro-Dashboard)

A professional-grade, real-time forensic face recognition ecosystem. Beyond simple identification, this platform integrates advanced biometric heuristics and psychometric modeling to provide deep insights into human attention, emotion, and presence.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-V2%20Elite-orange)

---

## 🌟 Elite "Forensic" Features

### 1. High-Precision Gaze & Iris Tracking
The system computes real-time gaze vectors by analyzing the relative displacement between the pupil center and the eye corners (canthi).
*   **Metric**: Bilateral Symmetry Quality.
*   **Insight**: Measures how well both eyes are converging on the screen, indicating focus or distraction.

### 2. Emotion Intelligence (V2)
Utilizing temporal smoothing and valence/arousal mapping, the dashboard provides a stabilized readout of the subject's emotional state.
*   **States**: Neutral, Happy, Surprised, Focused, Drowsy.
*   **Heuristics**: Derived from real-time **MAR** (Mouth Aspect Ratio) and **EAR** (Eye Aspect Ratio).

### 3. Head Pose Estimation (3D Projection)
Using a 6-point 3D generic face model paired with OpenCV's `solvePnP` algorithm, the system calculates orientation in 3D space.
*   **Axes**: Yaw (Left-Right), Pitch (Up-Down), Roll (Tilt).
*   **Attention Score**: A proprietary index combining head frontalization and gaze stability to measure active engagement.

### 4. Fatigue Monitoring (PERCLOS)
Implements the industrial Standard **PERCLOS** (Percentage of Eye Closure) to detect micro-sleeps and chronic fatigue.
*   **Algorithm**: Advanced EAR modeling.
*   **Alerts**: NOMINAL, ELEVATED, and HIGH_RISK states based on cumulative eye-closure duration.

---

## 📚 Conceptual Theory

### Eye Aspect Ratio (EAR)
The **EAR** is used to monitor eye openness and detect blinks. It is calculated from six facial landmarks detected around each eye.

$$ \text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||} $$

### Mouth Aspect Ratio (MAR)
Similar to EAR, the **MAR** measures the vertical opening of the mouth relative to its width, used to detect speech, yawning (Fatigue), or "Happy/Surprised" emotions.

### Head Pose via solvePnP
The system uses the **Perspective-n-Point (PnP)** pose computation algorithm. By mapping 2D landmarks (nose tip, chin, eye corners) to a static 3D model, we derive the rotation matrix and Euler angles.

---

## 🛠️ Technical Highlights

*   **Multithreaded Recognition**: Background processing thread ensures zero-lag UI performance (30+ FPS).
*   **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization in the **LAB Color Space** for optimal performance in low-light environments.
*   **Hungarian Tracking**: Optimal one-to-one identity assignment between frames to minimize ID swaps in crowded scenes.
*   **SQL-Backed Audit Log**: Persistent storage of every detection, emotion change, and fatigue alert for retrospective analysis.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   FastAPI & Uvicorn (Web Core)
*   OpenCV & Dlib (Vision Engine)
*   SQLite3 (Audit Logging)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranayr710/FaceRec-Pro-Dashboard.git
   cd FaceRec-Pro-Dashboard
   ```
2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run Application**:
   ```bash
   python FaceRecPro/app.py
   ```
Navigate to `http://localhost:8000` to access the dashboard.

---

## 📜 Repository Structure Update
> [!IMPORTANT]
> The previous stable version of this repository has been archived as the `legacy-main` branch. The current `main` branch represents the new **V2 Elite** development with full forensic capabilities.

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
