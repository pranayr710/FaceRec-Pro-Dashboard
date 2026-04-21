<div align="center">

<br/>

```
███████╗ █████╗  ██████╗███████╗██████╗ ███████╗ ██████╗    ██████╗ ██████╗  ██████╗
██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝    ██╔══██╗██╔══██╗██╔═══██╗
█████╗  ███████║██║     █████╗  ██████╔╝█████╗  ██║         ██████╔╝██████╔╝██║   ██║
██╔══╝  ██╔══██║██║     ██╔══╝  ██╔══██╗██╔══╝  ██║         ██╔═══╝ ██╔══██╗██║   ██║
██║     ██║  ██║╚██████╗███████╗██║  ██║███████╗╚██████╗    ██║     ██║  ██║╚██████╔╝
╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝
```

### **Elite Forensic Biometric Intelligence Platform**
*Real-time face recognition · Gaze tracking · Emotion AI · Fatigue detection*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision_Engine-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-V2_Elite-FF4757?style=for-the-badge)](https://github.com/pranayr710/FaceRec-Pro-Dashboard)

</div>

---

## 📌 Table of Contents

1. [The Problem We're Solving](#-the-problem-were-solving)
2. [Our Solution — How It Works](#-our-solution--how-it-works)
3. [System Architecture](#-system-architecture)
4. [Core Analytics Modules](#-core-analytics-modules)
   - [Face Recognition & Identity](#1--face-recognition--identity-tracking)
   - [Gaze & Iris Tracking](#2--high-precision-gaze--iris-tracking)
   - [Emotion Intelligence](#3--emotion-intelligence-v2)
   - [Head Pose Estimation](#4--head-pose-estimation-3d)
   - [Fatigue Monitoring](#5--fatigue-monitoring-perclos)
5. [Scientific Foundations](#-scientific-foundations)
6. [Technical Architecture Deep-Dive](#-technical-architecture-deep-dive)
7. [Getting Started](#-getting-started)
8. [Repository Structure](#-repository-structure)
9. [License](#-license)

---

## 🎯 The Problem We're Solving

> **Passive presence ≠ Active engagement.**

In high-stakes environments — exam halls, driver cabins, security checkpoints, clinical observation suites — knowing *who* is in the frame is not enough. Organizations routinely struggle with:

| Pain Point | Real-World Impact |
|---|---|
| 🚗 **Driver microsleep** | ~100,000 crashes/year in the US alone are fatigue-related *(NHTSA, 2022)* |
| 📝 **Exam impersonation** | Manual proctoring is inconsistent and labor-intensive |
| 🏭 **Operator inattention** | Industrial accidents spike sharply when PERCLOS > 0.15 |
| 🏥 **Patient non-responsiveness** | Early drowsiness detection is critical in post-op monitoring |
| 🔐 **Security bypass** | Static ID-based access fails without liveness & presence verification |

Existing solutions address these in silos — a separate gaze tracker here, a fatigue alarm there, a face-recognition badge reader elsewhere. **The result is fragmented data, missed correlations, and expensive integration overhead.**

**FaceRec Pro Dashboard solves this by unifying all biometric signals into a single, real-time, SQL-backed forensic intelligence platform.**

---

## 💡 Our Solution — How It Works

FaceRec Pro is built around a single core insight:

> **A face is not just an identity — it is a continuous stream of physiological and cognitive signals.**

The platform captures video from one or more cameras and processes each frame through a **layered analytics pipeline**:

```
  RAW FRAME
      │
      ▼
┌─────────────────────┐
│   CLAHE Enhancement  │  ← Low-light & contrast normalization (LAB color space)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Face Detection &   │  ← Dlib HOG / CNN detector
│   Landmark Mapping   │  ← 68-point facial landmark model
└──────────┬──────────┘
           │
      ┌────┴────┐
      ▼         ▼
┌──────────┐ ┌────────────────────────────────────────────┐
│ Identity │ │           Biometric Signal Extraction       │
│ Pipeline │ │  EAR · MAR · Gaze Vector · Pose Angles     │
└────┬─────┘ └───────────────────────┬────────────────────┘
     │                               │
     ▼                               ▼
┌──────────┐            ┌────────────────────────┐
│ Hungarian│            │  Temporal Smoothing &  │
│ Tracker  │            │  State Classification  │
└────┬─────┘            └───────────┬────────────┘
     │                              │
     └──────────────┬───────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  FastAPI Backend  │
          │  + SQLite Audit   │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Web Dashboard   │  ← Real-time display at 30+ FPS
          └──────────────────┘
```

Each layer runs **concurrently in a multithreaded architecture** so the UI never blocks on computation.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FaceRec Pro — V2 Elite                 │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐  │
│  │  Camera(s)  │───▶│  Vision Core │──▶│  Analytics  │  │
│  │  (RTSP/USB) │    │  (OpenCV +   │   │  Engine     │  │
│  └─────────────┘    │   Dlib)      │   │             │  │
│                     └──────────────┘   │  · EAR/MAR  │  │
│  ┌─────────────┐                       │  · PERCLOS  │  │
│  │  Face DB    │◀──────────────────────│  · PnP Pose │  │
│  │ (Encodings) │    ┌──────────────┐   │  · Emotion  │  │
│  └─────────────┘    │  FastAPI     │◀──└─────────────┘  │
│                     │  Server      │                     │
│  ┌─────────────┐    │  :8000       │   ┌─────────────┐  │
│  │  SQLite     │◀───│              │──▶│  Browser    │  │
│  │  Audit Log  │    └──────────────┘   │  Dashboard  │  │
│  └─────────────┘                       └─────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Analytics Modules

---

### 1. 👤 Face Recognition & Identity Tracking

The system builds a **128-dimensional face encoding** for each known subject using a ResNet-based deep metric learning model (via `face_recognition` / dlib). At runtime, each detected face is matched against the database using Euclidean distance with a configurable threshold.

**The Hungarian Algorithm** is applied across successive frames to achieve optimal one-to-one identity assignment — even in crowded scenes with partial occlusion:

```
Frame N detections:   [A, B, C]
Frame N+1 detections: [A', B', C']

Cost matrix (distance):       Optimal assignment:
     A'   B'   C'                  A → A'
A  [ 0.1  0.9  0.8 ]               B → B'
B  [ 0.8  0.1  0.7 ]               C → C'
C  [ 0.7  0.8  0.2 ]
```

This eliminates ID swaps — a critical requirement for forensic audit trails.

---

### 2. 👁️ High-Precision Gaze & Iris Tracking

**What's being measured:** The angular direction of the subject's gaze, computed from the relative displacement of the pupil center within the eye socket.

**How it works:** Using the 6 landmarks around each eye (detected by Dlib's 68-point model), the system computes the **Eye Aspect Ratio (EAR)** and **pupil-to-canthus displacement vector** to derive a bilateral gaze symmetry score.

```
Eye landmark positions (Dlib indices 36–47):

     p1 ──── p2
    /           \
  p6     👁     p3
    \           /
     p5 ──── p4

Gaze vector = normalize(pupil_center - eye_center)
Symmetry    = 1 - |left_gaze_angle - right_gaze_angle| / π
```

**Metric — Bilateral Symmetry Quality (BSQ):**
- **BSQ ≈ 1.0** → Both eyes converging on screen → High focus
- **BSQ < 0.7** → Divergent gaze → Distraction or disengagement

> 📖 *Reference: Villanueva & Cabeza, "A Novel Gaze Estimation System with One Calibration Point," IEEE Trans. Systems, Man, and Cybernetics, 2013.*

---

### 3. 😐 Emotion Intelligence V2

**What's being measured:** Real-time affective state derived from facial muscle configuration, modeled via valence-arousal dimensions.

**How it works:** Rather than a black-box neural classifier, FaceRec Pro uses **interpretable geometric heuristics** computed directly from facial landmarks — making the system fast, explainable, and auditable.

```
Emotion State Machine:

         High EAR
            │
     ┌──────┴──────┐
  High MAR       Low MAR
     │               │
  SURPRISED       FOCUSED ←── EAR normal, BSQ high
     │
  High temporal
  MAR variance
     │
   YAWNING / DROWSY ←── Low EAR + High MAR

  Happy: MAR elevated + lip corner displacement > threshold
  Neutral: all metrics in baseline range
```

**Temporal smoothing** (exponential moving average, α = 0.3) prevents jitter and ensures displayed emotions represent genuine sustained states rather than single-frame noise.

**Valence-Arousal mapping:**

| State | Valence | Arousal |
|---|---|---|
| Neutral | 0 | 0 |
| Happy | + | + |
| Surprised | 0 | ++ |
| Focused | 0 | + |
| Drowsy | − | −− |

> 📖 *Reference: Ekman, P. (1978). "Facial Action Coding System." Consulting Psychologists Press. & Russell, J.A. (1980). "A circumplex model of affect." J. Personality and Social Psychology.*

---

### 4. 🧭 Head Pose Estimation (3D)

**What's being measured:** The 3D orientation of the head in space — yaw (left/right), pitch (up/down), and roll (tilt) — expressed as Euler angles.

**How it works:** OpenCV's `solvePnP` solves the **Perspective-n-Point (PnP)** problem: given 6 known 3D world coordinates (a generic face model) and their corresponding 2D projections in the image, it recovers the rotation matrix **R** and translation vector **t** that describe the camera-to-head transformation.

```
3D Generic Face Model Points:       2D Image Landmarks:
  · Nose tip    (0, 0, 0)            → detected by Dlib
  · Chin        (0, -63, -13)
  · Left eye corner                  solvePnP solves:
  · Right eye corner                 minimize Σ || proj(R·X + t) - x ||²
  · Left mouth corner
  · Right mouth corner

Output:
  Yaw   → -30°..+30°  = on screen
  Pitch → -20°..+20°  = attentive
  Roll  → ±15°        = stable posture
```

**Attention Score** is a proprietary composite index:

```
Attention = α·(1 - |Yaw|/90) + β·(1 - |Pitch|/90) + γ·BSQ
           where α=0.4, β=0.4, γ=0.2
```

> 📖 *Reference: Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). "EPnP: An Accurate O(n) Solution to the PnP Problem." IJCV.*

---

### 5. 😴 Fatigue Monitoring (PERCLOS)

**What's being measured:** The **Percentage of Eye Closure (PERCLOS)** — the proportion of time in a rolling window during which the eyes are more than 80% closed.

**Why it matters:** PERCLOS is the single most validated objective measure of drowsiness, endorsed by the USDOT as the gold standard for driver fatigue detection.

**How it works:**

```
Eye Aspect Ratio (EAR):

         ||p2-p6|| + ||p3-p5||
EAR  =  ─────────────────────
              2·||p1-p4||

       p2  p3
      /      \
    p1        p4
      \      /
       p6  p5

EAR ≈ 0.30  →  Eyes open
EAR ≈ 0.15  →  Eyes ~80% closed (counts toward PERCLOS)
EAR < 0.10  →  Blink / microsleep event
```

**PERCLOS Calculation (60-second rolling window):**

```
PERCLOS = (frames where EAR < 0.15) / (total frames) × 100%
```

**Alert States:**

| Level | PERCLOS | Meaning | Action |
|---|---|---|---|
| 🟢 NOMINAL | < 8% | Alert and rested | Log only |
| 🟡 ELEVATED | 8–15% | Mild fatigue onset | Dashboard warning |
| 🔴 HIGH_RISK | > 15% | Acute drowsiness | Alert + audit record |

> 📖 *Reference: Wierwille, W.W. & Ellsworth, L.A. (1994). "Evaluation of driver drowsiness by trained raters." Accident Analysis & Prevention. — The foundational PERCLOS paper.*

---

## 📐 Scientific Foundations

All biometric computations in FaceRec Pro are grounded in peer-reviewed research:

### Eye Aspect Ratio (EAR) — Soukupová & Čech, 2016

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2\,||p_1 - p_4||}$$

Where $p_1 \ldots p_6$ are the six 2D landmark coordinates around the eye, ordered clockwise from the outer canthus. This formulation is invariant to absolute eye size and robust to minor head roll (< 15°).

> *Source: Soukupová, T. & Čech, J. (2016). "Real-Time Eye Blink Detection Using Facial Landmarks." 21st Computer Vision Winter Workshop.*

### Mouth Aspect Ratio (MAR)

$$\text{MAR} = \frac{||p_2 - p_8|| + ||p_3 - p_7|| + ||p_4 - p_6||}{3\,||p_1 - p_5||}$$

Analogous to EAR but applied to the 8 outer lip landmarks. Used to distinguish yawning (high sustained MAR), speech (oscillating MAR), and baseline closed-mouth state.

### Head Pose — Perspective-n-Point (PnP)

Given a set of $n$ 3D world points $\mathbf{X}_i$ and their 2D projections $\mathbf{x}_i$, PnP finds $\mathbf{R}, \mathbf{t}$ minimizing reprojection error:

$$\min_{\mathbf{R},\mathbf{t}} \sum_{i=1}^{n} \left\| \mathbf{x}_i - \pi\left(\mathbf{K}\left[\mathbf{R}\,|\,\mathbf{t}\right]\mathbf{X}_i\right) \right\|^2$$

Where $\mathbf{K}$ is the camera intrinsic matrix and $\pi$ is the perspective projection. EPnP solves this in $O(n)$ time.

---

## ⚙️ Technical Architecture Deep-Dive

### CLAHE — Low-Light Robustness

Ordinary histogram equalization globally stretches contrast, washing out bright regions. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** divides the image into tiles and equalizes each locally, with a clip limit preventing over-amplification of noise.

```
Standard HE: global redistribution → flat histogram
CLAHE:       tile-wise redistribution + clip limit → local contrast,
             applied in LAB L-channel only → no hue distortion
```

This is critical for reliable EAR/MAR computation in office lighting, vehicle dashboards, and dim corridors.

### Hungarian Algorithm — Zero ID-Swap Tracking

Between frames, detected face bounding boxes must be matched to known track IDs. Naive nearest-neighbour fails in crowded scenes. The **Hungarian (Kuhn-Munkres) algorithm** finds the globally optimal assignment in $O(n^3)$:

```
Cost matrix C[i][j] = IoU distance between track i and detection j
Hungarian solution: minimize Σ C[i, assignment[i]]
```

This guarantees each identity is assigned exactly once per frame, eliminating the ID flickers that corrupt audit logs.

### Multithreaded Architecture

```
Main Thread          Recognition Thread     Analytics Thread
───────────          ──────────────────     ────────────────
Camera capture   →   Encode faces       →   EAR / MAR / Pose
UI rendering     ←   Match identities   ←   PERCLOS / Emotion
SQLite writes        Hungarian assign.      Attention score
```

The UI thread never blocks on vision computation. Result queues decouple all three layers, maintaining **30+ FPS** display regardless of recognition latency.

---

## 🚀 Getting Started

### Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Runtime |
| FastAPI + Uvicorn | latest | Web server |
| OpenCV | 4.x | Vision engine |
| Dlib | 19.x | Face detection & landmarks |
| face_recognition | 1.3+ | Deep face encoding |
| SQLite3 | built-in | Audit persistence |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pranayr710/FaceRec-Pro-Dashboard.git
cd FaceRec-Pro-Dashboard

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the server
python FaceRecPro/app.py
```

Navigate to **`http://localhost:8000`** — the dashboard will be live.

### Registering Faces

```bash
# Add a subject to the recognition database
python FaceRecPro/register.py --name "John Doe" --images ./photos/john/

# Verify registration
python FaceRecPro/verify_db.py --list
```

### Configuration

Key parameters in `FaceRecPro/config.py`:

```python
EAR_THRESHOLD       = 0.15   # Below this → eye considered closed
MAR_THRESHOLD       = 0.65   # Above this → mouth considered open
PERCLOS_WINDOW_SEC  = 60     # Rolling window for fatigue calculation
ATTENTION_ALPHA     = 0.4    # Weight: yaw contribution to attention score
ATTENTION_BETA      = 0.4    # Weight: pitch contribution
ATTENTION_GAMMA     = 0.2    # Weight: gaze symmetry contribution
SMOOTHING_ALPHA     = 0.30   # EMA factor for emotion smoothing
MATCH_THRESHOLD     = 0.55   # Face recognition distance threshold
```

---

## 📁 Repository Structure

```
FaceRec-Pro-Dashboard/
│
├── FaceRecPro/
│   ├── app.py                  # FastAPI entry point + server startup
│   ├── config.py               # All tuneable parameters
│   ├── vision/
│   │   ├── detector.py         # Face detection & landmark extraction
│   │   ├── tracker.py          # Hungarian multi-object tracker
│   │   ├── gaze.py             # Gaze vector & BSQ computation
│   │   ├── pose.py             # solvePnP head pose estimation
│   │   ├── fatigue.py          # PERCLOS + EAR blink detection
│   │   └── emotion.py          # MAR/EAR emotion state machine
│   ├── recognition/
│   │   ├── encoder.py          # 128-dim face encoding
│   │   ├── database.py         # Face DB load / save / query
│   │   └── register.py         # Subject enrollment CLI
│   ├── analytics/
│   │   ├── attention.py        # Composite attention score
│   │   └── audit.py            # SQLite event logging
│   ├── api/
│   │   └── routes.py           # REST + WebSocket endpoints
│   └── static/                 # Dashboard frontend assets
│
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** The `legacy-main` branch preserves the original V1 stable build. The current `main` branch is the **V2 Elite** release with full forensic capabilities.

---

## 🗺️ Roadmap

- [ ] **Multi-camera fusion** — synchronized streams with cross-camera re-identification
- [ ] **Liveness detection** — anti-spoofing via 3D depth estimation
- [ ] **Action Unit (AU) mapping** — FACS-compliant emotion analysis
- [ ] **Edge deployment** — Jetson Nano / Raspberry Pi 5 optimization
- [ ] **GDPR compliance module** — anonymization, data retention policies, consent logging

---

## 📜 References

1. Soukupová, T. & Čech, J. (2016). *Real-Time Eye Blink Detection Using Facial Landmarks.* 21st CVWW.
2. Wierwille, W.W. & Ellsworth, L.A. (1994). *Evaluation of driver drowsiness by trained raters.* Accident Analysis & Prevention, 26(5).
3. Lepetit, V., Moreno-Noguer, F. & Fua, P. (2009). *EPnP: An Accurate O(n) Solution to the PnP Problem.* IJCV, 81(2).
4. Villanueva, A. & Cabeza, R. (2013). *A Novel Gaze Estimation System with One Calibration Point.* IEEE T-SMC.
5. Russell, J.A. (1980). *A circumplex model of affect.* J. Personality and Social Psychology, 39(6).
6. Kazemi, V. & Sullivan, J. (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees.* CVPR.
7. NHTSA (2022). *Drowsy Driving.* National Highway Traffic Safety Administration.

---

<div align="center">

**FaceRec Pro Dashboard** — Built with ❤️ and a lot of EAR math

[![GitHub](https://img.shields.io/badge/GitHub-pranayr710%2FFaceRec--Pro--Dashboard-181717?style=for-the-badge&logo=github)](https://github.com/pranayr710/FaceRec-Pro-Dashboard)

*Star ⭐ the repo if this helped you!*

</div>
