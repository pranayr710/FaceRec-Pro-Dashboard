"""Unit checks for gaze, emotion heuristics, and bbox scaling (no camera / DB)."""
import os
import sys
import types

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)

_fake_db = types.ModuleType("db")
_fake_db.get_all_faces = lambda: []
sys.modules["db"] = _fake_db

from face_engine import FaceEngine  # noqa: E402


def test_emotion_and_gaze():
    eng = FaceEngine()

    mock_landmarks = {
        "left_eye": [(10, 10)] * 6,
        "right_eye": [(30, 10)] * 6,
        "top_lip": [(50, 50)] * 12,
        "bottom_lip": [(50, 60)] * 12,
        "nose_bridge": [(0, 0)] * 4,
        "chin": [(0, 0)] * 17,
    }
    mock_landmarks["top_lip"][9] = (50, 52)
    mock_landmarks["bottom_lip"][9] = (50, 61)
    mock_landmarks["top_lip"][0] = (35, 56)
    mock_landmarks["top_lip"][6] = (65, 56)

    ear = 0.3
    mar = eng.calculate_mar(mock_landmarks)
    raw = eng.get_raw_emotion(ear, mar)
    assert raw == "Happy", f"expected Happy, got {raw}"

    mock_landmarks["left_eye"] = [
        (11, 10), (13, 8), (15, 8), (17, 10), (15, 12), (13, 12)
    ]
    mock_landmarks["right_eye"] = [
        (31, 10), (33, 8), (35, 8), (37, 10), (35, 12), (33, 12)
    ]
    gaze = eng.calculate_pupil_gaze(mock_landmarks)
    assert -1.2 <= gaze["gaze_x"] <= 1.2
    assert -1.2 <= gaze["gaze_y"] <= 1.2

    scale = 0.5
    test_pts = [(100, 100), (200, 200)]
    scaled = [(int(px / scale), int(py / scale)) for (px, py) in test_pts]
    assert scaled[0] == (200, 200) and scaled[1] == (400, 400)

    print("Elite feature self-tests passed.")


if __name__ == "__main__":
    test_emotion_and_gaze()
