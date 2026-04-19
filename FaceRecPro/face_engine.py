import face_recognition
import cv2
import numpy as np
import json
import io
import db
import config
from collections import deque
import time
import math

# 3D model points (generic face) for solvePnP head pose — nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
_HEAD_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -330.0, -65.0],  # chin
        [-225.0, 170.0, -135.0],  # left eye left corner
        [225.0, 170.0, -135.0],  # right eye right corner
        [-150.0, -150.0, -125.0],  # left mouth corner
        [150.0, -150.0, -125.0],  # right mouth corner
    ],
    dtype=np.float64,
)


class FaceEngine:
    def __init__(self):
        self.tolerance = config.MATCH_TOLERANCE
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.max_history = 50
        self.smooth_window = 7
        self._clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_GRID
        )
        self._next_track_id = 1
        self._tracks = {}
        self._last_history_log = {}
        self.load_known_faces()

    def load_known_faces(self):
        try:
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_ids = []
            faces = db.get_all_faces()
            for face in faces:
                self.known_face_ids.append(face["id"])
                self.known_face_names.append(face["name"])
                self.known_face_encodings.append(np.array(json.loads(face["encoding"])))
            print(f"Loaded {len(self.known_face_names)} enrolled identities (tolerance={self.tolerance}).")
        except Exception as e:
            print(f"Engine load error: {e}")

    def _apply_clahe_lab(self, bgr):
        if not config.USE_CLAHE:
            return bgr
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def enroll(self, image_bytes, name):
        try:
            image = face_recognition.load_image_file(io.BytesIO(image_bytes))
            locs = face_recognition.face_locations(image)
            if not locs:
                return False, "No face detected. Use a clear, front-facing photo."
            encodings = face_recognition.face_encodings(
                image, locs, num_jitters=config.ENROLL_NUM_JITTERS
            )
            if not encodings:
                return False, "Could not compute a stable encoding. Try better lighting."
            if len(locs) > 1:
                return False, "Multiple faces found. Crop to one person and retry."
            db.add_face(name, encodings[0])
            self.load_known_faces()
            return True, "Identity enrolled successfully."
        except Exception as e:
            return False, str(e)

    def _centroid(self, bbox):
        top, right, bottom, left = bbox
        return (0.5 * (left + right), 0.5 * (top + bottom))

    def _iou(self, a, b):
        at, ar, ab, al = a
        bt, br, bb, bl = b
        inter_l = max(al, bl)
        inter_t = max(at, bt)
        inter_r = min(ar, br)
        inter_b = min(ab, bb)
        if inter_r <= inter_l or inter_b <= inter_t:
            return 0.0
        inter = (inter_r - inter_l) * (inter_b - inter_t)
        ua = (ar - al) * (ab - at) + (br - bl) * (bb - bt) - inter
        return inter / (ua + 1e-6)

    def _assign_tracks(self, bboxes_full, now):
        """
        Assign a stable numeric track_id per face across frames.

        Faces are matched in reading order (top, then left) so greedy matching
        does not arbitrarily favour whichever box appeared first in the detector
        list — that reduces ID swaps when two people overlap or cross.
        """
        n = len(bboxes_full)
        if n == 0:
            return []

        try:
            from scipy.optimize import linear_sum_assignment

            return self._assign_tracks_hungarian(
                bboxes_full, now, linear_sum_assignment
            )
        except ImportError:
            pass

        order = sorted(range(n), key=lambda i: (bboxes_full[i][0], bboxes_full[i][3]))
        assigned_ids = [0] * n
        used = set()
        for idx in order:
            bbox = bboxes_full[idx]
            cx, cy = self._centroid(bbox)
            best_tid = None
            best_score = -1.0
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                iou = self._iou(bbox, tr["bbox"])
                oc = tr["centroid"]
                dist = math.hypot(cx - oc[0], cy - oc[1])
                score = iou * 2.5 + max(0, 1.0 - dist / config.TRACK_MATCH_MAX_DIST)
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is None or best_score < 0.35:
                best_tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_tid] = self._new_track_state()
            tr = self._tracks[best_tid]
            tr["bbox"] = bbox
            tr["centroid"] = (cx, cy)
            tr["last_t"] = now
            used.add(best_tid)
            assigned_ids[idx] = best_tid

        stale = [
            tid
            for tid, tr in self._tracks.items()
            if now - tr["last_t"] > 2.5 and tid not in used
        ]
        for tid in stale:
            del self._tracks[tid]
        return assigned_ids

    def _assign_tracks_hungarian(self, bboxes_full, now, linear_sum_assignment):
        """Optimal one-to-one matching (scipy) — reduces swaps with 2+ faces."""
        n = len(bboxes_full)
        candidates = [(tid, tr) for tid, tr in self._tracks.items()]
        m = len(candidates)
        used = set()
        assigned_ids = [0] * n
        max_cost_match = config.TRACK_MATCH_MAX_DIST * 2.2

        if m == 0:
            for j in range(n):
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = self._new_track_state()
                bbox = bboxes_full[j]
                cx, cy = self._centroid(bbox)
                self._tracks[tid].update(
                    {"bbox": bbox, "centroid": (cx, cy), "last_t": now}
                )
                assigned_ids[j] = tid
                used.add(tid)
            stale = [
                tid
                for tid, tr in self._tracks.items()
                if now - tr["last_t"] > 2.5 and tid not in used
            ]
            for tid in stale:
                del self._tracks[tid]
            return assigned_ids

        dim = max(m, n)
        big = 1e5
        cost = np.full((dim, dim), big, dtype=np.float64)
        for i in range(m):
            oc = candidates[i][1]["centroid"]
            prev = candidates[i][1]["bbox"]
            for j in range(n):
                bbox = bboxes_full[j]
                cx, cy = self._centroid(bbox)
                dist = math.hypot(cx - oc[0], cy - oc[1])
                iou = self._iou(prev, bbox)
                cost[i, j] = dist / (iou + 0.12)

        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if r >= m or c >= n:
                continue
            if cost[r, c] >= max_cost_match:
                continue
            tid = candidates[r][0]
            bbox = bboxes_full[c]
            cx, cy = self._centroid(bbox)
            self._tracks[tid].update(
                {"bbox": bbox, "centroid": (cx, cy), "last_t": now}
            )
            used.add(tid)
            assigned_ids[c] = tid

        for j in range(n):
            if assigned_ids[j] != 0:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = self._new_track_state()
            bbox = bboxes_full[j]
            cx, cy = self._centroid(bbox)
            self._tracks[tid].update(
                {"bbox": bbox, "centroid": (cx, cy), "last_t": now}
            )
            used.add(tid)
            assigned_ids[j] = tid

        stale = [
            tid
            for tid, tr in self._tracks.items()
            if now - tr["last_t"] > 2.5 and tid not in used
        ]
        for tid in stale:
            del self._tracks[tid]
        return assigned_ids

    def _new_track_state(self):
        return {
            "votes": deque(maxlen=config.IDENTITY_VOTE_WINDOW),
            "ear_hist": deque(maxlen=self.max_history),
            "gx_hist": deque(maxlen=self.max_history),
            "gy_hist": deque(maxlen=self.max_history),
            "t_hist": deque(maxlen=self.max_history),
            "emo_hist": deque(maxlen=self.smooth_window),
            "blink_times": deque(maxlen=40),
            "eyes_closed": False,
            "mar_hist": deque(maxlen=15),
        }

    def _vote_identity(self, track):
        if len(track["votes"]) < max(4, config.IDENTITY_VOTE_WINDOW // 2):
            return "Unknown", None, 0.0
        names = [v[0] for v in track["votes"]]
        best_name = max(set(names), key=names.count)
        ratio = names.count(best_name) / len(names)
        if best_name == "Unknown" or ratio < config.IDENTITY_VOTE_MIN_RATIO:
            return "Unknown", None, 0.0
        dists = [v[2] for v in track["votes"] if v[0] == best_name]
        mean_d = float(np.mean(dists)) if dists else 1.0
        if mean_d > self.tolerance + 0.04:
            return "Unknown", None, 0.0
        fid = next(
            (v[1] for v in reversed(track["votes"]) if v[0] == best_name and v[1] is not None),
            None,
        )
        conf = round((1.0 - mean_d) * 100, 1)
        return best_name, fid, conf

    def _update_blink(self, track, ear, now):
        closed = ear < 0.195
        if closed and not track["eyes_closed"]:
            track["eyes_closed"] = True
        elif not closed and track["eyes_closed"]:
            track["eyes_closed"] = False
            track["blink_times"].append(now)
        open_ears = [e for e in track["ear_hist"] if e > 0.22]
        baseline = float(np.mean(open_ears[-20:])) if len(open_ears) >= 5 else 0.28
        tmin = now - 60.0
        blinks_in_min = sum(1 for t in track["blink_times"] if t >= tmin)
        return baseline, blinks_in_min

    def _head_pose_deg(self, lm, image_shape):
        """PnP head pose from 2D landmarks — robust yaw/pitch/roll for attention & engagement."""
        h, w = image_shape[:2]
        if not all(
            k in lm
            for k in (
                "nose_tip",
                "chin",
                "left_eye",
                "right_eye",
                "top_lip",
            )
        ):
            return {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
        try:
            nose_tip = np.array(lm["nose_tip"][len(lm["nose_tip"]) // 2], dtype=np.float64)
            chin = np.array(lm["chin"][8], dtype=np.float64)
            left_eye = np.array(lm["left_eye"][0], dtype=np.float64)
            right_eye = np.array(lm["right_eye"][3], dtype=np.float64)
            mouth_l = np.array(lm["top_lip"][0], dtype=np.float64)
            mouth_r = np.array(lm["top_lip"][6], dtype=np.float64)
            pts_2d = np.array(
                [nose_tip, chin, left_eye, right_eye, mouth_l, mouth_r], dtype=np.float64
            ).reshape((6, 1, 2))
            obj_pts = _HEAD_MODEL_3D.reshape((6, 1, 3))
            focal = w
            center = (w / 2.0, h / 2.0)
            cam = np.array(
                [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
            dist = np.zeros((4, 1))
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, pts_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
            rot, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = self._rotation_to_head_angles(rot)
            yaw = max(-90.0, min(90.0, yaw))
            pitch = max(-90.0, min(90.0, pitch))
            roll = max(-90.0, min(90.0, roll))
            return {"yaw_deg": round(yaw, 1), "pitch_deg": round(pitch, 1), "roll_deg": round(roll, 1)}
        except Exception:
            le = np.mean(lm["left_eye"], axis=0)
            re = np.mean(lm["right_eye"], axis=0)
            mid = (le + re) / 2.0
            nt = np.array(lm["nose_tip"][2])
            fw = np.linalg.norm(re - le) + 1e-6
            yaw = float(np.degrees(np.arctan((nt[0] - mid[0]) / (0.45 * fw))))
            roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
            return {
                "yaw_deg": round(max(-90, min(90, yaw)), 1),
                "pitch_deg": 0.0,
                "roll_deg": round(max(-90, min(90, roll)), 1),
            }

    def _rotation_to_head_angles(self, R):
        """Euler angles (degrees) from rotation matrix; Y ~ left-right, X ~ nod, Z ~ tilt."""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy < 1e-6:
            roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            pitch = math.degrees(math.atan2(-R[2, 0], sy))
            yaw = 0.0
        else:
            roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            pitch = math.degrees(math.atan2(-R[2, 0], sy))
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return yaw, pitch, roll

    def _gaze_stability(self, track):
        if len(track["gx_hist"]) < 5:
            return 100.0
        gx = list(track["gx_hist"])
        gy = list(track["gy_hist"])
        sx = float(np.std(gx))
        sy = float(np.std(gy))
        score = 100.0 / (1.0 + 45.0 * (sx + sy))
        return round(max(0.0, min(100.0, score)), 1)

    def _screen_attention(self, gaze, head_pose):
        """Camera-as-screen proxy: central gaze + frontal head = attending."""
        gx, gy = gaze["gaze_x"], gaze["gaze_y"]
        yaw = abs(head_pose.get("yaw_deg", 0))
        pitch = abs(head_pose.get("pitch_deg", 0))
        g_term = max(0.0, 1.0 - (abs(gx) + abs(gy)) * 0.55)
        h_term = max(0.0, 1.0 - (yaw + pitch) / 90.0)
        return round(100.0 * (0.55 * g_term + 0.45 * h_term), 1)

    def _fatigue_index(self, perclos, ear, blink_bpm, pitch_deg):
        ear_f = max(0.0, min(100.0, (0.26 - min(ear, 0.26)) / 0.26 * 100.0))
        blink_f = max(0.0, min(100.0, (25.0 - min(blink_bpm, 40.0)) / 25.0 * 60.0))
        pitch_f = max(0.0, min(100.0, max(0.0, pitch_deg - 8.0) / 25.0 * 100.0))
        score = 0.45 * perclos + 0.25 * ear_f + 0.15 * blink_f + 0.15 * pitch_f
        return round(max(0.0, min(100.0, score)), 1)

    def _emotion_confidence(self, emo_hist):
        if not emo_hist:
            return 0.0
        labels = list(emo_hist)
        mode = max(set(labels), key=labels.count)
        return round(100.0 * labels.count(mode) / len(labels), 1)

    def _micro_expression_activity(self, track):
        """High variance in MAR over short window suggests facial micro-movement."""
        if len(track["mar_hist"]) < 5:
            return 0.0
        m = list(track["mar_hist"])
        return round(float(np.std(m)) * 200.0, 1)

    def get_raw_emotion(self, ear, mar):
        if ear < 0.21:
            return "Drowsy"
        if mar > 0.45:
            return "Surprised"
        if mar > 0.28:
            return "Happy"
        if mar < 0.11:
            return "Focused"
        return "Neutral"

    @staticmethod
    def _ear_ratio_single(p):
        p = np.array(p)
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def calculate_ear_bilateral(self, lm):
        if "left_eye" not in lm or "right_eye" not in lm:
            return {"mean": 1.0, "left": 1.0, "right": 1.0}
        el = float(self._ear_ratio_single(lm["left_eye"]))
        er = float(self._ear_ratio_single(lm["right_eye"]))
        return {
            "mean": round((el + er) / 2.0, 3),
            "left": round(el, 3),
            "right": round(er, 3),
        }

    def calculate_ear(self, lm):
        return self.calculate_ear_bilateral(lm)["mean"]

    def _gaze_symmetry_quality(self, ear_l, ear_r, gx_l, gx_r, gy_l, gy_r):
        """Bilateral agreement: occlusion / profile asymmetry lowers score."""
        de = abs(ear_l - ear_r)
        dgx = abs(gx_l - gx_r)
        dgy = abs(gy_l - gy_r)
        score = 100.0 - min(
            100.0, 120.0 * de + 35.0 * dgx + 25.0 * dgy
        )
        return round(max(0.0, min(100.0, score)), 1)

    def _fusion_presence_score(
        self, attention, emo_conf, gaze_stability, fatigue, gaze_sym
    ):
        """Single fusion index (eye + emotion + head stability) per integration spec."""
        f = (
            0.30 * attention
            + 0.22 * emo_conf
            + 0.20 * gaze_stability
            + 0.18 * max(0.0, 100.0 - fatigue)
            + 0.10 * gaze_sym
        )
        return round(max(0.0, min(100.0, f)), 1)

    def _alert_state(self, emotion, fatigue, attention, fusion):
        if fusion >= 55 and fatigue < 50:
            return "NOMINAL"
        if emotion == "Drowsy" or fatigue >= 72:
            return "HIGH_RISK"
        if fatigue >= 48 or attention < 38 or fusion < 42:
            return "ELEVATED"
        return "NOMINAL"

    def calculate_mar(self, lm):
        if "top_lip" not in lm:
            return 0.0
        v = np.linalg.norm(
            np.array(lm["top_lip"][9]) - np.array(lm["bottom_lip"][9])
        )
        h = np.linalg.norm(np.array(lm["top_lip"][0]) - np.array(lm["top_lip"][6]))
        return round(v / (h + 1e-6), 3)

    @staticmethod
    def _eye_gaze_components(eye):
        eye = np.array(eye)
        center = np.mean(eye, axis=0)
        midpoint = (eye[0] + eye[3]) / 2.0
        width = np.linalg.norm(eye[0] - eye[3]) + 1e-6
        gx = float((center[0] - midpoint[0]) / (width * 0.42))
        gy = float((center[1] - midpoint[1]) / (width * 0.22))
        return gx, gy

    def calculate_pupil_gaze(self, lm):
        detail = self.calculate_pupil_gaze_detailed(lm)
        return {"gaze_x": detail["gaze_x"], "gaze_y": detail["gaze_y"]}

    def calculate_pupil_gaze_detailed(self, lm):
        if "left_eye" not in lm:
            z = {"gx": 0.0, "gy": 0.0}
            return {
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "left": dict(z),
                "right": dict(z),
                "vergence": 0.0,
            }
        gx_l, gy_l = self._eye_gaze_components(lm["left_eye"])
        gx_r, gy_r = self._eye_gaze_components(lm["right_eye"])
        gx = float(np.clip((gx_l + gx_r) / 2.0, -1.2, 1.2))
        gy = float(np.clip((gy_l + gy_r) / 2.0, -1.2, 1.2))
        vergence = round(float(abs(gx_l - gx_r)), 3)
        return {
            "gaze_x": round(gx, 2),
            "gaze_y": round(gy, 2),
            "left": {"gx": round(gx_l, 2), "gy": round(gy_l, 2)},
            "right": {"gx": round(gx_r, 2), "gy": round(gy_r, 2)},
            "vergence": vergence,
        }

    def get_elite_psych(self, ear, mar, gaze, track):
        ear_list = list(track["ear_hist"])
        perclos = (
            (sum(1 for e in ear_list if e < 0.21) / len(ear_list)) * 100.0
            if ear_list
            else 0.0
        )
        saccadic = 0.0
        gx_list = list(track["gx_hist"])
        t_list = list(track["t_hist"])
        if len(gx_list) >= 2 and len(t_list) >= 2:
            dx = gx_list[-1] - gx_list[-2]
            dt = t_list[-1] - t_list[-2]
            saccadic = abs(dx) / (dt + 1e-6)
        return {
            "valence": round(float(mar * 2.0 - 0.42), 2),
            "arousal": round(float(1.05 - ear * 2.1), 2),
            "cognitive_load": round(
                min(100.0, 18.0 + saccadic * 4.5 + (30.0 if ear < 0.24 else 0.0)), 1
            ),
            "stress_score": round(
                min(100.0, abs(mar - 0.11) * 190.0 + (22.0 if saccadic > 1.8 else 0.0)), 1
            ),
            "perclos": round(float(perclos), 1),
            "saccadic_velocity": round(float(saccadic), 2),
        }

    def _maybe_log_history(self, track_id, name, conf, face_id, emotion, ear, gaze, engagement, metrics):
        now = time.time()
        last = self._last_history_log.get(track_id, 0.0)
        if now - last < config.HISTORY_LOG_INTERVAL_SEC:
            return
        self._last_history_log[track_id] = now
        try:
            db.add_history_entry(
                name,
                conf,
                face_id=face_id,
                emotion=emotion,
                ear=ear,
                gaze=gaze,
                engagement=engagement,
                metrics_json=metrics,
            )
        except Exception as e:
            print(f"History log skipped: {e}")

    def recognize(self, frame_np, scale=0.5):
        try:
            inv_scale = 1.0 / scale
            proc = self._apply_clahe_lab(frame_np)
            rgb = np.ascontiguousarray(proc[:, :, ::-1])
            locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
            lands = face_recognition.face_landmarks(rgb, locs)

            now = time.time()
            fh = int(round(rgb.shape[0] * inv_scale))
            fw = int(round(rgb.shape[1] * inv_scale))
            bboxes_full = [
                tuple(int(v * inv_scale) for v in bbox) for bbox in locs
            ]
            track_ids = self._assign_tracks(bboxes_full, now)

            results = []
            for i, (bbox, enc) in enumerate(zip(locs, encs)):
                tid = track_ids[i] if i < len(track_ids) else self._next_track_id
                tr = self._tracks.setdefault(tid, self._new_track_state())

                raw_name = "Unknown"
                raw_id = None
                raw_dist = 1.0
                if self.known_face_encodings:
                    dists = face_recognition.face_distance(self.known_face_encodings, enc)
                    best = int(np.argmin(dists))
                    raw_dist = float(dists[best])
                    if raw_dist <= self.tolerance:
                        raw_name = self.known_face_names[best]
                        raw_id = self.known_face_ids[best]

                tr["votes"].append((raw_name, raw_id, raw_dist))
                name, face_id, conf = self._vote_identity(tr)

                current_lands = lands[i] if i < len(lands) else {}
                ear_bil = self.calculate_ear_bilateral(current_lands)
                ear = ear_bil["mean"]
                mar = self.calculate_mar(current_lands)
                gaze_det = self.calculate_pupil_gaze_detailed(current_lands)
                gaze = {"gaze_x": gaze_det["gaze_x"], "gaze_y": gaze_det["gaze_y"]}
                head_pose = self._head_pose_deg(current_lands, (fh, fw))
                gaze_sym = self._gaze_symmetry_quality(
                    ear_bil["left"],
                    ear_bil["right"],
                    gaze_det["left"]["gx"],
                    gaze_det["right"]["gx"],
                    gaze_det["left"]["gy"],
                    gaze_det["right"]["gy"],
                )

                tr["ear_hist"].append(ear)
                tr["gx_hist"].append(gaze["gaze_x"])
                tr["gy_hist"].append(gaze["gaze_y"])
                tr["t_hist"].append(now)
                tr["mar_hist"].append(mar)
                _, blink_bpm = self._update_blink(tr, ear, now)

                raw_emo = self.get_raw_emotion(ear, mar)
                tr["emo_hist"].append(raw_emo)
                stable_emo = max(set(tr["emo_hist"]), key=list(tr["emo_hist"]).count)

                psych = self.get_elite_psych(ear, mar, gaze, tr)
                gaze_stability = self._gaze_stability(tr)
                attention_score = self._screen_attention(gaze, head_pose)
                fatigue = self._fatigue_index(
                    psych["perclos"], ear, float(blink_bpm), head_pose.get("pitch_deg", 0.0)
                )
                emo_conf = self._emotion_confidence(tr["emo_hist"])
                micro_act = self._micro_expression_activity(tr)
                engagement = max(
                    0.0,
                    min(
                        100.0,
                        0.5 * attention_score + 0.25 * gaze_stability + 0.25 * (100.0 - fatigue),
                    ),
                )
                fusion_presence = self._fusion_presence_score(
                    attention_score, emo_conf, gaze_stability, fatigue, gaze_sym
                )
                alert_state = self._alert_state(
                    stable_emo, fatigue, attention_score, fusion_presence
                )

                bbox_out = [int(v * inv_scale) for v in bbox]
                metrics = {
                    "head_pose": head_pose,
                    "gaze_stability": gaze_stability,
                    "blink_bpm": blink_bpm,
                    "attention_score": attention_score,
                    "fatigue_index": fatigue,
                    "emotion_confidence": emo_conf,
                    "micro_expression_activity": micro_act,
                    "track_id": tid,
                    "fusion_presence_score": fusion_presence,
                    "gaze_symmetry_quality": gaze_sym,
                    "alert_state": alert_state,
                    "gaze_detail": gaze_det,
                    "ear_left": ear_bil["left"],
                    "ear_right": ear_bil["right"],
                }

                self._maybe_log_history(
                    tid,
                    name,
                    conf,
                    face_id,
                    stable_emo,
                    ear,
                    gaze,
                    engagement,
                    metrics,
                )

                results.append(
                    {
                        "track_id": tid,
                        "name": name,
                        "face_id": face_id,
                        "confidence": conf,
                        "bbox": bbox_out,
                        "emotion": stable_emo,
                        "ear": ear,
                        "ear_left": ear_bil["left"],
                        "ear_right": ear_bil["right"],
                        "mar": mar,
                        "gaze": gaze,
                        "gaze_detail": gaze_det,
                        "gaze_symmetry_quality": gaze_sym,
                        "fusion_presence_score": fusion_presence,
                        "alert_state": alert_state,
                        "psych": psych,
                        "engagement": round(engagement, 1),
                        "head_pose": head_pose,
                        "gaze_stability": gaze_stability,
                        "blink_bpm": blink_bpm,
                        "attention_score": attention_score,
                        "fatigue_index": fatigue,
                        "emotion_confidence": emo_conf,
                        "micro_expression_activity": micro_act,
                        "landmarks": {
                            k: [(int(x * inv_scale), int(y * inv_scale)) for (x, y) in pts]
                            for k, pts in current_lands.items()
                        },
                    }
                )
            return {"detections": results, "frame_size": [fh, fw]}
        except Exception as e:
            print(f"Recognize error: {e}")
            return {"detections": [], "frame_size": [0, 0]}
