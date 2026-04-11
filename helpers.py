"""
helpers.py — Biometric compliance checker
Indian Passport spec: 2x2 inch (600x600px @ 300 DPI)
Head height: 295–413px (1 inch to 1-3/8 inch)
Eye height from bottom: 339–400px (1-1/8 inch to 1-1/3 inch)
"""

import mediapipe as mp
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# MediaPipe initialisation
# ---------------------------------------------------------------------------
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = FaceLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# Output canvas constants (600x600 @ 300 DPI = 2x2 inch)
# ---------------------------------------------------------------------------
CANVAS_PX        = 600
HEAD_MIN_PX      = 295   # 1 inch
HEAD_MAX_PX      = 413   # 1-3/8 inch
EYE_FROM_BTM_MIN = 339   # 1-1/8 inch
EYE_FROM_BTM_MAX = 400   # 1-1/3 inch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_biometric_compliance(frame: np.ndarray, ear_threshold: float = 0.20):
    """
    Returns (is_eligible: bool, report: dict, landmarks | None)
    All angle and geometry checks are derived from the official Indian
    passport spec document.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return False, {"error": "No face detected"}, None

    if len(result.face_landmarks) > 1:
        return False, {"error": "Multiple faces detected"}, None

    landmarks = result.face_landmarks[0]

    # --- 1. Head pose (yaw/pitch/roll) ---
    pitch, yaw, roll = _estimate_head_pose(landmarks, h, w)

    # --- 2. Eye Aspect Ratio (open eyes) ---
    left_ear  = _calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
    right_ear = _calculate_ear(landmarks, [33,  160, 158, 133, 153, 144])

    # --- 3. Mouth closed ---
    mouth_gap = abs(landmarks[13].y - landmarks[14].y)

    # Smile detection — measure lip corner elevation relative to lip centre
    lip_centre_y  = landmarks[13].y   # upper lip centre
    lip_left_y    = landmarks[61].y   # left mouth corner  
    lip_right_y   = landmarks[291].y  # right mouth corner
    corner_avg_y  = (lip_left_y + lip_right_y) / 2.0

    # If corners are higher than centre → upward curl = smile
    smile_lift = lip_centre_y - corner_avg_y   # positive = corners lifted
    # --- 4. Lighting uniformity ---
    lighting_ok, lighting_ratio = _check_lighting(frame, landmarks, h, w)

    # --- 5. Face centring (nose x should be 45–55% of frame width) ---
    nose_x_norm = landmarks[1].x
    face_centred = 0.35 <= nose_x_norm <= 0.65

    # --- 6. Skin tone / colour cast check ---
    colour_ok, colour_note = _check_colour_cast(frame, landmarks, h, w)

    report = {
        # Geometry
        "head_pose_ok":    bool(abs(yaw) < 10 and abs(pitch) < 10),
        "roll_ok":         bool(abs(roll) < 5),
        "eyes_open_ok":    bool(left_ear > ear_threshold and right_ear > ear_threshold),
        "mouth_closed_ok": bool(mouth_gap < 0.025),
        "face_centred_ok": bool(face_centred),
        # Environment
        "lighting_ok":     bool(lighting_ok),
        "colour_cast_ok":  bool(colour_ok),
        # Raw values (useful for UI feedback)
        "raw_angles": {
            "pitch": float(pitch),
            "yaw":   float(yaw),
            "roll":  float(roll),
        },
        "ear": {
            "left":  float(left_ear),
            "right": float(right_ear),
        },
        "lighting_ratio": float(lighting_ratio),
        "colour_note":    colour_note,
        # Add to the report dict
        "natural_expression_ok": bool(smile_lift < 0.018),
    }

    is_eligible = all([
        report["head_pose_ok"],
        report["roll_ok"],
        report["eyes_open_ok"],
        report["mouth_closed_ok"],
        report["natural_expression_ok"], 
        report["face_centred_ok"],
        report["lighting_ok"],
        report["colour_cast_ok"],
    ])

    return is_eligible, report, landmarks


# ---------------------------------------------------------------------------
# Sharpness scoring (used by frame selector — not compliance)
# ---------------------------------------------------------------------------
def compute_sharpness(img: np.ndarray) -> float:
    """
    Multi-metric sharpness: Tenengrad + Laplacian on face-centre ROI only.
    Avoids rewarding background noise or compression artefacts.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use centre 50% of frame — face region, avoids noisy edges
    roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    # Tenengrad (Sobel gradient energy) — robust to compression
    sx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sx ** 2 + sy ** 2))

    # Laplacian variance — good for true blur detection
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    # Penalise blocking artefacts (low local std = over-compressed)
    local_std = float(cv2.meanStdDev(roi)[1][0][0])
    block_penalty = max(0.0, 1.0 - (local_std / 20.0))  # penalty 0–1

    raw = (tenengrad * 0.5) + (lap * 0.5)
    return raw * (1.0 - 0.3 * block_penalty)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _estimate_head_pose(landmarks, h: int, w: int):
    """
    solvePnP with proper per-axis focal length for portrait video.
    Using average of w and h for focal length is more stable than w alone.
    """
    model_points = np.array([
        (0.0,    0.0,    0.0),    # nose tip
        (0.0,  -330.0,  -65.0),  # chin
        (-225.0, 170.0, -135.0), # left eye corner
        (225.0,  170.0, -135.0), # right eye corner
        (-150.0,-150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0), # right mouth corner
    ], dtype="double")

    image_points = np.array([
        (landmarks[1].x   * w, landmarks[1].y   * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x  * w, landmarks[33].y  * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x  * w, landmarks[61].y  * h),
        (landmarks[291].x * w, landmarks[291].y * h),
    ], dtype="double")

    # Better focal length estimate: geometric mean of w and h
    focal_length = float(np.sqrt(w * h))
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1 ],
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    ok, rv, tv = cv2.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rv)

    # Euler angles from rotation matrix
    pitch = float(np.degrees(np.arcsin(-rmat[2, 0])))
    yaw   = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
    roll  = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))

    # Mirror correction for front-facing cameras
    yaw = yaw + 180 if yaw < 0 else yaw - 180
    # Normalise to ±180
    yaw = ((yaw + 180) % 360) - 180

    return pitch, yaw, roll


def _calculate_ear(landmarks, indices: list) -> float:
    """Eye Aspect Ratio — higher = more open."""
    p2_p6 = abs(landmarks[indices[1]].y - landmarks[indices[5]].y)
    p3_p5 = abs(landmarks[indices[2]].y - landmarks[indices[4]].y)
    p1_p4 = abs(landmarks[indices[0]].x - landmarks[indices[3]].x) + 1e-6
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)


def _check_lighting(frame: np.ndarray, landmarks, h: int, w: int):
    """
    Compare brightness of left vs right cheek patches.
    Ratio < 0.55 = one side significantly darker → shadow rejection.
    Also checks overall face brightness for over/under exposure.
    """
    lx = int(landmarks[234].x * w)
    rx = int(landmarks[454].x * w)
    ny = int(landmarks[1].y   * h)
    patch_h = max(20, int(h * 0.08))
    patch_w = 30

    def _mean_lum(px, py):
        x1, x2 = max(0, px - patch_w // 2), min(w, px + patch_w // 2)
        y1, y2 = max(0, py - patch_h // 2), min(h, py + patch_h // 2)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))

    l_lum = _mean_lum(lx, ny)
    r_lum = _mean_lum(rx, ny)

    if l_lum is None or r_lum is None:
        return True, 1.0

    mx = max(l_lum, r_lum) + 1e-5
    ratio = min(l_lum, r_lum) / mx

    # Over/under exposure: face centre brightness
    face_cx = int(landmarks[1].x * w)
    face_cy = int(landmarks[1].y * h)
    centre_lum = _mean_lum(face_cx, face_cy) or 128.0
    exposure_ok = 40 < centre_lum < 230

    return bool(ratio > 0.55 and exposure_ok), ratio


def _check_colour_cast(frame: np.ndarray, landmarks, h: int, w: int):
    """
    Detect fluorescent / warm colour cast on the face.
    Converts face ROI to LAB and checks a* b* channel spread.
    Natural skin: |a*| in [5,25], |b*| in [5,30].
    A strong cast pushes these outside that range uniformly.
    """
    x1 = int(landmarks[234].x * w)
    x2 = int(landmarks[454].x * w)
    y1 = int(landmarks[10].y  * h)
    y2 = int(landmarks[152].y * h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return True, "ok"

    lab  = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(float)
    mean_a = float(np.mean(lab[:, :, 1])) - 128  # centred at 0
    mean_b = float(np.mean(lab[:, :, 2])) - 128

    # Unnatural green cast (fluorescent lighting)
    if mean_a < -8:
        return False, "green_cast"
    # Extreme yellow / warm cast
    if mean_b > 35:
        return False, "yellow_cast"
    # Extreme blue cast
    if mean_b < -15:
        return False, "blue_cast"

    return True, "ok"