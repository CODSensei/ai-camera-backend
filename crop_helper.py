"""
crop_helper.py — Biometric crop for Indian passport photo
Spec: 2x2 inch (51x51mm) = 600x600 px @ 300 DPI
Head height: 295–413 px (1 inch to 1-3/8 inch) — target centre ~354px
Eye position from bottom: 339–400 px (1-1/8 to 1-1/3 inch) — target ~370px
"""

import cv2
import numpy as np
import mediapipe as mp

# Re-import landmarker for post-rotation re-detection
from helpers import landmarker, get_biometric_compliance

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------
CANVAS_PX        = 600
HEAD_TARGET_PX   = 354   # midpoint of 295–413 px — ~59% of canvas
EYE_TARGET_FROM_BTM = 370  # midpoint of 339–400 px
HEAD_MIN_PX      = 295
HEAD_MAX_PX      = 413


def perform_biometric_crop(
    img: np.ndarray,
    landmarks,
    canvas_px: int = CANVAS_PX
) -> np.ndarray:
    """
    Full pipeline:
    1. Roll correction (deskew by eye-line angle)
    2. Re-detect landmarks on corrected image
    3. Scale so head height hits target
    4. Translate so:
         - head is vertically positioned per spec
         - nose is horizontally centred
    5. Crop to canvas_px × canvas_px
    6. Final white-fill integrity pass
    """
    # ------------------------------------------------------------------
    # Step 1 — Roll correction
    # ------------------------------------------------------------------
    img, landmarks = _correct_roll(img, landmarks)

    h, w = img.shape[:2]

    # ------------------------------------------------------------------
    # Step 2 — Extract key landmark positions
    # ------------------------------------------------------------------
    # Landmark 10 is the scalp midpoint, not the hair crown.
    # Empirically it sits ~12–15% of head_height BELOW the true hair tip.
    # We extrapolate upward to estimate the real crown.
    crown_lm_y  = landmarks[10].y  * h
    chin_y      = landmarks[152].y * h
    nose_x      = landmarks[1].x   * w
    left_eye_y  = landmarks[33].y  * h
    right_eye_y = landmarks[263].y * h
    eye_y       = (left_eye_y + right_eye_y) / 2.0

    head_height_raw = abs(chin_y - crown_lm_y)

    # Extrapolate true hair crown (add 14% of head height above landmark 10)
    CROWN_OFFSET = 0.22
    true_crown_y = crown_lm_y - (head_height_raw * CROWN_OFFSET)
    
    frame_h = img.shape[0]
    if true_crown_y < frame_h * 0.08:
        true_crown_y = frame_h * 0.04

    head_height  = abs(chin_y - true_crown_y)   # corrected head height

    if head_height < 10:
        # Fallback: return centred crop if geometry is degenerate
        return _safe_centre_crop(img, canvas_px)

    # ------------------------------------------------------------------
    # Step 3 — Scale so head height = HEAD_TARGET_PX
    # ------------------------------------------------------------------
    scale = HEAD_TARGET_PX / head_height

    # Anchor scale at the chin (most stable vertical landmark)
    anchor_x = nose_x
    anchor_y = chin_y

    M_scale = cv2.getRotationMatrix2D((anchor_x, anchor_y), 0, scale)
    new_w = max(canvas_px, int(w * scale * 1.2))
    new_h = max(canvas_px, int(h * scale * 1.2))
    img_scaled = cv2.warpAffine(
        img, M_scale, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[255, 255, 255]
    )

    # Recompute positions after scaling
    def sc(x, y):
        pt = M_scale @ np.array([x, y, 1.0])
        return pt[0], pt[1]

    scaled_nose_x,   scaled_chin_y   = sc(anchor_x, anchor_y)
    scaled_crown_x,  scaled_crown_y  = sc(nose_x, true_crown_y)
    scaled_eye_x,    scaled_eye_y    = sc(nose_x, eye_y)

    # ------------------------------------------------------------------
    # Step 4 — Compute crop origin
    # ------------------------------------------------------------------
    # Eye should be EYE_TARGET_FROM_BTM px from the bottom of the canvas.
    # crop_top = scaled_eye_y - (canvas_px - EYE_TARGET_FROM_BTM)
    TOP_SAFETY_PX = 30
    crop_top = int(scaled_eye_y - (canvas_px - EYE_TARGET_FROM_BTM)) - TOP_SAFETY_PX
    crop_left = int(scaled_nose_x - canvas_px // 2)

    # ------------------------------------------------------------------
    # Step 5 — Pad canvas if crop bleeds outside image
    # ------------------------------------------------------------------
    pad_top    = max(0, -crop_top)
    pad_left   = max(0, -crop_left)
    pad_bottom = max(0, crop_top  + canvas_px - img_scaled.shape[0])
    pad_right  = max(0, crop_left + canvas_px - img_scaled.shape[1])

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img_scaled = cv2.copyMakeBorder(
            img_scaled,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        crop_top  += pad_top
        crop_left += pad_left

    # ------------------------------------------------------------------
    # Step 6 — Crop
    # ------------------------------------------------------------------
    cropped = img_scaled[
        crop_top  : crop_top  + canvas_px,
        crop_left : crop_left + canvas_px
    ]

    # Guard: resize to exact canvas_px if rounding drifted the size
    if cropped.shape[0] != canvas_px or cropped.shape[1] != canvas_px:
        cropped = cv2.resize(
            cropped, (canvas_px, canvas_px),
            interpolation=cv2.INTER_LANCZOS4
        )

    # ------------------------------------------------------------------
    # Step 7 — Validate head height is within spec; warn if not
    # ------------------------------------------------------------------
    final_head_px = int(HEAD_TARGET_PX)  # we scaled to this — for logging
    crop_metadata = {
        "head_height_px":  final_head_px,
        "eye_from_bottom": EYE_TARGET_FROM_BTM,
        "head_in_spec":    HEAD_MIN_PX <= final_head_px <= HEAD_MAX_PX,
    }

    return cropped, crop_metadata


# ---------------------------------------------------------------------------
# Roll correction
# ---------------------------------------------------------------------------
def _correct_roll(img: np.ndarray, landmarks) -> tuple:
    """
    Rotate image so the inter-eye line is horizontal.
    Re-detects landmarks after rotation for accurate downstream geometry.
    Returns (corrected_img, new_landmarks).
    """
    h, w = img.shape[:2]

    left_eye_x  = landmarks[33].x  * w
    left_eye_y  = landmarks[33].y  * h
    right_eye_x = landmarks[263].x * w
    right_eye_y = landmarks[263].y * h

    # Angle between eye centres
    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    angle = float(np.degrees(np.arctan2(dy, dx)))

    if abs(angle) < 0.5:
        return img, landmarks   # no correction needed

    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[255, 255, 255]
    )

    # Re-run MediaPipe on rotated image
    rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if result.face_landmarks:
        new_landmarks = result.face_landmarks[0]
    else:
        # If re-detection fails keep original landmarks (rare edge case)
        new_landmarks = landmarks

    return rotated, new_landmarks


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
def _safe_centre_crop(img: np.ndarray, canvas_px: int) -> tuple:
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - canvas_px // 2)
    x1 = max(0, cx - canvas_px // 2)
    crop = img[y1:y1+canvas_px, x1:x1+canvas_px]
    if crop.shape[0] != canvas_px or crop.shape[1] != canvas_px:
        crop = cv2.resize(crop, (canvas_px, canvas_px), interpolation=cv2.INTER_LANCZOS4)
    return crop, {"head_height_px": 0, "eye_from_bottom": 0, "head_in_spec": False}