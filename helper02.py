import cv2
import numpy as np

def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian — but only on a centre ROI, not full frame (avoids background noise)
    h, w = gray.shape
    roi = gray[h//4:3*h//4, w//4:3*w//4]
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    
    # Tenengrad (gradient energy) — more robust than pure Laplacian
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x**2 + sobel_y**2))
    
    # BRISQUE-lite: local variance deviation (penalises compression blocking)
    local_var = float(cv2.meanStdDev(roi)[1][0][0])
    
    # Normalise and combine
    return (lap * 0.4) + (tenengrad * 0.4) + (local_var * 0.2)

def deduplicate_frames(scored, min_gap=10):
    """Keep only the best frame within each temporal window."""
    kept = []
    last_idx = -min_gap
    for item in sorted(scored, key=lambda x: x["composite"], reverse=True):
        frame_num = int(item["filename"].split("_")[1].split(".")[0])
        if frame_num - last_idx >= min_gap:
            kept.append(item)
            last_idx = frame_num
    return kept


def check_lighting(img, landmarks):
    h, w = img.shape[:2]
    face_left  = int(landmarks[234].x * w)  # left cheek
    face_right = int(landmarks[454].x * w)  # right cheek
    nose_y     = int(landmarks[1].y   * h)
    
    roi_h = int(h * 0.3)
    left_patch  = img[nose_y-roi_h//2 : nose_y+roi_h//2, face_left  : face_left+40]
    right_patch = img[nose_y-roi_h//2 : nose_y+roi_h//2, face_right-40 : face_right]
    
    if left_patch.size == 0 or right_patch.size == 0:
        return True, 0.0  # can't measure, pass through
    
    left_brightness  = float(np.mean(cv2.cvtColor(left_patch,  cv2.COLOR_BGR2GRAY)))
    right_brightness = float(np.mean(cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY)))
    
    ratio = min(left_brightness, right_brightness) / max(left_brightness, right_brightness + 1e-5)
    is_ok = ratio > 0.6  # within 40% brightness difference side-to-side
    return is_ok, ratio