"""
main.py — Pro Passport Photo API
Indian Passport spec: 2x2 inch (51x51mm) = 600x600 px @ 300 DPI
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import aiofiles, os, uuid, json
import subprocess
import cv2
import numpy as np

from helpers import get_biometric_compliance, compute_sharpness
from crop_helper import perform_biometric_crop
from bg_helper import remove_background, correct_colour_cast

app = FastAPI(title="Pro Passport API — Indian Spec")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DIRS = ["uploads", "frames", "results", "metadata"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------
CANVAS_PX   = 600          # 2 inch × 2 inch @ 300 DPI
JPEG_QUALITY = 95          # High quality but keeps file < 500 KB
MAX_FILE_KB  = 500


# ===========================================================================
# 1. Upload video
# ===========================================================================
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept a video file. Extract frames at an adaptive rate designed to
    maximise sharpness variety while staying under 300 frames.
    """
    sid  = str(uuid.uuid4())
    path = f"uploads/{sid}.mp4"
    fdir = f"frames/{sid}"
    os.makedirs(fdir, exist_ok=True)

    async with aiofiles.open(path, "wb") as f:
        await f.write(await file.read())

    # --- Probe video duration and native fps ---
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
        "-of", "json", path
    ]
    try:
        probe_out = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL)
        probe_data = json.loads(probe_out)
        stream = probe_data.get("streams", [{}])[0]
        duration = float(stream.get("duration", 10))
        fps_str  = stream.get("r_frame_rate", "30/1").split("/")
        native_fps = int(fps_str[0]) / max(1, int(fps_str[1]))
    except Exception:
        duration, native_fps = 10.0, 30.0

    # Extract at 4 fps (enough for compliance variety, avoids duplicates)
    # Cap at 200 frames total
    extract_fps = min(4.0, 200.0 / max(duration, 1.0))

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", path,
        "-vf", f"fps={extract_fps:.2f}",
        "-vframes", "200",
        "-q:v", "1",           # highest JPEG quality from FFmpeg
        f"{fdir}/frame_%04d.jpg",
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted([f for f in os.listdir(fdir) if f.endswith(".jpg")])
    _save_meta(sid, {
        "session_id": sid,
        "frame_dir":  fdir,
        "frame_files": frames,
        "video_duration_s": duration,
        "extract_fps": extract_fps,
    })
    return {"session_id": sid, "total_frames": len(frames), "extract_fps": round(extract_fps, 2)}


def _compute_ear_baseline(frame_files: list, frame_dir: str) -> float:
    """
    Sample up to 30 frames to find this person's natural open-eye EAR.
    Uses 65% of their median as the threshold so narrow eyes never false-flag.
    """
    from helpers import _calculate_ear   # import the internal helper
    ears = []
    sample = frame_files[::max(1, len(frame_files) // 30)][:30]

    for fname in sample:
        img = cv2.imread(f"{frame_dir}/{fname}")
        if img is None:
            continue
        _, _, lm = get_biometric_compliance(img)   # uses default threshold
        if lm is None:
            continue
        l = _calculate_ear(lm, [362, 385, 387, 263, 373, 380])
        r = _calculate_ear(lm, [33,  160, 158, 133, 153, 144])
        if l > 0.08 and r > 0.08:   # skip blink frames
            ears.append((l + r) / 2.0)

    if len(ears) < 3:
        return 0.18   # not enough data — use safe fallback

    baseline  = float(np.median(ears))
    threshold = baseline * 0.65          # 65% of their natural open eye
    return max(0.10, threshold)          # hard floor — physically can't be open below this


# ===========================================================================
# 2. Select best frames
# ===========================================================================
@app.post("/select-best-frames/{session_id}")
async def select_best(session_id: str, top_n: int = 5):
    """
    Score each frame on sharpness + compliance, deduplicate temporally,
    and return the top N candidates.
    """
    meta   = _load_meta(session_id)
    ear_threshold = _compute_ear_baseline(meta["frame_files"], meta["frame_dir"])
    meta["ear_threshold"] = ear_threshold
    _save_meta(session_id, meta)
    scored = []
    for fname in meta["frame_files"]:
        img = cv2.imread(f"{meta['frame_dir']}/{fname}")
        if img is None:
            continue
        
        # Multi-metric sharpness on face-centre ROI
        sharpness = compute_sharpness(img)

        # Biometric compliance (more nuanced than binary)
        eligible, report, _ = get_biometric_compliance(img)

        # Face score: use a graded scale instead of binary 0.1/1.0
        face_score = _grade_compliance(report)

        scored.append({
            "filename":   fname,
            "sharpness":  sharpness,
            "face_score": face_score,
            "compliance": report,
        })

    if not scored:
        raise HTTPException(404, "No readable frames found")

    # --- Normalise sharpness to 0–1 ---
    s_vals = [x["sharpness"] for x in scored]
    s_min, s_max = min(s_vals), max(s_vals)
    s_range = max(s_max - s_min, 1e-5)

    for x in scored:
        s_norm    = (x["sharpness"] - s_min) / s_range
        # 40% sharpness, 60% compliance — spec quality is paramount
        x["composite"] = (s_norm * 0.40) + (x["face_score"] * 0.60)

    # Sort descending by composite
    scored.sort(key=lambda x: x["composite"], reverse=True)

    # --- Temporal deduplication ---
    # Prevent consecutive near-identical frames from consuming all top_n slots
    MIN_FRAME_GAP = 8   # at 4fps this is ~2 seconds
    best = _temporal_dedup(scored, min_gap=MIN_FRAME_GAP, n=top_n)

    meta["best_frames"] = [b["filename"] for b in best]
    _save_meta(session_id, meta)

    return {"best_frames": best}


# ===========================================================================
# 3. Generate passport photo
# ===========================================================================
@app.post("/generate-passport-photo/{session_id}")
async def generate_photo(session_id: str, frame_index: int = 0):
    """
    Full pipeline for a single selected frame:
    1. Biometric compliance check
    2. Colour-cast correction
    3. Background removal → pure white
    4. Roll correction + biometric crop (600×600 @ 300 DPI)
    5. Final background integrity pass
    6. Save with spec-compliant JPEG settings
    """
    meta = _load_meta(session_id)
    if not meta.get("best_frames"):
        raise HTTPException(400, "Run /select-best-frames first")

    frame_path = f"{meta['frame_dir']}/{meta['best_frames'][frame_index]}"
    img = cv2.imread(frame_path)
    if img is None:
        raise HTTPException(404, "Frame file not found")

    # --- 1. Biometric compliance & landmarks ---
    ear_threshold = meta.get("ear_threshold", 0.18)   # use session baseline
    is_eligible, report, landmarks = get_biometric_compliance(img, ear_threshold=ear_threshold)
    if landmarks is None:
        raise HTTPException(422, "No face detected in selected frame")

    # --- 2. Colour-cast correction (before background removal) ---
    img = correct_colour_cast(img)

    # --- 3. Background removal ---
    img_no_bg = remove_background(img)

    # --- 4. Roll correction + biometric crop (returns 600×600) ---
    final_photo, crop_meta = perform_biometric_crop(
        img_no_bg, landmarks, canvas_px=CANVAS_PX
    )

    # --- 5. Final background integrity: clamp near-white pixels to pure white ---
    final_photo = _enforce_white_background(final_photo)

    # --- 6. Save ---
    pid      = str(uuid.uuid4())
    out_path = f"results/{pid}.jpg"

    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,   JPEG_QUALITY,
        cv2.IMWRITE_JPEG_OPTIMIZE,  1,
        # No PROGRESSIVE flag — some e-form portals reject progressive JPEGs
    ]
    cv2.imwrite(out_path, final_photo, encode_params)

    # Warn if file exceeds 500 KB
    file_kb = os.path.getsize(out_path) / 1024
    if file_kb > MAX_FILE_KB:
        # Re-encode at lower quality to meet portal limit
        q = int(JPEG_QUALITY * MAX_FILE_KB / file_kb) - 2
        q = max(80, min(q, 95))
        cv2.imwrite(out_path, final_photo, [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        file_kb = os.path.getsize(out_path) / 1024

    return {
        "photo_id":          pid,
        "is_compliant":      is_eligible,
        "compliance_report": report,
        "crop_metadata":     crop_meta,
        "file_size_kb":      round(file_kb, 1),
        "canvas_px":         CANVAS_PX,
        "spec":              "Indian Passport 2x2 inch 300DPI",
        "download_url":      f"/download/{pid}",
    }


# ===========================================================================
# 4. Download
# ===========================================================================
@app.get("/download/{photo_id}")
async def download(photo_id: str):
    path = f"results/{photo_id}.jpg"
    if not os.path.exists(path):
        raise HTTPException(404, "Photo not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="passport_{photo_id}.jpg"'}
    )


# ===========================================================================
# Internal helpers
# ===========================================================================
def _grade_compliance(report: dict) -> float:
    """
    Convert a compliance report into a 0–1 graded score.
    All-pass = 1.0. Each failed check deducts a weighted amount.
    This avoids the binary 0.1/1.0 cliff that swamped the composite score.
    """
    if "error" in report:
        return 0.0

    weights = {
        "head_pose_ok":    0.30,
        "roll_ok":         0.15,
        "eyes_open_ok":    0.20,
        "mouth_closed_ok": 0.10,
        "face_centred_ok": 0.10,
        "lighting_ok":     0.10,
        "colour_cast_ok":  0.05,
    }
    score = 0.0
    for key, w in weights.items():
        if report.get(key, False):
            score += w
    return score


def _temporal_dedup(scored: list, min_gap: int, n: int) -> list:
    """
    Walk through scored frames (sorted best-first) and only keep a frame
    if it is at least `min_gap` frames away from any already-kept frame.
    Returns up to `n` frames.
    """
    kept       = []
    kept_idxes = []

    for item in scored:
        try:
            idx = int(item["filename"].split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            idx = -999

        if all(abs(idx - k) >= min_gap for k in kept_idxes):
            kept.append(item)
            kept_idxes.append(idx)

        if len(kept) >= n:
            break

    # If dedup was too aggressive, fill remaining slots without the gap constraint
    if len(kept) < n:
        existing = set(x["filename"] for x in kept)
        for item in scored:
            if item["filename"] not in existing:
                kept.append(item)
                if len(kept) >= n:
                    break

    return kept


def _enforce_white_background(img: np.ndarray) -> np.ndarray:
    """
    Clamp any near-white background pixel (R,G,B all > 240) to
    pure (255,255,255). Prevents JPEG compression drift on the background.
    """
    mask = np.all(img > 240, axis=2)
    out  = img.copy()
    out[mask] = [255, 255, 255]
    return out


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_meta(sid: str, data: dict):
    with open(f"metadata/{sid}.json", "w") as f:
        json.dump(data, f, indent=2, cls=_NpEncoder)


def _load_meta(sid: str) -> dict:
    path = f"metadata/{sid}.json"
    if not os.path.exists(path):
        raise HTTPException(404, f"Session {sid} not found")
    with open(path) as f:
        return json.load(f)