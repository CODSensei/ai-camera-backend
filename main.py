from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import aiofiles
import os
import uuid
import ffmpeg
import cv2
import numpy as np
from pathlib import Path
import json
import math

app = FastAPI(title="AI Camera Passport Photo API")

# ── Directory setup ──────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
RESULTS_DIR = "results"
METADATA_DIR = "metadata"

for d in [UPLOAD_DIR, FRAMES_DIR, RESULTS_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# ── Passport photo spec (35mm × 45mm at 300 DPI) ─────────────────────────────
PASSPORT_W_PX = 413  # 35mm @ 300dpi
PASSPORT_H_PX = 531  # 45mm @ 300dpi
PASSPORT_DPI = 300


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Upload video & extract frames
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/upload-video/")
async def upload_and_extract(file: UploadFile = File(...)):
    """
    Accepts a 1080p 30fps video (5–10 s).
    Saves it, extracts every frame via ffmpeg, returns a session_id.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    # Use a uuid-based name to avoid collisions
    session_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    video_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
    frame_dir = os.path.join(FRAMES_DIR, session_id)
    os.makedirs(frame_dir, exist_ok=True)

    # ── Save upload ──────────────────────────────────────────────────────────
    total_bytes = 0
    async with aiofiles.open(video_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, detail="Video exceeds 200 MB limit."
                )
            await out_file.write(chunk)

    # ── Extract ALL frames (30 fps) ──────────────────────────────────────────
    frame_pattern = os.path.join(frame_dir, "frame_%04d.jpg")
    try:
        (
            ffmpeg.input(video_path)
            .output(
                frame_pattern,
                vf="fps=30",  # keep every frame
                vframes=300,  # cap at 300 frames (10 s × 30 fps)
                **{"q:v": "2"},  # high JPEG quality (colon args need dict syntax)
            )
            .run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as e:
        raise HTTPException(
            status_code=500, detail=f"FFmpeg error: {e.stderr.decode()}"
        )

    frame_files = sorted(
        f
        for f in os.listdir(frame_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )

    if not frame_files:
        raise HTTPException(
            status_code=500, detail="No frames were extracted from the video."
        )

    # ── Persist session metadata ─────────────────────────────────────────────
    meta = {
        "session_id": session_id,
        "video_path": video_path,
        "frame_dir": frame_dir,
        "frame_files": frame_files,
        "total_frames": len(frame_files),
    }
    _save_meta(session_id, meta)

    return {
        "session_id": session_id,
        "total_frames": len(frame_files),
        "preview": frame_files[:5],
        "message": "Frames extracted. Call /select-best-frames/{session_id} next.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Score & select the best frames
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/select-best-frames/{session_id}")
async def select_best_frames(session_id: str, top_n: int = 5):
    """
    Scores every extracted frame on:
      • Sharpness  (Laplacian variance)
      • Brightness (mean luminance in 40–220 range)
      • Face score (face detected + frontal + size)
      • Eye-open score (eye-aspect-ratio via landmarks)

    Returns the top_n frame filenames ranked by composite score.
    """
    meta = _load_meta(session_id)
    frame_dir = meta["frame_dir"]
    frame_files = meta["frame_files"]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    scored = []
    for fname in frame_files:
        path = os.path.join(frame_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        score_data = _score_frame(frame, face_cascade, eye_cascade)
        scored.append({"filename": fname, **score_data})

    if not scored:
        raise HTTPException(status_code=404, detail="Could not read any frames.")

    # ── Normalise each sub-score to [0,1] then weight ────────────────────────
    def _norm(lst, key):
        vals = [s[key] for s in lst]
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1
        for s in lst:
            s[f"{key}_norm"] = (s[key] - lo) / span

    _norm(scored, "sharpness")
    _norm(scored, "brightness")
    _norm(scored, "face_score")

    WEIGHTS = {"sharpness": 0.30, "brightness": 0.20, "face_score": 0.50}
    for s in scored:
        s["composite"] = (
            WEIGHTS["sharpness"] * s["sharpness_norm"]
            + WEIGHTS["brightness"] * s["brightness_norm"]
            + WEIGHTS["face_score"] * s["face_score_norm"]
        )

    ranked = sorted(scored, key=lambda x: x["composite"], reverse=True)
    best = ranked[:top_n]

    meta["best_frames"] = [b["filename"] for b in best]
    meta["scored"] = ranked  # full ranking stored for debug
    _save_meta(session_id, meta)

    return {
        "session_id": session_id,
        "top_n": top_n,
        "best_frames": meta["best_frames"],
        "scores": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()}
            for b in best
        ],
        "message": "Call /generate-passport-photo/{session_id} next.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Generate passport photo from best frame
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/generate-passport-photo/{session_id}")
async def generate_passport_photo(session_id: str, frame_index: int = 0):
    """
    Takes the Nth best frame (default: the best one),
    detects & crops the face with passport margins,
    resizes to 35×45 mm at 300 DPI, applies mild enhancement,
    and saves the result.
    Returns a photo_id you can use with /download/{photo_id}.
    """
    meta = _load_meta(session_id)

    if "best_frames" not in meta or not meta["best_frames"]:
        raise HTTPException(
            status_code=400,
            detail="No best frames found. Run /select-best-frames first.",
        )
    if frame_index >= len(meta["best_frames"]):
        raise HTTPException(status_code=400, detail=f"frame_index out of range.")

    chosen_file = meta["best_frames"][frame_index]
    frame_path = os.path.join(meta["frame_dir"], chosen_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not load selected frame.")

    # ── Face detection ───────────────────────────────────────────────────────
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the selected frame. Try a different frame_index.",
        )

    # Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # ── Passport crop with margin ────────────────────────────────────────────
    # Standard: head occupies 70–80% of frame height; chin-to-crown centred
    margin_top = int(h * 0.6)  # generous forehead room
    margin_sides = int(w * 0.5)
    margin_bottom = int(h * 0.3)  # chin room

    H, W = frame.shape[:2]
    x1 = max(0, x - margin_sides)
    y1 = max(0, y - margin_top)
    x2 = min(W, x + w + margin_sides)
    y2 = min(H, y + h + margin_bottom)

    cropped = frame[y1:y2, x1:x2]

    # ── Resize to passport dimensions ────────────────────────────────────────
    passport = cv2.resize(
        cropped, (PASSPORT_W_PX, PASSPORT_H_PX), interpolation=cv2.INTER_LANCZOS4
    )

    # ── Light enhancement ────────────────────────────────────────────────────
    passport = _enhance(passport)

    # ── Save result ──────────────────────────────────────────────────────────
    photo_id = str(uuid.uuid4())
    output_path = os.path.join(RESULTS_DIR, f"{photo_id}.jpg")
    cv2.imwrite(
        output_path,
        passport,
        [cv2.IMWRITE_JPEG_QUALITY, 97, cv2.IMWRITE_JPEG_RST_INTERVAL, PASSPORT_DPI],
    )

    meta["photo_id"] = photo_id
    meta["output_path"] = output_path
    meta["source_frame"] = chosen_file
    _save_meta(session_id, meta)

    return {
        "session_id": session_id,
        "photo_id": photo_id,
        "source_frame": chosen_file,
        "dimensions": f"{PASSPORT_W_PX}×{PASSPORT_H_PX}px ({PASSPORT_DPI} DPI)",
        "spec": "35mm × 45mm passport standard",
        "download_url": f"/download/{photo_id}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — Download the passport photo
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/download/{photo_id}")
async def download_photo(photo_id: str):
    """Serves the final passport photo JPEG."""
    # Basic validation — no path traversal
    if not _is_safe_id(photo_id):
        raise HTTPException(status_code=400, detail="Invalid photo_id.")

    path = os.path.join(RESULTS_DIR, f"{photo_id}.jpg")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Photo not found.")

    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=f"passport_{photo_id}.jpg",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Full pipeline in one shot (convenience)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/process-video/")
async def process_video_full(file: UploadFile = File(...), top_n: int = 5):
    """
    Convenience endpoint: upload → extract → score → generate passport photo.
    Returns the photo_id and download URL in a single call.
    """
    # Step 1
    upload_result = await upload_and_extract(file)
    sid = upload_result["session_id"]

    # Step 2
    await select_best_frames(sid, top_n=top_n)

    # Step 3
    photo_result = await generate_passport_photo(sid, frame_index=0)

    return {
        "session_id": sid,
        "photo_id": photo_result["photo_id"],
        "download_url": photo_result["download_url"],
        "source_frame": photo_result["source_frame"],
        "dimensions": photo_result["dimensions"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — Session info / debug
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Returns metadata for a session (useful for debugging)."""
    meta = _load_meta(session_id)
    # Don't expose full scored list — just summary
    meta.pop("scored", None)
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _score_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
) -> dict:
    """Returns raw (un-normalised) quality scores for one frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness — Laplacian variance (higher = sharper)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Brightness — mean luminance; penalise over/under-exposed
    mean_lum = float(np.mean(gray))
    brightness = 1.0 - abs(mean_lum - 128) / 128  # peaks at 128, falls off both ways

    # 3. Face score
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    face_score = 0.0
    if len(faces) > 0:
        # Pick largest face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces_sorted[0]

        # Size relative to frame area
        frame_area = frame.shape[0] * frame.shape[1]
        face_area = fw * fh
        size_score = min(face_area / frame_area * 10, 1.0)  # cap at 1

        # Frontal-ness: face should be near horizontal centre
        face_cx = fx + fw / 2
        frame_cx = frame.shape[1] / 2
        center_score = 1.0 - abs(face_cx - frame_cx) / frame_cx

        # Eye detection within face ROI
        face_roi = gray[fy : fy + fh, fx : fx + fw]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
        eye_score = min(len(eyes) / 2.0, 1.0)  # 2 eyes = perfect

        face_score = 0.35 * size_score + 0.25 * center_score + 0.40 * eye_score

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "face_score": face_score,
    }


def _enhance(img: np.ndarray) -> np.ndarray:
    """Mild, passport-safe enhancement: slight sharpening + CLAHE on luminance."""
    # Convert to LAB for luminance-only CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Light unsharp mask
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
    sharp = cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)

    return sharp


def _save_meta(session_id: str, data: dict):
    path = os.path.join(METADATA_DIR, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_meta(session_id: str) -> dict:
    if not _is_safe_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    path = os.path.join(METADATA_DIR, f"{session_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Session not found.")
    with open(path) as f:
        return json.load(f)


def _is_safe_id(id_str: str) -> bool:
    """Validates UUID format to prevent path traversal."""
    import re

    return bool(re.fullmatch(r"[0-9a-f\-]{36}", id_str))
