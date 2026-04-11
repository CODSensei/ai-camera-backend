"""
bg_helper.py — Background removal for Indian passport photos
Spec requirement: plain white background RGB(255,255,255), no shadows.
"""

import cv2
import numpy as np
from rembg import remove, new_session

# isnet-general-use gives the best high-res portrait edge quality
session = new_session("isnet-general-use")


def remove_background(img: np.ndarray) -> np.ndarray:
    """
    Studio-quality background removal.
    Returns a BGR image composited onto pure white (255,255,255).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Alpha matting ---
    # foreground_threshold: lower (240) than before so fine hair is preserved
    # background_threshold: 20 keeps the boundary clean
    mask_rgba = remove(
        img_rgb,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=13,
        only_mask=False
    )
    mask_rgba = np.array(mask_rgba)
    alpha = mask_rgba[:, :, 3].astype(np.float32)

    # --- Feather alpha channel (3px Gaussian) ---
    # Prevents pixelated hard edges on hair
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # --- Composite onto pure white ---
    fg  = img_rgb.astype(np.float32)
    bg  = np.full_like(fg, 255.0)
    a   = alpha[:, :, np.newaxis] / 255.0
    out = (fg * a + bg * (1.0 - a)).astype(np.uint8)

    # --- Guarantee background pixels are exactly (255,255,255) ---
    # JPEG compression can drift near-white background pixels slightly.
    # We clamp pixels where alpha < 10 to hard white.
    hard_bg_mask = (alpha < 10)
    out[hard_bg_mask] = [255, 255, 255]

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def correct_colour_cast(img: np.ndarray) -> np.ndarray:
    """
    Correct mild colour cast (e.g. fluorescent green, warm yellow)
    by white-balancing the image using the brightest background pixels
    as the white reference.
    Spec: "Appropriate filters can eliminate improper colour balance."
    """
    # Find pixels that should be white (near the corners of the image)
    h, w = img.shape[:2]
    corner_size = max(20, min(h, w) // 10)

    corners = np.concatenate([
        img[:corner_size,  :corner_size ].reshape(-1, 3),
        img[:corner_size,  -corner_size:].reshape(-1, 3),
        img[-corner_size:, :corner_size ].reshape(-1, 3),
        img[-corner_size:, -corner_size:].reshape(-1, 3),
    ], axis=0).astype(np.float32)

    # Take the brightest 20% as the white reference
    brightness  = corners.mean(axis=1)
    thresh      = np.percentile(brightness, 80)
    white_ref   = corners[brightness >= thresh].mean(axis=0)  # BGR

    if white_ref.min() < 10:
        return img  # degenerate — skip

    # Scale each channel so that the white reference maps to 255
    scale = 255.0 / white_ref          # shape (3,)
    scale = np.clip(scale, 0.5, 2.0)   # prevent wild corrections

    corrected = img.astype(np.float32) * scale[np.newaxis, np.newaxis, :]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected