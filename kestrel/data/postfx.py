"""Camera effects applied after rendering: exposure, motion blur, sensor noise, JPEG."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kestrel.sim.camera import Intrinsics

EXPOSURE_TIME_S = 1 / 60
# Translation blur is measured for a point this far ahead at the image center.
BLUR_REFERENCE_DEPTH_M = 3.0
SRGB_GAMMA = 2.2


@dataclass(frozen=True)
class CameraEffects:
    exposure_ev: float
    blur_px: tuple[float, float]
    """Motion blur streak in pixels, (u, v)."""
    noise_sigma: float
    """Gaussian noise standard deviation on the 0 to 255 scale."""
    jpeg_quality: int


def motion_blur_vector(
    velocity_cam: np.ndarray,
    angular_velocity_cam: np.ndarray,
    intrinsics: Intrinsics,
    max_px: float,
) -> tuple[float, float]:
    """Image streak at the image center during one exposure, from camera-frame motion.

    Camera frame is OpenCV: x right, y down, z forward. A world point at depth D on the optical
    axis moves in the camera frame at -v - w x (0, 0, D), so its pixel velocity is
    (fx (-vx / D - wy), fy (-vy / D + wx)).
    """
    vx, vy, _ = velocity_cam
    wx, wy, _ = angular_velocity_cam
    du = intrinsics.fx * (-vx / BLUR_REFERENCE_DEPTH_M - wy) * EXPOSURE_TIME_S
    dv = intrinsics.fy * (-vy / BLUR_REFERENCE_DEPTH_M + wx) * EXPOSURE_TIME_S
    length = float(np.hypot(du, dv))
    if length > max_px:
        du, dv = du * max_px / length, dv * max_px / length
    return float(du), float(dv)


def _blur_kernel(du: float, dv: float) -> np.ndarray | None:
    length = float(np.hypot(du, dv))
    if length < 1.0:
        return None
    # Two pixels of margin keep the rounded streak ends inside the kernel.
    size = (int(np.ceil(length)) + 2) | 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for t in np.linspace(-0.5, 0.5, 4 * size):
        kernel[round(center + t * dv), round(center + t * du)] += 1.0
    return kernel / kernel.sum()


def apply_effects(rgb: np.ndarray, effects: CameraEffects, rng: np.random.Generator) -> np.ndarray:
    """Exposure, blur and noise. JPEG is separate so datasets can save at the chosen quality."""
    image = rgb.astype(np.float32) * 2 ** (effects.exposure_ev / SRGB_GAMMA)
    kernel = _blur_kernel(*effects.blur_px)
    if kernel is not None:
        image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    if effects.noise_sigma > 0:
        image += rng.normal(0.0, effects.noise_sigma, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError(f"JPEG encoding failed at quality {quality}")
    return encoded.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG decoding failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def degrade(rgb: np.ndarray, effects: CameraEffects, rng: np.random.Generator) -> np.ndarray:
    """All effects including a JPEG round trip, as the pilot's camera would deliver frames."""
    return decode_jpeg(encode_jpeg(apply_effects(rgb, effects, rng), effects.jpeg_quality))
