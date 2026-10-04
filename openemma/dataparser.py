from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np


_PAIR_RE = re.compile(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]")


def parse_speed_curvature(response: str) -> np.ndarray:
    """Read ten future pairs, excluding prose and any echoed historical pairs."""
    sections = re.split(r"\bfuture speeds and curvatures\b", response, flags=re.IGNORECASE)
    if len(sections) == 1:
        # Without a prediction heading, only the requested raw numeric list is unambiguous.
        if _PAIR_RE.sub("", response).strip(" ,\n\r\t."):
            raise ValueError("Expected a raw numeric list or a future speeds and curvatures section")
    else:
        sections = sections[1:]
    groups = [_PAIR_RE.findall(section) for section in sections]
    groups = [pairs for pairs in groups if pairs]
    if len(groups) != 1 or len(groups[0]) != 10:
        raise ValueError(f"Expected exactly one prediction of 10 pairs, got {[len(pairs) for pairs in groups]}")
    values = np.asarray(groups[0], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Predicted speeds and curvatures must be finite")
    values[:, 1] /= 100.0
    return values


@dataclass(frozen=True)
class OpenEMMAHistoryFrame:
    image_base64: str
    ego_xy: np.ndarray
    heading_theta: float
    speed: float


def encode_front_image(obs_img: Dict[str, Any], camera: str = "FRONT", jpeg_quality: int = 90) -> str:
    rgb = obs_img[camera]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode {camera} image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def history_frame_from_observation(
    obs_img: Dict[str, Any],
    obs_info: Dict[str, Any],
    camera: str = "FRONT",
) -> OpenEMMAHistoryFrame:
    ego_pos = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)
    return OpenEMMAHistoryFrame(
        image_base64=encode_front_image(obs_img, camera=camera),
        ego_xy=ego_pos[:2].astype(np.float32),
        heading_theta=float(obs_info["heading_theta"]),
        speed=float(obs_info["ego_velo"]),
    )


def integrate_speed_curvature(speed_curvature: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(speed_curvature, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"speed_curvature must have shape Nx2, got {values.shape}")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and > 0, got {dt}")

    x_forward = 0.0
    y_left = 0.0
    heading = 0.0
    traj = np.zeros((values.shape[0], 2), dtype=np.float32)
    for idx, (speed, curvature) in enumerate(values):
        speed = float(speed)
        curvature = float(curvature)
        heading += curvature * speed * dt
        x_forward += speed * np.cos(heading) * dt
        y_left += speed * np.sin(heading) * dt
        traj[idx] = [x_forward, y_left]
    return traj
