from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import cv2
import numpy as np


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


def estimate_curvature(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)

    traj = points[:, :2]
    curvature = np.zeros(traj.shape[0], dtype=np.float32)
    for i in range(1, traj.shape[0] - 1):
        x1, y1 = traj[i - 1]
        x2, y2 = traj[i]
        x3, y3 = traj[i + 1]
        l1 = float(np.linalg.norm([x2 - x1, y2 - y1]))
        l2 = float(np.linalg.norm([x3 - x2, y3 - y2]))
        l3 = float(np.linalg.norm([x3 - x1, y3 - y1]))
        area_signed = 0.5 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if l1 > 0.0 and l2 > 0.0 and l3 > 0.0:
            curvature[i] = float(4.0 * area_signed / (l1 * l2 * l3))
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


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
