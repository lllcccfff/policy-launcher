from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


REQUIRED_CAMERAS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")


def _latest_rgb(obs_img: Dict[str, Any], camera: str) -> np.ndarray:
    arr = np.asarray(obs_img[camera])
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{camera} image must have shape HxWx3 or TxHxWx3, got {arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))


def _scale_crop(image: Image.Image, scale: int, start_x: int, crop_x: int, start_y: int, crop_y: int) -> np.ndarray:
    width, height = image.width // scale, image.height // scale
    if scale != 1:
        image = image.resize((width, height))
    arr = np.asarray(image)
    cropped = arr[start_y : start_y + crop_y, start_x : start_x + crop_x]
    if cropped.shape[:2] != (crop_y, crop_x):
        raise ValueError(f"Latent TransFuser camera crop expected {(crop_y, crop_x)}, got {cropped.shape[:2]}")
    return cropped


def _scale_center_crop(image: Image.Image, scale: int, crop_x: int, crop_y: int) -> np.ndarray:
    width, height = image.width // scale, image.height // scale
    start_x = (width - int(crop_x)) // 2
    start_y = (height - int(crop_y)) // 2
    if start_x < 0 or start_y < 0:
        raise ValueError(f"Latent TransFuser crop {(crop_y, crop_x)} exceeds image {(height, width)}")
    return _scale_crop(image, scale, start_x, crop_x, start_y, crop_y)


def _shift_x_scale_crop(image: Image.Image, scale: int, crop: tuple[int, int], crop_shift: int = 0) -> np.ndarray:
    crop_h, crop_w = crop
    width, height = int(image.width // scale), int(image.height // scale)
    resized = image.resize((width, height))
    arr = np.asarray(resized)
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2 + int(crop_shift // scale)
    cropped = arr[start_y : start_y + crop_h, start_x : start_x + crop_w]
    if cropped.shape[:2] != (crop_h, crop_w):
        raise ValueError(f"Latent TransFuser crop expected {(crop_h, crop_w)}, got {cropped.shape[:2]}")
    return np.transpose(cropped, (2, 0, 1)).astype(np.float32)


def _prepare_rgb(obs_img: Dict[str, Any], config: Any) -> torch.Tensor:
    crops = []
    for camera in REQUIRED_CAMERAS:
        rgb = _latest_rgb(obs_img, camera)
        crop = _scale_center_crop(
            Image.fromarray(rgb),
            config.scale,
            config.img_width,
            config.img_resolution[0],
        )
        crops.append(crop)
    rgb_concat = np.concatenate(crops, axis=1)
    image = _shift_x_scale_crop(Image.fromarray(rgb_concat), config.scale, config.img_resolution)
    return torch.from_numpy(image).unsqueeze(0)


def _route_waypoints(obs_info: Dict[str, Any]) -> np.ndarray:
    route_key = "latent_transfuser_route_waypoints"
    waypoints = np.asarray(obs_info.get(route_key, obs_info["route_waypoints"]), dtype=np.float32)
    return waypoints[:, :2].astype(np.float32)


def _target_point(obs_info: Dict[str, Any], target_point_index: int) -> np.ndarray:
    waypoints = _route_waypoints(obs_info)
    ego_pos = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)
    if ego_pos.size < 2:
        raise ValueError(f"ego_pos must contain at least xy, got {ego_pos}")

    del target_point_index
    target_idx = int(obs_info["latent_transfuser_target_route_idx"])
    if target_idx < 0 or target_idx >= waypoints.shape[0]:
        raise ValueError(f"latent_transfuser_target_route_idx {target_idx} is outside route length {waypoints.shape[0]}")
    delta = waypoints[target_idx] - ego_pos[:2]
    yaw = float(obs_info["heading_theta"])
    rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    local_point = (rotation.T @ delta.astype(np.float32)).astype(np.float32)
    forward = float(local_point[0])
    left = float(local_point[1])
    return np.asarray([-left, -forward], dtype=np.float32)


def _target_point_image(target_point: np.ndarray) -> torch.Tensor:
    from data import draw_target_point

    return torch.from_numpy(draw_target_point(target_point)).unsqueeze(0)


def parse_latent_transfuser_obs(
    obs_img: Dict[str, Any],
    obs_info: Dict[str, Any],
    config: Any,
    device: torch.device,
    target_point_index: int = 4,
) -> Dict[str, torch.Tensor]:
    image = _prepare_rgb(obs_img, config).to(device=device, dtype=torch.float32)
    target_point_np = _target_point(obs_info, target_point_index)
    target_point = torch.from_numpy(target_point_np.reshape(1, 2)).to(device=device, dtype=torch.float32)
    target_point_image = _target_point_image(target_point_np).to(device=device, dtype=torch.float32)
    lidar_bev = torch.zeros(
        (1, 2, config.lidar_resolution_width, config.lidar_resolution_height),
        device=device,
        dtype=torch.float32,
    )
    gt_velocity = torch.FloatTensor([float(obs_info["ego_velo"])]).to(device=device, dtype=torch.float32)
    velocity = gt_velocity.reshape(1, 1)
    return {
        "image": image,
        "lidar_bev": lidar_bev,
        "target_point": target_point,
        "target_point_image": target_point_image,
        "velocity": velocity,
        "gt_velocity": gt_velocity,
    }
