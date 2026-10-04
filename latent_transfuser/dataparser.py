from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from policy_launcher.common.coordinates import (
    metadrive_to_carla_xy,
    metadrive_to_carla_yaw,
    rotation_matrix_2d,
)


REQUIRED_CAMERAS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")


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
    return np.transpose(cropped, (2, 0, 1)).copy()


def _prepare_rgb(obs_img: Dict[str, Any], config: Any) -> torch.Tensor:
    crops = []
    for camera in REQUIRED_CAMERAS:
        rgb = obs_img[camera]
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


def _target_point(obs_info: Dict[str, Any]) -> np.ndarray:
    target_waypoint = obs_info["carla_style_target"]
    ego_pos = obs_info["ego_pos"]

    target_waypoint_carla = metadrive_to_carla_xy(target_waypoint)
    ego_pos_carla = metadrive_to_carla_xy(ego_pos)
    delta = target_waypoint_carla - ego_pos_carla
    yaw = metadrive_to_carla_yaw(obs_info["heading_theta"])
    rotation = rotation_matrix_2d(np.pi / 2.0 + yaw)
    return rotation.T @ delta


def _target_point_image(target_point: np.ndarray) -> torch.Tensor:
    from data import draw_target_point

    return torch.from_numpy(draw_target_point(target_point)).unsqueeze(0)


def parse_latent_transfuser_obs(
    obs_img: Dict[str, Any],
    obs_info: Dict[str, Any],
    config: Any,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    image = _prepare_rgb(obs_img, config).to(device=device, dtype=torch.float32)
    target_point_np = _target_point(obs_info)
    target_point = torch.from_numpy(target_point_np.reshape(1, 2)).to(device=device)
    target_point_image = _target_point_image(target_point_np).to(device=device, dtype=torch.float32)
    lidar_bev = torch.zeros(
        (1, 2, config.lidar_resolution_width, config.lidar_resolution_height),
        device=device,
    )
    velocity = torch.FloatTensor([float(obs_info["ego_velo"])]).to(device=device)
    return {
        "image": image,
        "lidar_bev": lidar_bev,
        "target_point": target_point,
        "target_point_image": target_point_image,
        "velocity": velocity.reshape(1, 1),
    }
