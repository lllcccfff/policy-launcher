from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as SCR

CAMERA_ORDER = ("FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT")
IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
RESIZE = 0.44
CROP = np.array([0, 140, 704, 396], dtype=np.int32)
FINAL_HW = (256, 704)
FIXED_LIDAR_TOP_TO_EGO = np.array(
    [
        [0.002033272, 0.99970406, 0.024241721, 0.943713],
        [-0.9999805, 0.002175657, -0.005848639, 0.0],
        [-0.00589965, -0.024229359, 0.99968904, 1.24523],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
IMAGE_AUG_MATRIX = np.array(
    [
        [RESIZE, 0.0, -float(CROP[0]), 0.0],
        [0.0, RESIZE, -float(CROP[1]), 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _latest_rgb(obs_img: Dict[str, Any], camera: str) -> np.ndarray:
    arr = np.asarray(obs_img[camera])
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{camera} image must have shape HxWx3 or TxHxWx3, got {arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))


def _projection_mat_from_sim_cameras(obs_info: Dict[str, Any]) -> np.ndarray:
    camera_params = obs_info["cam_params"]
    projections = []
    for camera in CAMERA_ORDER:
        cam_info = camera_params[camera]
        intrinsic = np.asarray(cam_info["K"], dtype=np.float32)
        if intrinsic.shape == (3, 3):
            k3 = intrinsic
        elif intrinsic.shape == (4, 4):
            k3 = intrinsic[:3, :3]
        else:
            raise ValueError(f"{camera} K must be 3x3 or 4x4, got {intrinsic.shape}")

        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = k3

        ego2camera = np.asarray(cam_info["ego2camera"], dtype=np.float32)
        if ego2camera.shape != (4, 4):
            raise ValueError(f"{camera} ego2camera must be 4x4, got {ego2camera.shape}")
        projections.append((IMAGE_AUG_MATRIX @ viewpad @ ego2camera @ FIXED_LIDAR_TOP_TO_EGO).astype(np.float32))
    return np.stack(projections, axis=0)


def _preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.shape[:2] != (900, 1600):
        raise ValueError(f"SparseDrive expects 900x1600 images, got {image_rgb.shape[:2]}")
    resized = Image.fromarray(image_rgb).resize((int(1600 * RESIZE), int(900 * RESIZE))).crop(tuple(CROP.tolist()))
    image = np.asarray(resized, dtype=np.float32)
    if image.shape[:2] != FINAL_HW:
        raise ValueError(f"SparseDrive image preprocessing produced {image.shape[:2]}, expected {FINAL_HW}")
    image = (image - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)


def _ego_to_global(obs_info: Dict[str, Any]) -> np.ndarray:
    ego_rot = np.asarray(obs_info["ego_rot"], dtype=np.float32).reshape(3)
    ego_pos = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(3)
    ego_to_global = np.eye(4, dtype=np.float32)
    ego_to_global[:3, :3] = SCR.from_euler("XYZ", ego_rot).as_matrix().astype(np.float32)
    ego_to_global[:3, 3] = ego_pos
    return ego_to_global


def _world_vec_to_ego(obs_info: Dict[str, Any], key: str) -> np.ndarray:
    ego_rot = np.asarray(obs_info["ego_rot"], dtype=np.float32).reshape(3)
    vec = np.asarray(obs_info[key], dtype=np.float32).reshape(3)
    world_to_ego = SCR.from_euler("XYZ", ego_rot).as_matrix().astype(np.float32).T
    return (world_to_ego @ vec).astype(np.float32)


def _ego_status(obs_info: Dict[str, Any]) -> np.ndarray:
    accel = _world_vec_to_ego(obs_info, "accelerate")
    rotation_rate = np.asarray(obs_info["angular_velocity"], dtype=np.float32).reshape(3)
    velocity = _world_vec_to_ego(obs_info, "linear_velocity")
    steering = np.array([float(obs_info["ego_steer"])], dtype=np.float32)
    return np.concatenate([accel, rotation_rate, velocity, steering], axis=0).astype(np.float32)


def _command_one_hot(command: int) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    out[int(command)] = 1.0
    return out


def parse_sparsedrive_obs(
    obs_img: Dict[str, Any],
    obs_info: Dict[str, Any],
) -> Dict[str, Any]:
    missing = [name for name in CAMERA_ORDER if name not in obs_img]
    if missing:
        raise KeyError(f"SparseDrive requires cameras {CAMERA_ORDER}, missing {missing}")

    images = [_preprocess_image(_latest_rgb(obs_img, camera)) for camera in CAMERA_ORDER]
    projection_mat = _projection_mat_from_sim_cameras(obs_info)
    image_wh = np.tile(np.array([[float(FINAL_HW[1]), float(FINAL_HW[0])]], dtype=np.float32), (len(CAMERA_ORDER), 1))

    ego_to_global = _ego_to_global(obs_info)
    t_global = (ego_to_global @ FIXED_LIDAR_TOP_TO_EGO).astype(np.float32)
    t_global_inv = np.linalg.inv(t_global).astype(np.float32)
    timestamp = float(obs_info["current_timestamp"])

    return {
        "img": torch.from_numpy(np.stack(images, axis=0)).float(),
        "projection_mat": torch.from_numpy(projection_mat).float(),
        "image_wh": torch.from_numpy(image_wh).float(),
        "timestamp": torch.tensor(timestamp, dtype=torch.float32),
        "T_global": torch.from_numpy(t_global).float(),
        "T_global_inv": torch.from_numpy(t_global_inv).float(),
        "img_metas": {
            "T_global": t_global,
            "T_global_inv": t_global_inv,
            "timestamp": timestamp,
        },
        "ego_status": torch.from_numpy(_ego_status(obs_info)).float(),
        "gt_ego_fut_cmd": torch.from_numpy(_command_one_hot(int(obs_info["command"]))).float(),
        "gt_ego_fut_trajs": torch.zeros((6, 2), dtype=torch.float32),
        "gt_ego_fut_masks": torch.zeros((6,), dtype=torch.float32),
    }
