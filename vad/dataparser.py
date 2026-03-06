import math
from typing import Dict

import mmcv
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

VAD_IMG_NORM_CFG = {
    # Keep legacy VAD preprocessing used by existing RL pipeline.
    "mean": np.array([103.530, 116.280, 123.675], dtype=np.float32),
    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "to_rgb": False,
}

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

CAMERA_ALIASES = {
    "CAM_FRONT": ("CAM_FRONT", "FRONT"),
    "CAM_FRONT_RIGHT": ("CAM_FRONT_RIGHT", "FRONT_RIGHT"),
    "CAM_FRONT_LEFT": ("CAM_FRONT_LEFT", "FRONT_LEFT"),
    "CAM_BACK": ("CAM_BACK", "BACK"),
    "CAM_BACK_LEFT": ("CAM_BACK_LEFT", "BACK_LEFT"),
    "CAM_BACK_RIGHT": ("CAM_BACK_RIGHT", "BACK_RIGHT"),
}

LIDAR_TO_EGO_4X4 = np.array(
    [
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.0],
        [0, 0, 1, 1.5],
        [0, 0, 0, 1.0],
    ],
    dtype=np.float32,
)

def get_vad_img_norm_cfg() -> Dict[str, object]:
    return {
        "mean": np.array(VAD_IMG_NORM_CFG["mean"], dtype=np.float32),
        "std": np.array(VAD_IMG_NORM_CFG["std"], dtype=np.float32),
        "to_rgb": bool(VAD_IMG_NORM_CFG["to_rgb"]),
    }


def command_to_one_hot(command: int, device: torch.device) -> torch.Tensor:
    # Simulator convention: 0=right, 1=left, 2=straight.
    if command not in (0, 1, 2):
        raise ValueError(f"Unsupported command id: {command}")
    one_hot = torch.zeros(3, dtype=torch.float32, device=device)
    one_hot[command] = 1.0
    return one_hot


def _resolve_camera_key(data: dict, canonical_name: str) -> str:
    for key in CAMERA_ALIASES[canonical_name]:
        if key in data:
            return key
    raise KeyError(
        f"Missing `{canonical_name}` in data. "
        f"Accepted aliases: {CAMERA_ALIASES[canonical_name]}, got: {sorted(data.keys())}"
    )


def _to_k_4x4(cam_info: dict) -> np.ndarray:
    k = np.asarray(cam_info["K"], dtype=np.float32)
    if k.shape == (4, 4):
        return k
    if k.shape == (3, 3):
        k4 = np.eye(4, dtype=np.float32)
        k4[:3, :3] = k
        return k4
    raise ValueError(f"`K` must be 3x3 or 4x4, got {k.shape}")

def _to_l2c(cam_info: dict) -> np.ndarray:
    ego2camera = np.asarray(cam_info["ego2camera"], dtype=np.float32)
    if ego2camera.shape != (4, 4):
        raise ValueError(f"`ego2camera` must be 4x4, got {ego2camera.shape}")
    return ego2camera @ LIDAR_TO_EGO_4X4


def _build_can_bus(info) -> np.ndarray:
    ego_rot = np.asarray(info["ego_rot"], dtype=np.float32)
    ego_pos = np.asarray(info["ego_pos"], dtype=np.float32)
    r_world_ego = SCR.from_euler("XYZ", ego_rot).as_matrix().astype(np.float32)

    # Keep can_bus as float64 so scalar indexing (e.g. can_bus[-1]) yields a
    # Python-float-compatible value for downstream torchvision.rotate().
    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[:3] = ego_pos
    v_quat = SCR.from_euler("XYZ", ego_rot).as_quat()[[3, 0, 1, 2]]
    can_bus[3:7] = v_quat

    linear_vel = np.asarray(info["linear_velocity"], dtype=np.float32)
    can_bus[7:10] = r_world_ego.T @ linear_vel

    linear_acc = np.asarray(info["linear_acceleration"], dtype=np.float32)
    can_bus[10:13] = r_world_ego.T @ linear_acc

    angular_vel = np.asarray(info["angular_velocity"], dtype=np.float32)
    can_bus[13:16] = angular_vel

    yaw = float(ego_rot[2])
    if yaw < 0:
        yaw += np.pi * 2
    can_bus[-2] = yaw
    can_bus[-1] = yaw / np.pi * 180
    return can_bus


def parse_vad_obs(obs, info, cameras, img_norm_cfg) -> Dict[str, object]:
    del cameras

    cam_params = info["cam_params"]
    imgs = []
    img_shape = []
    lidar2img = []
    scale_factor = 0.8

    for cam in CAMERA_ORDER:
        obs_cam = _resolve_camera_key(obs, cam)
        info_cam = _resolve_camera_key(cam_params, cam)

        im = obs[obs_cam]
        if len(im.shape) == 4:
            im = im[-1]
        im = np.asarray(im, dtype=np.float32)
        if scale_factor != 1.0:
            h, w = im.shape[:2]
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))
            im = cv2.resize(im, (new_w, new_h))

        im_norm = mmcv.imnormalize(
            im,
            img_norm_cfg["mean"],
            img_norm_cfg["std"],
            img_norm_cfg["to_rgb"],
        )
        imgs.append(torch.from_numpy(im_norm).permute(2, 0, 1).contiguous())
        img_shape.append([int(im_norm.shape[0]), int(im_norm.shape[1]), int(im_norm.shape[2])])

        cam_info = cam_params[info_cam]
        k4 = _to_k_4x4(cam_info)
        if scale_factor != 1.0:
            k4 = k4.copy()
            k4[0, :] *= scale_factor
            k4[1, :] *= scale_factor
        l2c = _to_l2c(cam_info)
        lidar2img.append((k4 @ l2c).astype(np.float32))

    img_tensor = torch.stack(imgs, dim=0).unsqueeze(0).float().cuda()
    can_bus = _build_can_bus(info)

    cmd_vec = command_to_one_hot(int(info["command"]), img_tensor.device)
    img_meta = {
        "scene_token": str(info["scene_token"]),
        "can_bus": can_bus,
        "img_shape": img_shape,
        "lidar2img": lidar2img,
    }

    return {
        "img": img_tensor,
        # Keep nested format expected by VAD forward_test.
        "img_metas": [[img_meta]],
        "ego_his_trajs": None,
        "ego_fut_cmd": cmd_vec.view(1, 1, 1, 3),
    }
