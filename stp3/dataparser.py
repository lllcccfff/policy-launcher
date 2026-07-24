from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation as SCR

from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4 as LIDAR_TOP_TO_EGO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STP3_ROOT = REPO_ROOT / "ST-P3"


def ensure_stp3_root(stp3_root: Union[str, Path] = DEFAULT_STP3_ROOT) -> Path:
    root = Path(stp3_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ST-P3 root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ensure_stp3_root()

from stp3.utils.geometry import invert_matrix_egopose_numpy, mat2pose_vec, resize_and_crop_image, update_intrinsics  # noqa: E402
import stp3.utils.sampler as trajectory_sampler  # noqa: E402

CAMERA_ORDER = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT")
SAMPLE_INTERVAL = 0.5

NORMALISE_IMAGE = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _image_augmentation(cfg: Any) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    original_height = int(cfg.IMAGE.ORIGINAL_HEIGHT)
    original_width = int(cfg.IMAGE.ORIGINAL_WIDTH)
    final_height, final_width = map(int, cfg.IMAGE.FINAL_DIM)
    resize_scale = float(cfg.IMAGE.RESIZE_SCALE)
    resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
    crop_w = int(max(0, (resize_dims[0] - final_width) / 2))
    crop_h = int(cfg.IMAGE.TOP_CROP)
    return resize_dims, (crop_w, crop_h, crop_w + final_width, crop_h + final_height)


def build_image_tensor(
    history_images: Sequence[Dict[str, Any]],
    cfg: Any,
    device: torch.device,
) -> torch.Tensor:
    resize_dims, crop = _image_augmentation(cfg)
    expected_shape = (int(cfg.IMAGE.ORIGINAL_HEIGHT), int(cfg.IMAGE.ORIGINAL_WIDTH))
    frames = []
    for obs_img in history_images:
        camera_tensors = []
        for camera in CAMERA_ORDER:
            image_rgb = np.asarray(obs_img[camera], dtype=np.uint8)
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                raise ValueError(f"{camera} image must have shape HxWx3, got {image_rgb.shape}")
            if image_rgb.shape[:2] != expected_shape:
                raise ValueError(f"NuScenes ST-P3 expects {expected_shape[1]}x{expected_shape[0]} {camera} images, got {image_rgb.shape[:2][::-1]}")
            image = resize_and_crop_image(Image.fromarray(image_rgb), resize_dims=resize_dims, crop=crop)
            camera_tensors.append(NORMALISE_IMAGE(image))
        frames.append(torch.stack(camera_tensors, dim=0))
    return torch.stack(frames, dim=0).unsqueeze(0).to(device=device)


def camera_parameters(
    cam_params: Dict[str, Dict[str, Any]],
    receptive_field: int,
    cfg: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, crop = _image_augmentation(cfg)
    resize_scale = float(cfg.IMAGE.RESIZE_SCALE)
    extrinsics = []
    intrinsics = []
    ego_to_lidar = np.linalg.inv(LIDAR_TOP_TO_EGO).astype(np.float32)
    for camera in CAMERA_ORDER:
        cam_info = cam_params[camera]
        ego2camera = np.asarray(cam_info["ego2camera"], dtype=np.float32)
        if ego2camera.shape != (4, 4):
            raise ValueError(f"{camera} ego2camera must be 4x4, got {ego2camera.shape}")
        camera_to_lidar = ego_to_lidar @ np.linalg.inv(ego2camera)
        extrinsics.append(camera_to_lidar.astype(np.float32))

        intrinsic = torch.as_tensor(cam_info["K"], dtype=torch.float32)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"{camera} K must be 3x3, got {tuple(intrinsic.shape)}")
        intrinsics.append(
            update_intrinsics(
                intrinsic,
                top_crop=crop[1],
                left_crop=crop[0],
                scale_width=resize_scale,
                scale_height=resize_scale,
            )
        )

    extrinsics = torch.from_numpy(np.stack(extrinsics, axis=0)).unsqueeze(0).expand(
        receptive_field, len(CAMERA_ORDER), 4, 4
    )
    intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0).expand(receptive_field, len(CAMERA_ORDER), 3, 3)
    return (
        extrinsics.unsqueeze(0).to(device=device, dtype=torch.float32),
        intrinsics.unsqueeze(0).to(device=device, dtype=torch.float32),
    )


def lidar_to_world(obs_info: Dict[str, Any]) -> np.ndarray:
    ego_rot = np.asarray(obs_info["ego_rot"], dtype=np.float32).reshape(3)
    ego_pos = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(3)
    ego_to_world = np.eye(4, dtype=np.float32)
    ego_to_world[:3, :3] = SCR.from_euler("XYZ", ego_rot).as_matrix().astype(np.float32)
    ego_to_world[:3, 3] = ego_pos
    return (ego_to_world @ LIDAR_TOP_TO_EGO).astype(np.float32)


def build_future_egomotion(history_lidar_to_world: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    if len(history_lidar_to_world) < 2:
        raise ValueError("NuScenes ST-P3 requires at least two lidar poses")

    future_egomotions = []
    for t0, t1 in zip(history_lidar_to_world[:-1], history_lidar_to_world[1:]):
        future_egomotion = invert_matrix_egopose_numpy(t1).dot(t0)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0
        future_egomotions.append(mat2pose_vec(torch.from_numpy(future_egomotion).float()))
    future_egomotions.append(torch.zeros(6, dtype=torch.float32))
    return torch.stack(future_egomotions, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)


def _longitudinal_velocity(obs_info: Dict[str, Any]) -> float:
    ego_rot = np.asarray(obs_info["ego_rot"], dtype=np.float32).reshape(3)
    linear_velocity = np.asarray(obs_info["linear_velocity"], dtype=np.float32).reshape(3)
    ego_velocity = SCR.from_euler("XYZ", ego_rot).as_matrix().astype(np.float32).T @ linear_velocity
    return float(ego_velocity[0])


def sample_trajectories(cfg: Any, obs_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
    velocity = _longitudinal_velocity(obs_info)
    steering = float(obs_info["ego_steer"])
    kappa = 2.0 * steering / 2.588
    tangent = np.array([0.0, 1.0], dtype=np.float32)
    normal = np.array([1.0, 0.0], dtype=np.float32) if kappa <= 0.0 else np.array([-1.0, 0.0], dtype=np.float32)
    t_end = float(cfg.N_FUTURE_FRAMES) * SAMPLE_INTERVAL
    t_interval = SAMPLE_INTERVAL / 10.0
    tt = np.arange(0.0, t_end + t_interval, t_interval)
    sampled_fine = trajectory_sampler.sample(velocity, kappa, tangent, normal, tt, int(cfg.PLANNING.SAMPLE_NUM))
    return torch.from_numpy(sampled_fine[:, ::10]).float().unsqueeze(0).to(device=device)


def command_string(command: Any) -> str:
    command_int = int(command)
    if command_int == 1:
        return "LEFT"
    if command_int == 0:
        return "RIGHT"
    if command_int == 2:
        return "FORWARD"
    raise ValueError(f"NuScenes ST-P3 command must be 0, 1, or 2, got {command!r}")


def parse_stp3_obs(
    history_images: Sequence[Dict[str, Any]],
    history_lidar_to_world: Sequence[np.ndarray],
    obs_info: Dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> Dict[str, Any]:
    receptive_field = int(cfg.TIME_RECEPTIVE_FIELD)
    if len(history_images) != receptive_field or len(history_lidar_to_world) != receptive_field:
        raise ValueError(
            f"NuScenes ST-P3 requires {receptive_field} history frames and poses, got "
            f"{len(history_images)} frames and {len(history_lidar_to_world)} poses"
        )

    extrinsics, intrinsics = camera_parameters(obs_info["cam_params"], receptive_field, cfg, device)
    return {
        "image": build_image_tensor(history_images, cfg, device),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "future_egomotion": build_future_egomotion(history_lidar_to_world, device),
        "sample_trajectory": sample_trajectories(cfg, obs_info, device),
        "target_point": torch.zeros((1, 2), dtype=torch.float32, device=device),
        "command": [command_string(obs_info["command"])],
    }
