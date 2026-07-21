from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image

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

from stp3.datas.CarlaData import scale_and_crop_image  # noqa: E402
from stp3.utils.geometry import invert_matrix_egopose_numpy, mat2pose_vec, update_intrinsics  # noqa: E402
import stp3.utils.sampler as trajectory_sampler  # noqa: E402

CAMERA_ORDER = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK")
IMAGE_CROP = 256
CARLA_SAMPLE_INTERVAL = 0.5

NORMALISE_IMAGE = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_image_tensor(
    history_images: Sequence[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    frames = []
    for obs_img in history_images:
        camera_tensors = []
        for camera in CAMERA_ORDER:
            image_rgb = obs_img[camera]
            cropped = scale_and_crop_image(Image.fromarray(image_rgb), scale=1.0, crop=IMAGE_CROP)
            camera_tensors.append(NORMALISE_IMAGE(cropped.copy()))
        frames.append(torch.stack(camera_tensors, dim=0))
    return torch.stack(frames, dim=0).unsqueeze(0).to(device=device)


def camera_parameters(
    cam_params: Dict[str, Dict[str, Any]],
    receptive_field: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    extrinsics = []
    intrinsics = []
    for camera in CAMERA_ORDER:
        cam_info = cam_params[camera]
        extrinsics.append(np.linalg.inv(np.asarray(cam_info["ego2camera"], dtype=np.float32)))

        height = int(cam_info["H"])
        width = int(cam_info["W"])
        intrinsics.append(
            update_intrinsics(
                torch.as_tensor(cam_info["K"], dtype=torch.float32),
                height // 2 - IMAGE_CROP // 2,
                width // 2 - IMAGE_CROP // 2,
                scale_width=1.0,
                scale_height=1.0,
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


def _pose_matrix(x: float, y: float, theta: float) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[:2, :2] = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    matrix[2, 2] = 1.0
    matrix[0, 3] = float(x)
    matrix[1, 3] = float(y)
    matrix[3, 3] = 1.0
    return matrix


def build_future_egomotion(history_states: Sequence[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    if len(history_states) < 2:
        raise ValueError("ST-P3 future egomotion requires at least two history states")

    future_egomotions = []
    for i in range(len(history_states) - 1):
        state_t0 = history_states[i]
        state_t1 = history_states[i + 1]
        pos_t0 = np.asarray(state_t0["ego_pos"], dtype=np.float32).reshape(-1)
        pos_t1 = np.asarray(state_t1["ego_pos"], dtype=np.float32).reshape(-1)
        theta_t0 = float(state_t0["heading_theta"])
        theta_t1 = float(state_t1["heading_theta"])
        egopose_t0 = _pose_matrix(float(pos_t0[0]), float(pos_t0[1]), theta_t0)
        egopose_t1 = _pose_matrix(float(pos_t1[0]), float(pos_t1[1]), theta_t1)
        future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0
        future_egomotions.append(mat2pose_vec(torch.from_numpy(future_egomotion).float()).unsqueeze(0))

    return torch.cat(future_egomotions, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)


def sample_trajectories(cfg: Any, velocity: float, steering: float, device: torch.device) -> torch.Tensor:
    kappa = 2.0 * float(steering) / 2.588
    tangent = np.array([0.0, 1.0], dtype=np.float32)
    normal = np.array([1.0, 0.0], dtype=np.float32) if kappa <= 0.0 else np.array([-1.0, 0.0], dtype=np.float32)
    t_end = float(cfg.N_FUTURE_FRAMES) * CARLA_SAMPLE_INTERVAL
    t_interval = CARLA_SAMPLE_INTERVAL / 10.0
    tt = np.arange(0.0, t_end + t_interval, t_interval)
    sampled_fine = trajectory_sampler.sample(float(velocity), kappa, tangent, normal, tt, int(cfg.PLANNING.SAMPLE_NUM))
    sampled = sampled_fine[:, ::10]
    return torch.from_numpy(sampled).float().unsqueeze(0).to(device=device, dtype=torch.float32)


def command_string(command: Any) -> str:
    command_int = int(command)
    if command_int == 1:
        return "LEFT"
    if command_int == 0:
        return "RIGHT"
    if command_int == 2:
        return "FORWARD"
    return "LANE"


def target_point_from_waypoint(obs_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
    target_waypoint = np.asarray(obs_info["target_waypoint"], dtype=np.float32)
    ego_pos = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)

    delta = target_waypoint[:2] - ego_pos[:2]
    theta = np.pi / 2.0 + float(obs_info["heading_theta"])
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    local = rot.T.dot(delta) * np.array([1.0, -1.0], dtype=np.float32)
    return torch.from_numpy(local).unsqueeze(0).to(device=device, dtype=torch.float32)


def parse_stp3_obs(
    history_images: Sequence[Dict[str, Any]],
    history_states: Sequence[Dict[str, Any]],
    obs_info: Dict[str, Any],
    cfg: Any,
    device: torch.device,
    last_steer: float,
) -> Dict[str, Any]:
    receptive_field = int(cfg.TIME_RECEPTIVE_FIELD)

    extrinsics, intrinsics = camera_parameters(obs_info["cam_params"], receptive_field, device)
    return {
        "image": build_image_tensor(history_images, device),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "future_egomotion": build_future_egomotion(history_states, device),
        "sample_trajectory": sample_trajectories(cfg, float(obs_info["ego_velo"]), float(last_steer), device),
        "target_point": target_point_from_waypoint(obs_info, device),
        "command": [command_string(obs_info["command"])],
    }
