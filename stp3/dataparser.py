from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from pyquaternion import Quaternion

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
CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300
CAMERA_FOCAL = 330.0
CAMERA_OFFSETS = {
    "FRONT": (1.70079118954, 0.0159456324149, 1.51095763913),
    "FRONT_LEFT": (1.52387798135, 0.494631336551, 1.50932822144),
    "FRONT_RIGHT": (1.5508477543, -0.493404796419, 1.49574800619),
    "BACK": (0.0283260309358, 0.00345136761476, 1.57910346144),
}
CARLA_SAMPLE_INTERVAL = 0.5

NORMALISE_IMAGE = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _latest_rgb(obs_img: Dict[str, Any], camera: str) -> np.ndarray:
    arr = np.asarray(obs_img[camera])
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{camera} image must have shape HxWx3 or TxHxWx3, got {arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))


def _preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    cropped = scale_and_crop_image(Image.fromarray(image_rgb), scale=1.0, crop=IMAGE_CROP)
    return NORMALISE_IMAGE(np.asarray(cropped).copy()).to(dtype=torch.float32)


def build_image_tensor(
    history_images: Sequence[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    frames = []
    for obs_img in history_images:
        camera_tensors = [_preprocess_image(_latest_rgb(obs_img, camera)) for camera in CAMERA_ORDER]
        frames.append(torch.stack(camera_tensors, dim=0))
    return torch.stack(frames, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)


def _camera_to_ego(dof: Sequence[float]) -> np.ndarray:
    yaw = math.radians(float(dof[5]))
    rotation = Quaternion(scalar=np.cos(yaw / 2.0), vector=[0.0, 0.0, np.sin(yaw / 2.0)])
    translation = np.asarray(dof[:3], dtype=np.float32)[:, None]
    return np.vstack(
        [
            np.hstack((rotation.rotation_matrix.astype(np.float32), translation)),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def fixed_camera_parameters(receptive_field: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    camera_dofs = [
        [*CAMERA_OFFSETS["FRONT"], 0.0, 0.0, 0.0],
        [*CAMERA_OFFSETS["FRONT_LEFT"], 0.0, 0.0, -60.0],
        [*CAMERA_OFFSETS["FRONT_RIGHT"], 0.0, 0.0, 60.0],
        [*CAMERA_OFFSETS["BACK"], 0.0, 0.0, 180.0],
    ]
    extrinsics = torch.from_numpy(np.stack([_camera_to_ego(dof) for dof in camera_dofs], axis=0)).float()

    intrinsic = torch.tensor(
        [
            [CAMERA_FOCAL, 0.0, CAMERA_WIDTH / 2.0],
            [0.0, CAMERA_FOCAL, CAMERA_HEIGHT / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    intrinsic = update_intrinsics(
        intrinsic,
        (CAMERA_HEIGHT - IMAGE_CROP) / 2.0,
        (CAMERA_WIDTH - IMAGE_CROP) / 2.0,
        scale_width=1.0,
        scale_height=1.0,
    )
    intrinsics = intrinsic.unsqueeze(0).expand(len(CAMERA_ORDER), 3, 3).contiguous()

    extrinsics = extrinsics.unsqueeze(0).expand(receptive_field, len(CAMERA_ORDER), 4, 4)
    intrinsics = intrinsics.unsqueeze(0).expand(receptive_field, len(CAMERA_ORDER), 3, 3)
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


def target_point_from_route(obs_info: Dict[str, Any], target_route_idx: int, device: torch.device) -> torch.Tensor:
    route = np.asarray(obs_info["route_waypoints"], dtype=np.float32)
    if route.ndim != 2 or route.shape[1] < 2:
        raise ValueError(f"route_waypoints must have shape Nx2 or NxM, got {route.shape}")
    if target_route_idx < 0 or target_route_idx >= route.shape[0]:
        raise ValueError(f"target_route_idx={target_route_idx} outside route length {route.shape[0]}")

    ego_pos = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)
    if ego_pos.size < 2:
        raise ValueError(f"ego_pos must contain at least xy, got {ego_pos}")

    delta = route[target_route_idx, :2] - ego_pos[:2]
    theta = np.pi / 2.0 + float(obs_info["heading_theta"])
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    local = rot.T.dot(delta) * np.array([1.0, -1.0], dtype=np.float32)
    return torch.from_numpy(local.astype(np.float32)).unsqueeze(0).to(device=device, dtype=torch.float32)


def parse_stp3_obs(
    history_images: Sequence[Dict[str, Any]],
    history_states: Sequence[Dict[str, Any]],
    obs_info: Dict[str, Any],
    cfg: Any,
    device: torch.device,
    target_route_idx: int,
    last_steer: float,
) -> Dict[str, Any]:
    receptive_field = int(cfg.TIME_RECEPTIVE_FIELD)
    if len(history_images) != receptive_field:
        raise ValueError(f"Expected {receptive_field} image history frames, got {len(history_images)}")
    if len(history_states) != receptive_field:
        raise ValueError(f"Expected {receptive_field} state history frames, got {len(history_states)}")

    extrinsics, intrinsics = fixed_camera_parameters(receptive_field, device)
    return {
        "image": build_image_tensor(history_images, device),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "future_egomotion": build_future_egomotion(history_states, device),
        "sample_trajectory": sample_trajectories(cfg, float(obs_info["ego_velo"]), float(last_steer), device),
        "target_point": target_point_from_route(obs_info, target_route_idx, device),
        "command": [command_string(obs_info["command"])],
    }
