from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from rl_framework.common.trajectory import traj2control
from rl_framework.diffusiondrive.dataparser import FIXED_LIDAR_TOP_TO_EGO, parse_diffusiondrive_obs


def _lidar_top_points_to_ego_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"LIDAR_TOP points must have shape Nx2, got {points.shape}")
    homogeneous = np.ones((points.shape[0], 4), dtype=np.float32)
    homogeneous[:, :2] = points
    homogeneous[:, 2] = 0.0
    ego_points = homogeneous @ FIXED_LIDAR_TOP_TO_EGO.T
    return ego_points[:, :2].astype(np.float32)


class DiffusionDriveNativeAgent:
    def __init__(self, diffusiondrive_model: torch.nn.Module):
        self.diffusiondrive_model = diffusiondrive_model
        self.model_device = self._infer_model_device()

    def _infer_model_device(self) -> torch.device:
        return next(self.diffusiondrive_model.parameters()).device

    def reset(self) -> None:
        head = self.diffusiondrive_model.head
        head.det_head.instance_bank.reset()
        head.map_head.instance_bank.reset()
        head.motion_plan_head.instance_queue.reset()

    def _batch_raw_data(self, raw_data: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        img = raw_data["img"].to(self.model_device).unsqueeze(0)
        data: Dict[str, Any] = {}
        for key, value in raw_data.items():
            if key == "img":
                continue
            if key == "img_metas":
                data[key] = [value]
            elif torch.is_tensor(value):
                data[key] = value.to(self.model_device).unsqueeze(0)
            else:
                data[key] = value
        return img, data

    def prepare_raw_input(
        self,
        obs_img: Dict[str, Any],
        obs_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return parse_diffusiondrive_obs(obs_img, obs_info)

    def predict_traj_from_raw_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        img, data = self._batch_raw_data(raw_data)
        with torch.inference_mode():
            output = self.diffusiondrive_model.forward(img, **data)
        planning = output[0]["img_bbox"]["final_planning"].detach().cpu().numpy().astype(np.float32)
        if planning.ndim != 2 or planning.shape[1] != 2:
            raise ValueError(f"DiffusionDrive final_planning must have shape Nx2, got {planning.shape}")
        return _lidar_top_points_to_ego_xy(planning)

    def predict_action_from_traj(self, trajectory: np.ndarray, obs_info: Dict[str, Any]) -> Tuple[float, float]:
        acc, steer = traj2control(trajectory, obs_info, horizon=3.0)
        return float(steer), float(acc)
