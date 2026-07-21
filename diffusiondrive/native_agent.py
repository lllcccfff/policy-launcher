from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from rl_framework.common.trajectory import lidar_to_ego_front_left, traj2control
from rl_framework.diffusiondrive.dataparser import FIXED_LIDAR_TOP_TO_EGO, parse_diffusiondrive_obs


class DiffusionDriveNativeAgent:
    def __init__(self, diffusiondrive_model: torch.nn.Module):
        self.diffusiondrive_model = diffusiondrive_model
        self.model_device = self._infer_model_device()
        self.last_trajectory: np.ndarray | None = None

    def _infer_model_device(self) -> torch.device:
        return next(self.diffusiondrive_model.parameters()).device

    def reset(self) -> None:
        self.last_trajectory = None
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

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        raw_data = parse_diffusiondrive_obs(obs_img, obs_info)
        img, data = self._batch_raw_data(raw_data)
        with torch.inference_mode():
            output = self.diffusiondrive_model.forward(img, **data)
        planning = output[0]["img_bbox"]["final_planning"].detach().cpu().numpy().astype(np.float32)
        if planning.ndim != 2 or planning.shape[1] != 2:
            raise ValueError(f"DiffusionDrive final_planning must have shape Nx2, got {planning.shape}")
        self.last_trajectory = lidar_to_ego_front_left(planning, FIXED_LIDAR_TOP_TO_EGO)
        acc, steer = traj2control(self.last_trajectory, obs_info, horizon=3.0)
        return float(steer), float(acc)
