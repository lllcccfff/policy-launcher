from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from policy_launcher.common.tensor import to_device
from policy_launcher.common.trajectory import lidar_to_ego_front_left
from policy_launcher.genad.dataparser import parse_genad_obs
from policy_launcher.uniad.dataparser import LIDAR_TO_EGO_4X4


def _decode_trajectory(pts_bbox: Dict[str, Any]) -> np.ndarray:
    increments = pts_bbox["ego_fut_preds"].detach().cpu().numpy().astype(np.float32)
    command = pts_bbox["ego_fut_cmd"].detach().cpu().numpy().reshape(-1)
    trajectory_lidar = np.cumsum(increments[int(np.argmax(command))], axis=0).astype(np.float32)
    return lidar_to_ego_front_left(trajectory_lidar, LIDAR_TO_EGO_4X4)


class GenADNativeAgent:
    def __init__(self, device: str):
        from policy_launcher.genad.loader import create_genad

        self.model = create_genad(
            {
                "config_path": "GenAD/projects/configs/GenAD/GenAD_config.py",
                "checkpoint_path": "GenAD/ckpts/checkpoints.pth",
                "AD_root": "GenAD",
                "device": device,
            }
        )
        model_impl = self.model.module if hasattr(self.model, "module") else self.model
        self.model_device = next(model_impl.parameters()).device

    def predict_action(
        self,
        obs_img: Dict[str, np.ndarray],
        obs_info: Dict[str, Any],
    ) -> np.ndarray:
        model_input = to_device(parse_genad_obs(obs_img, obs_info), self.model_device)
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **model_input)

        trajectory = _decode_trajectory(results[0]["pts_bbox"])
        return trajectory
