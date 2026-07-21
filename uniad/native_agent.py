from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from rl_framework.common.tensor import to_device
from rl_framework.common.trajectory import lidar_to_ego_front_left, traj2control
from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4, parse_uniad_obs

def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class UniADNativeAgent:
    def __init__(
        self,
        uniad_model: torch.nn.Module,
    ):
        self.uniad_model = uniad_model
        self.model_device = self._infer_model_device()
        self.last_trajectory: np.ndarray | None = None
        self.last_raw_data: Dict[str, Any] | None = None

    def _infer_model_device(self) -> torch.device:
        model_impl = self.uniad_model.module if hasattr(self.uniad_model, "module") else self.uniad_model
        return next(model_impl.parameters()).device

    def predict_action(
        self,
        obs_img: Dict[str, np.ndarray],
        obs_info: Dict[str, Any],
    ) -> Tuple[float, float]:
        raw_data = parse_uniad_obs(obs_img, obs_info)
        model_input = to_device(raw_data, self.model_device)

        with torch.no_grad():
            results = self.uniad_model(
                return_loss=False,
                rescale=True,
                feature_extractor=False,
                **model_input,
            )

        traj = _to_numpy(results[0]["planning"]["result_planning"]["sdc_traj"]).astype(np.float32)[0]
        pred_traj = lidar_to_ego_front_left(traj, LIDAR_TO_EGO_4X4)
        acc, steer = traj2control(pred_traj, obs_info)
        self.last_trajectory = pred_traj
        self.last_raw_data = raw_data
        return float(steer), float(acc)
