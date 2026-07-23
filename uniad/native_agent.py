from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from rl_framework.common.tensor import to_device
from rl_framework.common.trajectory import lidar_to_ego_front_left
from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4, parse_uniad_obs

def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class UniADNativeAgent:
    def __init__(self, device: str):
        from rl_framework.uniad.loader import create_uniad

        config = {
            "config_path": "UniAD_SIM/projects/configs/stage2_e2e/base_e2e.py",
            "checkpoint_path": "UniAD_SIM/ckpts/uniad_base_e2e.pth",
            "AD_root": "./UniAD_SIM",
            "device": device,
        }
        self.uniad_model = create_uniad(config)
        self.model_device = self._infer_model_device()

    def _infer_model_device(self) -> torch.device:
        model_impl = self.uniad_model.module if hasattr(self.uniad_model, "module") else self.uniad_model
        return next(model_impl.parameters()).device

    def predict_action(
        self,
        obs_img: Dict[str, np.ndarray],
        obs_info: Dict[str, Any],
    ) -> np.ndarray:
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
        return pred_traj
