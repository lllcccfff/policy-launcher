from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from policy_launcher.common.tensor import to_device
from policy_launcher.common.trajectory import lidar_to_ego_front_left
from policy_launcher.uniad.dataparser import LIDAR_TO_EGO_4X4
from policy_launcher.vad.dataparser import parse_vad_obs

def _decode_traj_from_pts_bbox(pts_bbox: Dict[str, Any]) -> np.ndarray:
    preds = pts_bbox["ego_fut_preds"].detach().cpu().numpy().astype(np.float32)
    cmd = pts_bbox["ego_fut_cmd"]
    mode_idx = np.argmax(cmd.detach().cpu().numpy().reshape(-1))
    increments = preds[mode_idx]

    traj = np.cumsum(increments, axis=0).astype(np.float32)
    return lidar_to_ego_front_left(traj, LIDAR_TO_EGO_4X4)


class VADNativeAgent:
    def __init__(self, device: str):
        from policy_launcher.vad.loader import create_vad

        config = {
            "config_path": "VAD/projects/configs/VAD/VAD_base_stage_2.py",
            "checkpoint_path": "VAD/ckpts/VAD_base.pth",
            "AD_root": "./VAD",
            "device": device,
        }
        self.vad_model = create_vad(config)
        self.model_device = self._infer_model_device()

    def _infer_model_device(self) -> torch.device:
        model_impl = self.vad_model.module if hasattr(self.vad_model, "module") else self.vad_model
        return next(model_impl.parameters()).device

    def predict_action(self, obs_img: Dict[str, np.ndarray], obs_info: Dict[str, Any]) -> np.ndarray:
        raw_data = parse_vad_obs(obs_img, obs_info)
        raw_data = to_device(raw_data, self.model_device)

        with torch.no_grad():
            results = self.vad_model(
                return_loss=False,
                rescale=True,
                **raw_data,
            )

        pts_bbox = results[0]["pts_bbox"]
        traj = _decode_traj_from_pts_bbox(pts_bbox)
        return traj
