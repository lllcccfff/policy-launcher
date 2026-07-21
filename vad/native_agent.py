from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from rl_framework.common.tensor import to_device
from rl_framework.common.trajectory import lidar_to_ego_front_left, traj2control
from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4
from rl_framework.vad.dataparser import parse_vad_obs

def _extract_pts_bbox(result_item: Any) -> Dict[str, Any]:
    return result_item["pts_bbox"]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _choose_mode_index(cmd: Any, num_modes: int) -> int:
    cmd_arr = _to_numpy(cmd).astype(np.float32).reshape(-1)
    cmd_idx = int(np.argmax(cmd_arr)) if cmd_arr.size > 1 else int(cmd_arr[0]) if cmd_arr.size else 2
    return int(np.clip(cmd_idx, 0, num_modes - 1))


def _decode_traj_from_pts_bbox(pts_bbox: Dict[str, Any]) -> np.ndarray:
    preds = _to_numpy(pts_bbox["ego_fut_preds"]).astype(np.float32)

    while preds.ndim > 3 and preds.shape[0] == 1:
        preds = preds[0]

    cmd = pts_bbox["ego_fut_cmd"]
    if preds.ndim == 3:
        mode_idx = _choose_mode_index(cmd, preds.shape[0])
        increments = preds[mode_idx]
    else:
        increments = preds

    traj = np.cumsum(increments, axis=0).astype(np.float32)
    return lidar_to_ego_front_left(traj, LIDAR_TO_EGO_4X4)


class VADNativeAgent:
    def __init__(
        self,
        vad_model: torch.nn.Module,
    ):
        self.vad_model = vad_model
        self.model_device = self._infer_model_device()

    def _infer_model_device(self) -> torch.device:
        try:
            model_impl = self.vad_model.module
        except AttributeError:
            model_impl = self.vad_model
        try:
            return next(model_impl.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_action(self, obs_img: Dict[str, np.ndarray], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        raw_data = parse_vad_obs(obs_img, obs_info)
        raw_data = to_device(raw_data, self.model_device)

        with torch.no_grad():
            results = self.vad_model(
                return_loss=False,
                rescale=True,
                feature_extractor=False,
                native_inference=True,
                **raw_data,
        )

        pts_bbox = _extract_pts_bbox(results[0])
        traj = _decode_traj_from_pts_bbox(pts_bbox)
        acc, steer = traj2control(traj, obs_info)
        return float(steer), float(acc)
