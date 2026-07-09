from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from rl_framework.common.trajectory import lidar_to_ego_front_left, traj2control
from rl_framework.vad.dataparser import LIDAR_TO_EGO_4X4, get_vad_img_norm_cfg, parse_vad_obs

FIXED_CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT")


def _to_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
    if arr.size >= 3:
        return arr[:3].astype(np.float32)
    out = np.zeros(3, dtype=np.float32)
    out[: arr.size] = arr.astype(np.float32)
    return out


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
        cameras: Sequence[str] = FIXED_CAMERAS,
        img_norm_cfg: Dict[str, Any] | None = None,
    ):
        self.vad_model = vad_model
        self.cameras = tuple(cameras)
        self.img_norm_cfg = img_norm_cfg or get_vad_img_norm_cfg()
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

    def _to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.model_device)
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._to_device(v) for v in data)
        return data

    def _prepare_obs_info(self, obs_info: Dict[str, Any]) -> None:
        obs_info["scene_token"] = str(obs_info["scene_token"])
        obs_info["command"] = int(obs_info["command"])
        obs_info["linear_velocity"] = _to_vec3(obs_info["linear_velocity"])
        obs_info["linear_acceleration"] = _to_vec3(obs_info["linear_acceleration"])
        obs_info["accelerate"] = _to_vec3(obs_info["accelerate"])
        obs_info["angular_velocity"] = _to_vec3(obs_info["angular_velocity"])
        obs_info["ego_velo"] = float(obs_info["ego_velo"])
        obs_info["ego_steer"] = float(obs_info["ego_steer"])
        obs_info["cam_params"] = obs_info["cam_params"]

    def predict_traj(self, obs_img: Dict[str, np.ndarray], obs_info: Dict[str, Any]) -> np.ndarray:
        self._prepare_obs_info(obs_info)

        raw_data = parse_vad_obs(obs_img, obs_info, self.cameras, self.img_norm_cfg)
        raw_data.pop("raw_imgs", None)
        raw_data = self._to_device(raw_data)

        with torch.no_grad():
            results = self.vad_model(
                return_loss=False,
                rescale=True,
                feature_extractor=False,
                native_inference=True,
                **raw_data,
            )

        pts_bbox = _extract_pts_bbox(results[0])
        return _decode_traj_from_pts_bbox(pts_bbox)

    def predict_action(self, obs_img: Dict[str, np.ndarray], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        traj = self.predict_traj(obs_img, obs_info)
        acc, steer = traj2control(traj, obs_info)
        return float(steer), float(acc)
