from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from rl_framework.common.trajectory import lidar_to_ego_front_left, traj2control
from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4, parse_uniad_obs

FIXED_CAMERAS = ("FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT")


def get_uniad_img_norm_cfg() -> Dict[str, np.ndarray | bool]:
    return {
        "mean": np.array([103.530, 116.280, 123.675], dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "to_rgb": False,
    }


def _to_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"expected vec3, got shape {np.asarray(value).shape}")
    return arr


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class UniADNativeAgent:
    def __init__(
        self,
        uniad_model: torch.nn.Module,
        cameras: Sequence[str] = FIXED_CAMERAS,
        img_norm_cfg: Dict[str, Any] | None = None,
    ):
        self.uniad_model = uniad_model
        self.cameras = tuple(cameras)
        self.img_norm_cfg = get_uniad_img_norm_cfg() if img_norm_cfg is None else img_norm_cfg
        self.model_device = self._infer_model_device()

    def _infer_model_device(self) -> torch.device:
        model_impl = self.uniad_model.module if hasattr(self.uniad_model, "module") else self.uniad_model
        return next(model_impl.parameters()).device

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

    def predict(
        self,
        obs_img: Dict[str, np.ndarray],
        obs_info: Dict[str, Any],
    ) -> Tuple[np.ndarray, Tuple[float, float], Dict[str, Any]]:
        obs_info["scene_token"] = str(obs_info["scene_token"])
        obs_info["ego_pos"] = _to_vec3(obs_info["ego_pos"])
        obs_info["ego_rot"] = _to_vec3(obs_info["ego_rot"])
        obs_info["linear_velocity"] = _to_vec3(obs_info["linear_velocity"])
        obs_info["linear_acceleration"] = _to_vec3(obs_info["linear_acceleration"])
        obs_info["angular_velocity"] = _to_vec3(obs_info["angular_velocity"])
        obs_info["ego_velo"] = float(obs_info["ego_velo"])
        obs_info["ego_steer"] = float(obs_info["ego_steer"])
        obs_info["relative_timestamp"] = float(obs_info["relative_timestamp"])

        raw_data = parse_uniad_obs(obs_img, obs_info, self.cameras, self.img_norm_cfg)
        raw_data.pop("raw_imgs")
        model_input = self._to_device(raw_data)

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
        action = (float(steer), float(acc))
        return pred_traj, action, raw_data
