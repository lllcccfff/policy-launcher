from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
import torch

from policy_launcher.common.trajectory import lidar_to_ego_front_left
from policy_launcher.stp3.dataparser import LIDAR_TOP_TO_EGO, lidar_to_world, parse_stp3_obs


class STP3NativeAgent:
    def __init__(self, device: str):
        from policy_launcher.stp3.loader import create_stp3

        repo_root = Path(__file__).resolve().parents[2]
        self.model = create_stp3(
            {
                "stp3_root": str(repo_root / "ST-P3"),
                "checkpoint_path": str(repo_root / "ST-P3" / "ckpts" / "stp3_nuscenes.ckpt"),
                "device": device,
            }
        )
        self.cfg = self.model.cfg
        self.device = self._infer_model_device()
        self.receptive_field = int(self.cfg.TIME_RECEPTIVE_FIELD)

        self._image_history: Deque[Dict[str, Any]] = deque(maxlen=self.receptive_field)
        self._lidar_to_world_history: Deque[np.ndarray] = deque(maxlen=self.receptive_field)
        self._scene_token: str | None = None

    def _infer_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self) -> None:
        self._image_history.clear()
        self._lidar_to_world_history.clear()
        self._scene_token = None

    def _reset_for_scene(self, scene_token: str) -> None:
        if scene_token != self._scene_token:
            self.reset()
            self._scene_token = scene_token

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Optional[np.ndarray]:
        self._reset_for_scene(str(obs_info["scene_token"]))
        self._image_history.append(dict(obs_img))
        self._lidar_to_world_history.append(lidar_to_world(obs_info))
        if len(self._image_history) < self.receptive_field:
            return None

        raw_data = parse_stp3_obs(
            list(self._image_history),
            list(self._lidar_to_world_history),
            obs_info,
            self.cfg,
            self.device,
        )
        with torch.inference_mode():
            output = self.model(
                raw_data["image"],
                raw_data["intrinsics"],
                raw_data["extrinsics"],
                raw_data["future_egomotion"],
            )

            n_present = int(self.model.receptive_field)
            segmentation = torch.argmax(output["segmentation"].detach(), dim=2, keepdim=True)
            pedestrian = torch.argmax(output["pedestrian"].detach(), dim=2, keepdim=True)
            occupancy = torch.logical_or(segmentation, pedestrian)

            _, final_traj = self.model.planning(
                cam_front=output["cam_front"].detach(),
                trajs=raw_data["sample_trajectory"][:, :, 1:],
                gt_trajs=None,
                cost_volume=output["costvolume"][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output["hdmap"].detach(),
                commands=raw_data["command"],
                target_points=raw_data["target_point"],
            )

        final_traj = final_traj[0].detach().cpu().numpy()
        return lidar_to_ego_front_left(final_traj, LIDAR_TOP_TO_EGO)
