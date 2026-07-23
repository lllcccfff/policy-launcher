from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np
import torch

from rl_framework.common.coordinates import right_forward_to_front_left
from rl_framework.stp3.dataparser import parse_stp3_obs

MAX_STEER_RAD = np.deg2rad(40.0)


class STP3NativeAgent:
    def __init__(
        self,
        stp3_model: torch.nn.Module,
    ):
        self.model = stp3_model
        self.cfg = stp3_model.cfg
        self.device = self._infer_model_device()
        self.receptive_field = int(self.cfg.TIME_RECEPTIVE_FIELD)

        self._image_history: Deque[Dict[str, Any]] = deque(maxlen=self.receptive_field)
        self._state_history: Deque[Dict[str, Any]] = deque(maxlen=self.receptive_field)
        self._frames_seen = 0

    def _infer_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self) -> None:
        self._image_history.clear()
        self._state_history.clear()
        self._frames_seen = 0

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Optional[np.ndarray]:
        self._image_history.append(dict(obs_img))
        self._frames_seen += 1
        self._state_history.append(
            {
                "ego_pos": np.asarray(obs_info["ego_pos"], dtype=np.float32).copy(),
                "heading_theta": float(obs_info["heading_theta"]),
            }
        )
        if self._frames_seen <= self.receptive_field + 1:
            return None

        raw_data = parse_stp3_obs(
            list(self._image_history),
            list(self._state_history),
            obs_info,
            self.cfg,
            self.device,
            float(obs_info["ego_steer"]) / MAX_STEER_RAD,
        )
        with torch.no_grad():
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
        return right_forward_to_front_left(final_traj[:, :2])
