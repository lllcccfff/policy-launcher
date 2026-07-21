from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np
import torch

from rl_framework.stp3.dataparser import parse_stp3_obs


class PIDController(object):
    def __init__(self, k_p: float = 1.0, k_i: float = 0.0, k_d: float = 0.0, n: int = 20):
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.window: Deque[float] = deque([0.0 for _ in range(n)], maxlen=n)

    def step(self, error: float) -> float:
        self.window.append(float(error))
        if len(self.window) >= 2:
            integral = float(np.mean(self.window))
            derivative = float(self.window[-1] - self.window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        return self.k_p * float(error) + self.k_i * integral + self.k_d * derivative


def _right_forward_to_front_left(plan_traj: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if plan_traj.ndim < 2 or plan_traj.shape[-1] < 2:
        raise ValueError(f"trajectory must have last dimension >= 2, got {plan_traj.shape}")
    return np.stack([plan_traj[..., 1], -plan_traj[..., 0]], axis=-1).astype(np.float32)


class STP3NativeAgent:
    def __init__(
        self,
        stp3_model: torch.nn.Module,
    ):
        self.model = stp3_model
        self.cfg = stp3_model.cfg
        self.device = self._infer_model_device()
        self.receptive_field = int(self.cfg.TIME_RECEPTIVE_FIELD)

        self.turn_controller = PIDController(k_p=1.25, k_i=0.75, k_d=0.3, n=40)
        self.speed_controller = PIDController(k_p=5.0, k_i=0.5, k_d=1.0, n=40)
        self.last_steer = 0.0
        self.last_pred_traj = None
        self._image_history: Deque[Dict[str, Any]] = deque(maxlen=self.receptive_field)
        self._state_history: Deque[Dict[str, Any]] = deque(maxlen=self.receptive_field)

    def _infer_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self) -> None:
        self.turn_controller = PIDController(k_p=1.25, k_i=0.75, k_d=0.3, n=40)
        self.speed_controller = PIDController(k_p=5.0, k_i=0.5, k_d=1.0, n=40)
        self.last_steer = 0.0
        self.last_pred_traj = None
        self._image_history.clear()
        self._state_history.clear()

    def _control_pid(
        self,
        final_traj: np.ndarray,
        velocity: float,
    ) -> Tuple[float, float, bool]:
        waypoints = np.asarray(final_traj, dtype=np.float32)[:, :2]
        speed = float(velocity)

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2.0 - np.arctan2(aim[1], aim[0])) / 90.0
        steer = float(np.clip(self.turn_controller.step(float(angle)), -1.0, 1.0))

        desired_speed = float(np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0)
        brake = bool((speed / desired_speed) > 1.2)

        delta = float(np.clip(desired_speed - speed, 0.0, 0.25))
        throttle = float(np.clip(self.speed_controller.step(delta), 0.0, 0.75))
        if brake:
            throttle = 0.0

        return steer, throttle, brake

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        self._image_history.append(dict(obs_img))
        self._state_history.append(
            {
                "ego_pos": np.asarray(obs_info["ego_pos"], dtype=np.float32).copy(),
                "heading_theta": float(obs_info["heading_theta"]),
            }
        )
        if len(self._image_history) < self.receptive_field:
            self.last_pred_traj = None
            return 0.0, 0.0

        raw_data = parse_stp3_obs(
            list(self._image_history),
            list(self._state_history),
            obs_info,
            self.cfg,
            self.device,
            self.last_steer,
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
        self.last_pred_traj = _right_forward_to_front_left(final_traj[:, :2])
        carla_steer, throttle, brake = self._control_pid(final_traj, float(obs_info["ego_velo"]))
        self.last_steer = float(carla_steer)
        return float(carla_steer), -1.0 if brake else float(throttle)
