from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from rl_framework.latent_transfuser.dataparser import parse_latent_transfuser_obs
from rl_framework.latent_transfuser.loader import LatentTransFuserBundle


class LatentTransFuserNativeAgent:
    def __init__(self, bundle: LatentTransFuserBundle):
        if not bundle.nets:
            raise ValueError("Latent TransFuser bundle must contain at least one network")
        self.bundle = bundle
        self.nets = bundle.nets
        self.config = bundle.config
        self.device = bundle.device
        self.stuck_detector = 0
        self.forced_move = 0
        self.steer_damping = self.config.steer_damping

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        raw_data = parse_latent_transfuser_obs(
            obs_img,
            obs_info,
            self.config,
            self.device,
        )

        is_stuck = False
        if self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration:
            is_stuck = True
            self.forced_move += 1

        pred_wps = []
        with torch.inference_mode():
            for net in self.nets:
                pred_wp, _ = net.forward_ego(
                    raw_data["image"],
                    raw_data["lidar_bev"],
                    raw_data["target_point"],
                    raw_data["target_point_image"],
                    raw_data["velocity"],
                    num_points=None,
                )
                pred_wps.append(pred_wp)

        pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)
        waypoints = pred_wp[0].detach().cpu().numpy()
        pid_waypoints = waypoints.copy()
        pid_waypoints[:, 0] += self.config.lidar_pos[0]

        gt_velocity = float(raw_data["gt_velocity"].item())
        desired_speed = float(np.linalg.norm(pid_waypoints[0] - pid_waypoints[1]) * 2.0)
        if is_stuck:
            desired_speed = float(self.config.default_speed)

        brake = bool(
            (desired_speed < self.config.brake_speed)
            or ((gt_velocity / desired_speed) > self.config.brake_ratio)
        )
        delta = np.clip(desired_speed - gt_velocity, 0.0, self.config.clip_delta)
        throttle = float(np.clip(self.nets[0].speed_controller.step(delta), 0.0, self.config.clip_throttle))
        if brake:
            throttle = 0.0

        aim = (pid_waypoints[1] + pid_waypoints[0]) / 2.0
        angle = float(np.degrees(np.arctan2(aim[1], aim[0])) / 90.0)
        if gt_velocity < 0.01:
            angle = 0.0

        steer = float(np.clip(self.nets[0].turn_controller.step(angle), -1.0, 1.0))
        if is_stuck and self.forced_move == 1:
            steer = 0.0
        if brake or is_stuck:
            steer *= self.steer_damping

        if gt_velocity < 0.1:
            self.stuck_detector += 1
        elif gt_velocity > 0.1 and not is_stuck:
            self.stuck_detector = 0
            self.forced_move = 0

        metadrive_steer = -float(steer)
        throttle_brake = 0.0 if brake else float(throttle)
        return metadrive_steer, float(throttle_brake)
