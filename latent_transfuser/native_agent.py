from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from rl_framework.latent_transfuser.dataparser import parse_latent_transfuser_obs
from rl_framework.latent_transfuser.loader import LatentTransFuserBundle


class LatentTransFuserNativeAgent:
    def __init__(self, bundle: LatentTransFuserBundle, target_point_index: int = 4):
        if not bundle.nets:
            raise ValueError("Latent TransFuser bundle must contain at least one network")
        self.bundle = bundle
        self.nets = bundle.nets
        self.config = bundle.config
        self.device = bundle.device
        self.target_point_index = target_point_index
        self.stuck_detector = 0
        self.forced_move = 0
        self.steer_damping = self.config.steer_damping
        self.last_debug: Dict[str, Any] = {}
        self._route_scene_token: str | None = None
        self._route_pop_idx = 0

    @staticmethod
    def _route_xy(obs_info: Dict[str, Any]) -> np.ndarray:
        route = np.asarray(obs_info["route_waypoints"], dtype=np.float32)
        if route.ndim != 2 or route.shape[1] < 2:
            raise ValueError(f"route_waypoints must have shape Nx2/Nx3, got {route.shape}")
        route = route[:, :2].astype(np.float32, copy=False)
        if route.shape[0] < 2:
            raise ValueError("route_waypoints must contain at least two points")
        return route

    @staticmethod
    def _transfuser_downsample_route_xy(route: np.ndarray, sample_distance: float) -> np.ndarray:
        """
        XY-only equivalent of TransFuser's leaderboard downsample_route(route, 50).

        The original function expects (carla.Transform, RoadOption) route entries and
        preserves RoadOption changes. Navigation obs only exposes xy points, so the
        distance-based branch is the part we can apply here.
        """
        if route.shape[0] <= 2:
            return route.copy()

        ids_to_sample = [0]
        dist = 0.0
        for i in range(1, route.shape[0]):
            if dist > sample_distance:
                ids_to_sample.append(i)
                dist = 0.0
            elif i == route.shape[0] - 1:
                ids_to_sample.append(i)
                dist = 0.0
            else:
                dist += float(np.linalg.norm(route[i] - route[i - 1]))

        if ids_to_sample[-1] != route.shape[0] - 1:
            ids_to_sample.append(route.shape[0] - 1)
        return route[np.asarray(ids_to_sample, dtype=np.int64)].copy()

    def _prepare_route(self, obs_info: Dict[str, Any]) -> np.ndarray:
        raw_route = self._route_xy(obs_info)
        sample_distance = float(getattr(self.config, "route_downsample_distance", self.config.route_planner_max_distance))
        route = self._transfuser_downsample_route_xy(raw_route, sample_distance)
        obs_info["latent_transfuser_raw_route_len"] = int(raw_route.shape[0])
        obs_info["latent_transfuser_downsampled_route_len"] = int(route.shape[0])
        obs_info["latent_transfuser_route_sample_distance"] = sample_distance
        obs_info["latent_transfuser_route_waypoints"] = route
        return route

    def _sync_route_progress(self, obs_info: Dict[str, Any]) -> None:
        scene_token = str(obs_info["scene_token"])
        if self._route_scene_token != scene_token:
            self._route_scene_token = scene_token
            self._route_pop_idx = 0

        route = self._prepare_route(obs_info)
        ego_xy = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)[:2]
        min_distance = float(self.config.route_planner_min_distance)
        max_distance = float(self.config.route_planner_max_distance)

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        route_tail = route[self._route_pop_idx :]
        for i in range(1, len(route_tail)):
            if cumulative_distance > max_distance:
                break
            cumulative_distance += float(np.linalg.norm(route_tail[i] - route_tail[i - 1]))
            distance = float(np.linalg.norm(route_tail[i] - ego_xy))
            if distance <= min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        max_pop = max(0, route.shape[0] - self._route_pop_idx - 2)
        self._route_pop_idx += min(to_pop, max_pop)
        obs_info["latent_transfuser_route_pop_idx"] = int(self._route_pop_idx)
        obs_info["latent_transfuser_target_route_idx"] = min(self._route_pop_idx + 1, route.shape[0] - 1)

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        obs_info["ego_velo"] = float(obs_info["ego_velo"])
        obs_info["ego_pos"] = np.asarray(obs_info["ego_pos"], dtype=np.float32)
        obs_info["ego_rot"] = np.asarray(obs_info["ego_rot"], dtype=np.float32)
        self._sync_route_progress(obs_info)
        raw_data = parse_latent_transfuser_obs(
            obs_img,
            obs_info,
            self.config,
            self.device,
            target_point_index=self.target_point_index,
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
        steer_before_damping = steer
        if is_stuck and self.forced_move == 1:
            steer = 0.0
        if brake or is_stuck:
            steer *= self.steer_damping

        debug_stuck_detector = int(self.stuck_detector)
        debug_forced_move = int(self.forced_move)
        if gt_velocity < 0.1:
            self.stuck_detector += 1
        elif gt_velocity > 0.1 and not is_stuck:
            self.stuck_detector = 0
            self.forced_move = 0

        metadrive_steer = -float(steer)
        throttle_brake = 0.0 if brake else float(throttle)
        self.last_debug = {
            "target_point": raw_data["target_point"][0].detach().cpu().numpy().astype(float).tolist(),
            "gt_velocity": gt_velocity,
            "pred_wp": waypoints.astype(float).tolist(),
            "pid_waypoints": pid_waypoints.astype(float).tolist(),
            "desired_speed": desired_speed,
            "brake": brake,
            "throttle": float(throttle),
            "aim": aim.astype(float).tolist(),
            "pid_angle": angle,
            "steer_before_damping": steer_before_damping,
            "is_stuck": bool(is_stuck),
            "stuck_detector": debug_stuck_detector,
            "forced_move": debug_forced_move,
            "carla_steer": float(steer),
            "steer": metadrive_steer,
            "throttle_brake": float(throttle_brake),
            "raw_route_len": int(obs_info["latent_transfuser_raw_route_len"]),
            "downsampled_route_len": int(obs_info["latent_transfuser_downsampled_route_len"]),
            "route_sample_distance": float(obs_info["latent_transfuser_route_sample_distance"]),
            "route_pop_idx": int(obs_info["latent_transfuser_route_pop_idx"]),
            "target_route_idx": int(obs_info["latent_transfuser_target_route_idx"]),
        }
        return metadrive_steer, float(throttle_brake)
