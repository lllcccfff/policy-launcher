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
        target_point_index: int = 4,
        route_planner_min_distance: float = 1.0,
        route_planner_max_distance: float = 50.0,
    ):
        self.model = stp3_model
        self.cfg = stp3_model.cfg
        self.device = self._infer_model_device()
        self.receptive_field = int(self.cfg.TIME_RECEPTIVE_FIELD)
        self.target_point_index = int(target_point_index)
        self.route_planner_min_distance = float(route_planner_min_distance)
        self.route_planner_max_distance = float(route_planner_max_distance)

        self.turn_controller = PIDController(k_p=1.25, k_i=0.75, k_d=0.3, n=40)
        self.speed_controller = PIDController(k_p=5.0, k_i=0.5, k_d=1.0, n=40)
        self.last_steer = 0.0
        self.last_pred_traj = None
        self.last_debug: Dict[str, Any] = {}
        self._route_scene_token = None
        self._route_pop_idx = 0
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
        self.last_debug = {}
        self._route_scene_token = None
        self._route_pop_idx = 0
        self._image_history.clear()
        self._state_history.clear()

    def _sync_route_progress(self, obs_info: Dict[str, Any]) -> int:
        scene_token = str(obs_info["scene_token"])
        if self._route_scene_token != scene_token:
            self._route_scene_token = scene_token
            self._route_pop_idx = 0

        route = np.asarray(obs_info["route_waypoints"], dtype=np.float32)
        if route.ndim != 2 or route.shape[1] < 2:
            raise ValueError(f"route_waypoints must have shape Nx2 or NxM, got {route.shape}")
        if route.shape[0] < 2:
            raise ValueError("route_waypoints must contain at least two points")

        ego_xy = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)[:2]
        route_tail = route[self._route_pop_idx :, :2]
        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(route_tail)):
            if cumulative_distance > self.route_planner_max_distance:
                break
            cumulative_distance += float(np.linalg.norm(route_tail[i] - route_tail[i - 1]))
            distance = float(np.linalg.norm(route_tail[i] - ego_xy))
            if distance <= self.route_planner_min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        max_pop = max(0, route.shape[0] - self._route_pop_idx - 2)
        self._route_pop_idx += min(to_pop, max_pop)
        target_route_idx = min(self._route_pop_idx + self.target_point_index, route.shape[0] - 1)
        obs_info["stp3_route_pop_idx"] = int(self._route_pop_idx)
        obs_info["stp3_target_route_idx"] = int(target_route_idx)
        return int(target_route_idx)

    def _prepare_obs_info(self, obs_info: Dict[str, Any]) -> None:
        obs_info["scene_token"] = str(obs_info["scene_token"])
        obs_info["command"] = int(obs_info["command"])
        obs_info["ego_pos"] = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)
        obs_info["heading_theta"] = float(obs_info["heading_theta"])
        obs_info["ego_velo"] = float(obs_info["ego_velo"])
        obs_info["ego_steer"] = float(obs_info["ego_steer"])

    def _append_history(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> None:
        self._image_history.append(dict(obs_img))
        self._state_history.append(
            {
                "ego_pos": np.asarray(obs_info["ego_pos"], dtype=np.float32).copy(),
                "heading_theta": float(obs_info["heading_theta"]),
            }
        )

    def is_ready(self) -> bool:
        return len(self._image_history) == self.receptive_field and len(self._state_history) == self.receptive_field

    def prepare_raw_input(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Dict[str, Any]:
        self._prepare_obs_info(obs_info)
        target_route_idx = self._sync_route_progress(obs_info)
        self._append_history(obs_img, obs_info)
        if not self.is_ready():
            raise RuntimeError(
                f"ST-P3 needs {self.receptive_field} history frames, "
                f"got {len(self._image_history)}"
            )
        return parse_stp3_obs(
            list(self._image_history),
            list(self._state_history),
            obs_info,
            self.cfg,
            self.device,
            target_route_idx,
            self.last_steer,
        )

    def predict_traj_from_raw_data(self, raw_data: Dict[str, Any]) -> np.ndarray:
        with torch.no_grad():
            output = self.model(
                raw_data["image"],
                raw_data["intrinsics"],
                raw_data["extrinsics"],
                raw_data["future_egomotion"],
            )

            n_present = int(self.model.receptive_field)
            seg_prediction = torch.argmax(output["segmentation"].detach(), dim=2, keepdim=True)
            pedestrian_prediction = torch.argmax(output["pedestrian"].detach(), dim=2, keepdim=True)
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)

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

        final_traj_np = final_traj[0].detach().cpu().numpy().astype(np.float32)
        self.last_pred_traj = _right_forward_to_front_left(final_traj_np[:, :2])
        return final_traj_np

    def _control_pid(
        self,
        final_traj: np.ndarray,
        velocity: float,
        command: str,
    ) -> Tuple[float, float, bool, Dict[str, Any]]:
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

        debug = {
            "speed": speed,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "desired_speed": desired_speed,
            "angle": float(angle),
            "aim": aim.astype(float).tolist(),
            "delta": delta,
            "command": command,
            "pred_traj_front_left": self.last_pred_traj.astype(float).tolist() if self.last_pred_traj is not None else None,
        }
        return steer, throttle, brake, debug

    def predict_action_from_raw_data(self, raw_data: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        final_traj = self.predict_traj_from_raw_data(raw_data)
        carla_steer, throttle, brake, debug = self._control_pid(
            final_traj,
            float(obs_info["ego_velo"]),
            str(raw_data["command"][0]),
        )
        self.last_steer = float(carla_steer)

        brake_value = 1.0 if brake else 0.0
        if throttle > brake_value:
            brake_value = 0.0

        metadrive_steer = float(carla_steer)
        throttle_brake = -float(brake_value) if brake_value > 0.0 else float(throttle)
        debug["metadrive_steer"] = metadrive_steer
        debug["throttle_brake"] = throttle_brake
        self.last_debug = debug
        return metadrive_steer, throttle_brake

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        self._prepare_obs_info(obs_info)
        self._sync_route_progress(obs_info)
        self._append_history(obs_img, obs_info)
        if not self.is_ready():
            self.last_pred_traj = None
            self.last_debug = {"warmup_frames": len(self._image_history), "required_frames": self.receptive_field}
            return 0.0, 0.0

        raw_data = parse_stp3_obs(
            list(self._image_history),
            list(self._state_history),
            obs_info,
            self.cfg,
            self.device,
            int(obs_info["stp3_target_route_idx"]),
            self.last_steer,
        )
        return self.predict_action_from_raw_data(raw_data, obs_info)
