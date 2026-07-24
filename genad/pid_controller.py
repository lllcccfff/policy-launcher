from __future__ import annotations

import numpy as np

from rl_framework.common.trajectory import PID

class Bench2DrivePIDController:
    def __init__(self):
        self.turn_controller = PID(k_p=0.75, k_i=0.75, k_d=0.3, window_size=40)
        self.speed_controller = PID(k_p=5.0, k_i=0.5, k_d=1.0, window_size=40)
        self.max_throttle = 0.75
        self.brake_speed = 0.4
        self.brake_ratio = 1.1
        self.clip_delta = 0.25
        self.aim_dist = 4.0
        self.angle_thresh = 0.3
        self.dist_thresh = 10.0

    def control_pid(
        self,
        waypoints: np.ndarray,
        speed: np.float32,
        target: np.ndarray,
    ) -> tuple[float, float, bool]:
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0.0
        aim = waypoints[0]

        for index in range(num_pairs):
            desired_speed += np.linalg.norm(waypoints[index + 1] - waypoints[index]) * 2.0 / num_pairs
            norm = np.linalg.norm((waypoints[index + 1] + waypoints[index]) / 2.0)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = waypoints[index]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]
        # Geometry is ego [forward, left], matching MetaDrive's positive-left steering.
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        angle_last = np.degrees(np.arctan2(aim_last[1], aim_last[0])) / 90.0
        angle_target = np.degrees(np.arctan2(target[1], target[0])) / 90.0

        use_target = abs(angle_target) < abs(angle)
        use_target = use_target or (
            abs(angle_target - angle_last) > self.angle_thresh and target[0] < self.dist_thresh
        )
        angle_final = angle_target if use_target else angle

        steer = float(np.clip(self.turn_controller.step(float(angle_final)), -1.0, 1.0))
        brake = bool(desired_speed < self.brake_speed or speed / desired_speed > self.brake_ratio)
        delta = float(np.clip(desired_speed - speed, 0.0, self.clip_delta))
        throttle = float(np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0
        return steer, throttle, brake
