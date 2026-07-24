from collections import deque

import numpy as np

from rl_framework.common.coordinates import front_right_to_front_left, right_forward_to_front_left

MAX_STEER_RAD = np.deg2rad(40.0)
WHEELBASE = 2.469


def smooth_1d(arr, kernel_size=5):
    if kernel_size == 5:
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    else:
        kernel = np.ones(kernel_size, dtype=np.float32)
    kernel = kernel / kernel.sum()

    pad = kernel_size // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    smoothed = np.convolve(arr_pad, kernel, mode='valid')
    return smoothed.astype(np.float32)


def lidar_to_ego_front_left(plan_traj, lidar_to_ego):
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if plan_traj.ndim < 2 or plan_traj.shape[-1] < 2:
        raise ValueError(f"trajectory must have last dimension >= 2, got {plan_traj.shape}")

    lidar_to_ego = np.asarray(lidar_to_ego, dtype=np.float32)
    if lidar_to_ego.shape != (4, 4):
        raise ValueError(f"lidar_to_ego must be 4x4, got {lidar_to_ego.shape}")

    lidar_points = np.zeros(plan_traj.shape[:-1] + (4,), dtype=np.float32)
    lidar_points[..., :2] = plan_traj[..., :2]
    lidar_points[..., 3] = 1.0
    ego_points = lidar_points @ lidar_to_ego.T
    return ego_points[..., :2].astype(np.float32)

class PID:
    def __init__(self, k_p: float, k_i: float, k_d: float, window_size: int):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window = deque([0.0] * window_size, maxlen=window_size)

    def step(self, error: float) -> float:
        self.window.append(error)
        integral = float(np.mean(self.window))
        derivative = self.window[-1] - self.window[-2]
        return self.k_p * error + self.k_i * integral + self.k_d * derivative
