from __future__ import annotations

import numpy as np


def metadrive_to_carla_xy(xy: np.ndarray) -> np.ndarray:
    return np.array((xy[0], -xy[1]), dtype=np.float32)


def metadrive_to_carla_yaw(yaw: float) -> float:
    return -float(yaw)


def rotation_matrix_2d(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )


def front_right_to_front_left(plan_traj: np.ndarray) -> np.ndarray:
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if plan_traj.ndim < 2 or plan_traj.shape[-1] < 2:
        raise ValueError(f"trajectory must have last dimension >= 2, got {plan_traj.shape}")
    return np.stack([plan_traj[..., 0], -plan_traj[..., 1]], axis=-1).astype(np.float32)
