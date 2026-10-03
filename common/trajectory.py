import numpy as np


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
