import numpy as np
from rl_framework.ilqr.lqr import plan2control

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


def right_forward_to_front_left(plan_traj):
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if plan_traj.ndim < 2 or plan_traj.shape[-1] < 2:
        raise ValueError(f"trajectory must have last dimension >= 2, got {plan_traj.shape}")
    return np.stack([plan_traj[..., 1], -plan_traj[..., 0]], axis=-1).astype(np.float32)


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


def front_right_to_front_left(plan_traj):
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if plan_traj.ndim < 2 or plan_traj.shape[-1] < 2:
        raise ValueError(f"trajectory must have last dimension >= 2, got {plan_traj.shape}")
    return np.stack([plan_traj[..., 0], -plan_traj[..., 1]], axis=-1).astype(np.float32)


def traj2control(plan_traj, info, horizon=3.0):
    """Convert (x_front, y_left) relative trajectory to ego-frame control targets."""
    horizon = float(horizon)
    if horizon <= 0.0:
        raise ValueError(f"trajectory horizon must be positive, got {horizon}")
    control_dt = 0.1

    N = plan_traj.shape[0]
    dt_in = horizon / N
    t_orig = np.linspace(dt_in, horizon, N, dtype=np.float32)
    t_ref = np.concatenate(([0.0], t_orig))

    x_forward_orig = plan_traj[:, 0]
    y_left_orig = plan_traj[:, 1]

    x_forward_ref = np.concatenate(([0.0], x_forward_orig))
    y_left_ref = np.concatenate(([0.0], y_left_orig))

    x_forward_ref_smooth = smooth_1d(x_forward_ref, kernel_size=5)
    y_left_ref_smooth = smooth_1d(y_left_ref, kernel_size=5)

    t_new = np.arange(control_dt, horizon + 1e-6, control_dt, dtype=np.float32)
    M = t_new.shape[0]

    x_forward_new = np.interp(t_new, t_ref, x_forward_ref_smooth)
    y_left_new = np.interp(t_new, t_ref, y_left_ref_smooth)
    plan_traj_ego = np.stack([x_forward_new, y_left_new], axis=1)

    plan_traj_stats = np.zeros((M + 1, 5), dtype=np.float32)
    plan_traj_stats[1:, :2] = plan_traj_ego

    headings = np.zeros(M, dtype=np.float32)
    for i in range(M):
        if i == 0:
            delta_vec = plan_traj_ego[min(1, M - 1)] - plan_traj_ego[0]
        elif i == M - 1:
            delta_vec = plan_traj_ego[M - 1] - plan_traj_ego[M - 2]
        else:
            delta_vec = plan_traj_ego[i + 1] - plan_traj_ego[i - 1]
        headings[i] = np.arctan2(delta_vec[1], delta_vec[0])

    plan_traj_stats[1:, 2] = headings

    # Desired speed profile derived from consecutive waypoints
    delta_pos = np.diff(plan_traj_ego, axis=0)
    segment_dist = np.linalg.norm(delta_pos, axis=1)
    segment_speed = segment_dist / control_dt
    ref_speed = np.zeros(M, dtype=np.float32)
    if M > 0:
        ref_speed[0] = max(info['ego_velo'], segment_speed[0] if segment_speed.size > 0 else info['ego_velo'])
    if segment_speed.size > 0:
        ref_speed[1:] = segment_speed[:M - 1]
        ref_speed = smooth_1d(ref_speed, kernel_size=5)
    plan_traj_stats[1:, 3] = ref_speed

    ego_steer_rad = float(info['ego_steer'])
    max_steer_rad = MAX_STEER_RAD
    ego_steer_rad = float(np.clip(ego_steer_rad, -max_steer_rad, max_steer_rad))

    curvature = np.zeros(M, dtype=np.float32)
    eps = 1e-3
    for i in range(1, M):
        ds = max(segment_dist[i - 1], eps)
        dtheta = headings[i] - headings[i - 1]
        curvature[i] = dtheta / ds

    steer_profile = np.arctan(np.clip(curvature * WHEELBASE, -np.tan(max_steer_rad), np.tan(max_steer_rad)))
    if M > 1:
        steer_profile = smooth_1d(steer_profile, kernel_size=5)
    steer_profile = np.clip(steer_profile, -max_steer_rad, max_steer_rad)
    plan_traj_stats[1:, 4] = steer_profile

    plan_traj_stats[0, 3] = info['ego_velo']
    plan_traj_stats[0, 4] = ego_steer_rad

    curr_stat = np.array(
        [0.0, 0.0, 0.0, info['ego_velo'], ego_steer_rad],
        dtype=np.float32
    )
    acc, steer_rate = plan2control(plan_traj_stats, curr_stat)
    return acc, steer_rate
