#!/usr/bin/env python3
"""Run native ST-P3 on remote gRPC env and save one composed video per scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STP3_ROOT = str(REPO_ROOT / "ST-P3")
DEFAULT_STP3_CKPT = str(REPO_ROOT / "ST-P3" / "ckpts" / "stp3_carla.ckpt")
DEFAULT_OUTPUT_DIR = "./tmp_stp3"
PRED_TRAJ_HEIGHT_BELOW_EGO_M = 1.0
EGO_PLANE_SIZE_PX = 720
EGO_PLANE_HALF_RANGE_M = 20.0


def _sync_scene_token(obs_info: dict, step_info: dict) -> None:
    obs_info["scene_token"] = str(step_info["scene_name"])


def _safe_camera_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _terminal_reason(step_info: dict | None, terminated: bool, truncated: bool) -> str:
    if step_info is not None:
        reason = step_info.get("reason")
        if reason:
            return _safe_camera_name(str(reason))
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "max_rollout_steps"


def _is_ordered_scenario_exhausted(exc: BaseException) -> bool:
    message = str(exc)
    return "No more scenarios to evaluate" in message or "No completed scene metrics" in message


def _rename_with_terminal_reason(path: Path, terminal_reason: str) -> Path:
    final_path = path.with_name(f"{path.stem}__{terminal_reason}{path.suffix}")
    if final_path == path:
        return path
    if final_path.exists():
        final_path.unlink()
    path.rename(final_path)
    return final_path


def _to_k3(cam_info: dict) -> np.ndarray:
    k = cam_info["K"]
    if k.shape == (3, 3):
        return k
    if k.shape == (4, 4):
        return k[:3, :3]
    raise ValueError(f"`K` must be 3x3 or 4x4, got {k.shape}")


def _overlay_pred_traj_on_front_image(obs_img: dict, obs_info: dict, trajectory: np.ndarray | None) -> dict:
    if trajectory is None:
        return obs_img

    front_img = obs_img["FRONT"]
    base_img = front_img[-1] if front_img.ndim == 4 else front_img
    if base_img.ndim != 3 or base_img.shape[2] < 3:
        raise ValueError(f"FRONT image must have shape HxWx3 or TxHxWx3, got {front_img.shape}")

    cam_info = obs_info["cam_params"]["FRONT"]
    k3 = _to_k3(cam_info)
    ego2camera = cam_info["ego2camera"]
    if ego2camera.shape != (4, 4):
        raise ValueError(f"`ego2camera` must be 4x4, got {ego2camera.shape}")
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape Nx2, got {trajectory.shape}")

    x_forward = trajectory[:, 0]
    y_left = trajectory[:, 1]
    ego_points = np.stack(
        [x_forward, y_left, np.full_like(x_forward, -PRED_TRAJ_HEIGHT_BELOW_EGO_M)],
        axis=1,
    )
    camera_points = (ego2camera[:3, :3] @ ego_points.T + ego2camera[:3, 3:4]).T
    camera_points = camera_points[camera_points[:, 2] > 1e-6]

    projected = base_img.copy()
    if camera_points.size:
        pixels = (k3 @ (camera_points / camera_points[:, 2:3]).T).T
        h, w = base_img.shape[:2]
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        points_2d = pixels[valid, :2].astype(np.int32)
        for point in points_2d:
            cv2.circle(projected, tuple(point), radius=2, color=(64, 255, 64), thickness=-1)
        for start, end in zip(points_2d[:-1], points_2d[1:]):
            cv2.line(projected, tuple(start), tuple(end), color=(64, 255, 64), thickness=1)

    out_obs_img = dict(obs_img)
    if front_img.ndim == 4:
        updated = front_img.copy()
        updated[-1] = projected
        out_obs_img["FRONT"] = updated
    else:
        out_obs_img["FRONT"] = projected
    return out_obs_img


def _trajectory_to_world(trajectory: np.ndarray, obs_info: dict) -> np.ndarray:
    ego_xy = np.asarray(obs_info["ego_pos"], dtype=np.float32)[:2]
    heading = float(obs_info["heading_theta"])
    forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    left = np.array([-forward[1], forward[0]], dtype=np.float32)
    return ego_xy + trajectory[:, :1] * forward + trajectory[:, 1:2] * left


def _ego_plane_frame(obs_info: dict, trajectory: np.ndarray | None) -> np.ndarray:
    size = EGO_PLANE_SIZE_PX
    center = size // 2
    scale = center / EGO_PLANE_HALF_RANGE_M
    canvas = np.full((size, size, 3), 20, dtype=np.uint8)
    ego_xy = np.asarray(obs_info["ego_pos"], dtype=np.float32)[:2]
    heading = float(obs_info["heading_theta"])
    forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    left = np.array([-forward[1], forward[0]], dtype=np.float32)

    def project(world_xy: np.ndarray) -> tuple[int, int]:
        delta = world_xy[:2] - ego_xy
        return (
            int(round(center + delta[0] * scale)),
            int(round(center - delta[1] * scale)),
        )

    for meters in range(-int(EGO_PLANE_HALF_RANGE_M), int(EGO_PLANE_HALF_RANGE_M) + 1, 5):
        offset = int(round(meters * scale))
        cv2.line(canvas, (center + offset, 0), (center + offset, size - 1), (48, 48, 48), 1)
        cv2.line(canvas, (0, center - offset), (size - 1, center - offset), (48, 48, 48), 1)

    origin = (center, center)
    forward_end = project(ego_xy + forward * 5.0)
    left_end = project(ego_xy + left * 5.0)
    cv2.arrowedLine(canvas, origin, forward_end, (64, 255, 96), 3, cv2.LINE_AA, tipLength=0.16)
    cv2.arrowedLine(canvas, origin, left_end, (80, 220, 255), 3, cv2.LINE_AA, tipLength=0.16)
    cv2.putText(canvas, "F", forward_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 255, 96), 2, cv2.LINE_AA)
    cv2.putText(canvas, "L", left_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)

    target = np.asarray(obs_info["target_waypoint"], dtype=np.float32)
    target_pixel = project(target)
    if 0 <= target_pixel[0] < size and 0 <= target_pixel[1] < size:
        cv2.circle(canvas, target_pixel, 7, (255, 96, 255), 2, cv2.LINE_AA)

    if trajectory is not None:
        world_trajectory = _trajectory_to_world(trajectory, obs_info)
        pixels = np.asarray([project(point) for point in world_trajectory], dtype=np.int32)
        if len(pixels) > 1:
            cv2.polylines(canvas, [pixels], False, (255, 220, 64), 3, cv2.LINE_AA)
        for point in pixels:
            if 0 <= point[0] < size and 0 <= point[1] < size:
                cv2.circle(canvas, tuple(point), 4, (255, 220, 64), -1, cv2.LINE_AA)

    cv2.circle(canvas, origin, 8, (255, 80, 80), -1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-P3 on gRPC env and save videos to ./tmp_stp3")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step-size", type=float, default=0.05, help="Environment trajectory update period in seconds.")
    parser.add_argument("--timeout", type=float, default=360.0, help="gRPC timeout seconds")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output video directory")
    parser.add_argument("--device", type=str, default="cuda", help="ST-P3 inference device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.common.video_writer import EpisodeGridVideoWriter
    from rl_framework.stp3.loader import create_stp3
    from rl_framework.stp3.native_agent import STP3NativeAgent

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stp3_cfg = {
        "stp3_root": DEFAULT_STP3_ROOT,
        "checkpoint_path": DEFAULT_STP3_CKPT,
        "device": args.device,
    }
    stp3_model = create_stp3(stp3_cfg)
    agent = STP3NativeAgent(stp3_model)

    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)
    camera_layout = [
        ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        ["BACK"],
    ]

    try:
        first_scene = None
        scenes_completed = 0
        while True:
            try:
                obs, reset_info = env.reset()
            except Exception as exc:
                if scenes_completed > 0 and _is_ordered_scenario_exhausted(exc):
                    break
                raise
            scene_name = str(reset_info.get("scene_name", "scene"))
            if first_scene is None:
                first_scene = scene_name
            elif scene_name == first_scene:
                break

            agent.reset()
            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_scene_token(obs_info, reset_info)

            safe_scene = _safe_camera_name(scene_name)
            video_path = output_dir / f"{safe_scene}.mp4"
            stale_patterns = [
                f"{safe_scene}.mp4",
                f"{safe_scene}.avi",
                f"{safe_scene}__*.mp4",
                f"{safe_scene}__*.avi",
            ]
            for stale_path in {path for pattern in stale_patterns for path in output_dir.glob(pattern)}:
                if stale_path.exists():
                    stale_path.unlink()
            writer = EpisodeGridVideoWriter(
                video_path,
                step_size=args.step_size,
                camera_layout=camera_layout,
            )
            ego_plane_path = output_dir / f"{safe_scene}__ego_plane.mp4"
            ego_plane_writer = EpisodeGridVideoWriter(
                ego_plane_path,
                step_size=args.step_size,
                camera_layout=[["EGO_PLANE"]],
            )

            total_reward = 0.0
            steps = 0
            frame_count = 0
            terminated = False
            truncated = False
            last_step_info = dict(reset_info)

            try:
                for _ in range(args.max_steps):
                    trajectory = agent.predict_action(obs_img, obs_info)
                    writer.write(_overlay_pred_traj_on_front_image(obs_img, obs_info, trajectory))
                    ego_plane_writer.write({"EGO_PLANE": _ego_plane_frame(obs_info, trajectory)})
                    frame_count += 1
                    obs, reward, terminated, truncated, step_info = env.step(trajectory)
                    last_step_info = step_info
                    obs_img, obs_info, _, _ = unpack_ad_observation(obs)
                    _sync_scene_token(obs_info, step_info)

                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break
            finally:
                writer.close()
                video_path = writer.output_path
                ego_plane_writer.close()
                ego_plane_path = ego_plane_writer.output_path

            terminal_reason = _terminal_reason(last_step_info, terminated, truncated)
            video_path = _rename_with_terminal_reason(video_path, terminal_reason)
            ego_plane_path = _rename_with_terminal_reason(ego_plane_path, terminal_reason)
            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={terminal_reason} video={video_path} "
                f"ego_plane={ego_plane_path} codec={writer.codec} frames={frame_count}"
            )
            scenes_completed += 1

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
