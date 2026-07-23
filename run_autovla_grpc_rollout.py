#!/usr/bin/env python3
"""Run AutoVLA on remote StreetWorld gRPC env and save one video per scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTOVLA_ROOT = str(REPO_ROOT / "AutoVLA")
DEFAULT_AUTOVLA_CONFIG = str(REPO_ROOT / "AutoVLA" / "config" / "eval" / "qwen2.5-vl-3B-nusc-sft-eval.yaml")
DEFAULT_AUTOVLA_CKPT = str(REPO_ROOT / "AutoVLA" / "checkpoints" / "AutoVLA" / "AutoVLA_PDMS_89.ckpt")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "outputs" / "autovla")
PRED_TRAJ_HEIGHT_BELOW_EGO_M = 1.0


def _sync_runtime_fields(obs_info: dict, step_info: dict) -> None:
    obs_info["scene_token"] = str(step_info["scene_name"])
    obs_info["relative_timestamp"] = float(step_info["relative_timestamp"]) * 1e-6
    obs_info["timestamp"] = float(step_info["relative_timestamp"]) * 1e-6
    obs_info["current_timestamp"] = float(step_info["current_timestamp"]) * 1e-6


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


def _debug_state(obs_info: dict) -> dict:
    keys = (
        "ego_pos",
        "ego_rot",
        "linear_velocity",
        "linear_acceleration",
        "angular_velocity",
        "ego_velo",
        "ego_steer",
        "relative_timestamp",
        "current_timestamp",
        "command",
    )
    state = {}
    for key in keys:
        value = obs_info[key]
        if isinstance(value, np.ndarray):
            state[key] = value.astype(float).tolist()
        elif isinstance(value, (list, tuple)):
            state[key] = np.asarray(value, dtype=float).tolist()
        else:
            state[key] = float(value) if isinstance(value, (np.floating, float)) else int(value)
    return state


def _to_k3(cam_info: dict) -> np.ndarray:
    k = np.asarray(cam_info["K"], dtype=np.float32)
    if k.shape == (3, 3):
        return k
    if k.shape == (4, 4):
        return k[:3, :3]
    raise ValueError(f"`K` must be 3x3 or 4x4, got {k.shape}")


def _overlay_pred_traj_on_front_image(obs_img: dict, obs_info: dict, pred_traj: np.ndarray | None) -> dict:
    if pred_traj is None:
        return obs_img
    if "FRONT" not in obs_img:
        return obs_img
    if "cam_params" not in obs_info or "FRONT" not in obs_info["cam_params"]:
        return obs_img

    front_img = np.asarray(obs_img["FRONT"])
    is_stacked = front_img.ndim == 4
    base_img = front_img[-1] if is_stacked else front_img
    if base_img.ndim != 3 or base_img.shape[2] < 3:
        return obs_img

    cam_info = obs_info["cam_params"]["FRONT"]
    k3 = _to_k3(cam_info)
    ego2camera = np.asarray(cam_info["ego2camera"], dtype=np.float32)
    if ego2camera.shape != (4, 4):
        raise ValueError(f"`ego2camera` must be 4x4, got {ego2camera.shape}")

    traj = np.asarray(pred_traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Unexpected predicted trajectory shape: {traj.shape}")

    x_forward = traj[:, 0]
    y_left = traj[:, 1]
    z_up = np.full_like(x_forward, -PRED_TRAJ_HEIGHT_BELOW_EGO_M)
    ego_points = np.stack([x_forward, y_left, z_up], axis=1)

    h, w = int(base_img.shape[0]), int(base_img.shape[1])
    camera_points = (ego2camera[:3, :3] @ ego_points.T + ego2camera[:3, 3:4]).T
    camera_points = camera_points[camera_points[:, 2] > 1e-6]

    projected = np.array(base_img, copy=True)
    if camera_points.size:
        normalized = camera_points / camera_points[:, 2:3]
        pixels = (k3 @ normalized.T).T
        u = pixels[:, 0]
        v = pixels[:, 1]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        points_2d = np.stack([u[valid], v[valid]], axis=1).astype(np.int32)
        for point in points_2d:
            cv2.circle(projected, tuple(point), radius=2, color=(48, 220, 96), thickness=-1)
        for start, end in zip(points_2d[:-1], points_2d[1:]):
            cv2.line(projected, tuple(start), tuple(end), color=(48, 220, 96), thickness=1)

    out_obs_img = dict(obs_img)
    if is_stacked:
        updated = np.array(front_img, copy=True)
        updated[-1] = projected.astype(updated.dtype, copy=False)
        out_obs_img["FRONT"] = updated
    else:
        out_obs_img["FRONT"] = projected.astype(front_img.dtype, copy=False)
    return out_obs_img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoVLA on StreetWorld gRPC env and save videos")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step-size", type=float, default=0.5, help="Environment trajectory update period in seconds")
    parser.add_argument("--timeout", type=float, default=360.0, help="gRPC timeout seconds")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output video directory")
    parser.add_argument("--scene-name", type=str, default=None, help="Run only this scene name")
    parser.add_argument("--autovla-root", type=str, default=DEFAULT_AUTOVLA_ROOT, help="AutoVLA checkout root")
    parser.add_argument("--config-path", type=str, default=DEFAULT_AUTOVLA_CONFIG, help="AutoVLA config path")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_AUTOVLA_CKPT, help="AutoVLA checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0", help="AutoVLA inference device")
    parser.add_argument("--history-frames", type=int, default=4, help="AutoVLA video history length")
    parser.add_argument("--frame-interval", type=float, default=0.5, help="Seconds between cached AutoVLA video frames")
    parser.add_argument("--max-length", type=int, default=None, help="Override AutoVLA generation max_length")
    parser.add_argument("--temperature", type=float, default=None, help="Override AutoVLA generation temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Override AutoVLA generation top_k")
    parser.add_argument("--top-p", type=float, default=None, help="Override AutoVLA generation top_p")
    parser.add_argument("--debug-jsonl", type=str, default="", help="Optional per-step AutoVLA debug JSONL path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.autovla import AutoVLANativeAgent, create_autovla
    from rl_framework.common.video_writer import EpisodeGridVideoWriter

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_file = None
    if args.debug_jsonl:
        debug_path = Path(args.debug_jsonl)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path.open("w", encoding="utf-8")

    bundle = create_autovla(
        {
            "autovla_root": args.autovla_root,
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
            "device": args.device,
        }
    )
    agent = AutoVLANativeAgent(
        bundle,
        history_frames=args.history_frames,
        frame_interval=args.frame_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length,
    )

    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)
    camera_layout = [
        ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        ["BACK_LEFT", "BACK", "BACK_RIGHT"],
    ]

    try:
        seen_scenes: set[str] = set()
        scenes_completed = 0
        while True:
            try:
                obs, reset_info = env.reset()
            except Exception as exc:
                if _is_ordered_scenario_exhausted(exc):
                    if args.scene_name is not None and scenes_completed == 0:
                        raise RuntimeError(f"Scene {args.scene_name!r} was not found before ordered scenario exhaustion.") from exc
                    if scenes_completed > 0:
                        break
                raise

            scene_name = str(reset_info.get("scene_name", "scene"))
            if scene_name in seen_scenes:
                if args.scene_name is not None and scenes_completed == 0:
                    raise RuntimeError(f"Scene {args.scene_name!r} was not found before scenario order repeated.")
                break
            seen_scenes.add(scene_name)

            if args.scene_name is not None and scene_name != args.scene_name:
                print(f"[Scene {scene_name}] skipped; waiting for {args.scene_name}")
                continue

            run_single_scene = args.scene_name is not None and scene_name == args.scene_name
            agent.reset()
            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_runtime_fields(obs_info, reset_info)

            safe_scene = _safe_camera_name(scene_name)
            video_path = output_dir / f"{safe_scene}.mp4"
            stale_patterns = [
                f"{safe_scene}.mp4",
                f"{safe_scene}.avi",
                f"{safe_scene}__actions.png",
                f"{safe_scene}__*__actions.png",
            ]
            for stale_path in {path for pattern in stale_patterns for path in output_dir.glob(pattern)}:
                if stale_path.exists():
                    stale_path.unlink()
            writer = EpisodeGridVideoWriter(video_path, step_size=args.step_size, camera_layout=camera_layout)

            total_reward = 0.0
            steps = 0
            frame_count = 0
            terminated = False
            truncated = False
            last_step_info = dict(reset_info)
            try:
                for _ in range(args.max_steps):
                    trajectory = agent.predict_action(obs_img, obs_info)
                    pred_traj = trajectory
                    debug = agent.last_debug

                    vis_obs_img = _overlay_pred_traj_on_front_image(obs_img, obs_info, pred_traj)
                    writer.write(vis_obs_img)
                    frame_count += 1

                    if debug_file is not None:
                        debug_file.write(
                            json.dumps(
                                {
                                    "scene": scene_name,
                                    "step": steps,
                                    "trajectory": trajectory.tolist(),
                                    "state": _debug_state(obs_info),
                                    "agent": debug,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        debug_file.flush()

                    obs, reward, terminated, truncated, step_info = env.step(trajectory)
                    last_step_info = step_info
                    obs_img, obs_info, _, _ = unpack_ad_observation(obs)
                    _sync_runtime_fields(obs_info, step_info)

                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break
            finally:
                writer.close()
                video_path = writer.output_path

            terminal_reason = _terminal_reason(last_step_info, terminated, truncated)
            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={terminal_reason} video={video_path} "
                f"codec={writer.codec} frames={frame_count}"
            )
            scenes_completed += 1

            if run_single_scene:
                break
    finally:
        env.close()
        if debug_file is not None:
            debug_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
