#!/usr/bin/env python3
"""Run Epona on remote gRPC env and save one composed video per scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPONA_ROOT = str(REPO_ROOT / "Epona")
DEFAULT_EPONA_CONFIG = str(REPO_ROOT / "Epona" / "configs" / "dit_config_dcae_nuscenes.py")
DEFAULT_EPONA_CKPT = str(REPO_ROOT / "Epona" / "pretrained" / "epona_nuplan+nusc.pkl")
DEFAULT_EPONA_VAE_CKPT = str(REPO_ROOT / "Epona" / "pretrained" / "dcae_td_20000.pkl")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "outputs" / "epona")
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


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_action_plot(plot_path: Path, steering: list[float], th_brk: list[float], scene_name: str, terminal_reason: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(steering), dtype=np.int32)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(steps, steering, color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("steering")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, th_brk, color="tab:orange", linewidth=1.5)
    axes[1].set_ylabel("th_brk")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{scene_name} | terminal={terminal_reason}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _to_k3(cam_info: dict) -> np.ndarray:
    k = np.asarray(cam_info["K"], dtype=np.float32)
    if k.shape == (3, 3):
        return k
    if k.shape == (4, 4):
        return k[:3, :3]
    raise ValueError(f"`K` must be 3x3 or 4x4, got {k.shape}")


def _overlay_pred_traj_on_front_image(obs_img: dict, obs_info: dict, pred_traj: np.ndarray) -> dict:
    if "FRONT" not in obs_img:
        raise KeyError("Epona visualization requires FRONT camera")
    if "cam_params" not in obs_info or "FRONT" not in obs_info["cam_params"]:
        raise KeyError("Epona visualization requires FRONT camera params")

    front_img = np.asarray(obs_img["FRONT"])
    is_stacked = front_img.ndim == 4
    base_img = front_img[-1] if is_stacked else front_img
    if base_img.ndim != 3 or base_img.shape[2] < 3:
        raise ValueError(f"FRONT image must have shape HxWx3 or TxHxWx3, got {front_img.shape}")

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
            cv2.circle(projected, tuple(point), radius=2, color=(255, 128, 0), thickness=-1)
        if len(points_2d) >= 2:
            for start, end in zip(points_2d[:-1], points_2d[1:]):
                cv2.line(projected, tuple(start), tuple(end), color=(255, 128, 0), thickness=1)

    out_obs_img = dict(obs_img)
    if is_stacked:
        updated = np.array(front_img, copy=True)
        updated[-1] = projected.astype(updated.dtype, copy=False)
        out_obs_img["FRONT"] = updated
    else:
        out_obs_img["FRONT"] = projected.astype(front_img.dtype, copy=False)
    return out_obs_img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Epona on gRPC env and save videos")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step-size", type=float, default=0.1, help="Environment control step size in seconds")
    parser.add_argument("--timeout", type=float, default=360.0, help="gRPC timeout seconds")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output video directory")
    parser.add_argument("--device", type=str, default="cuda", help="Epona inference CUDA device")
    parser.add_argument("--epona-root", type=str, default=DEFAULT_EPONA_ROOT, help="Epona checkout root")
    parser.add_argument("--config-path", type=str, default=DEFAULT_EPONA_CONFIG, help="Epona config path")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_EPONA_CKPT, help="Epona world-model checkpoint path")
    parser.add_argument("--vae-checkpoint-path", type=str, default=DEFAULT_EPONA_VAE_CKPT, help="Epona DCAE checkpoint path")
    parser.add_argument("--camera", type=str, default="FRONT", help="Camera used by Epona")
    parser.add_argument("--scene-name", type=str, default=None, help="Run only this scene name")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed for deterministic Epona sampling")
    parser.add_argument("--num-sampling-steps", type=int, default=None, help="Override Epona diffusion sampling steps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.common.video_writer import EpisodeGridVideoWriter
    from rl_framework.epona import EponaNativeAgent, create_epona

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_epona(
        {
            "epona_root": args.epona_root,
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
            "vae_checkpoint_path": args.vae_checkpoint_path,
            "device": args.device,
            "batch_size": 1,
            "num_sampling_steps": args.num_sampling_steps,
        }
    )
    agent = EponaNativeAgent(bundle, camera=args.camera)

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

            _seed_torch(int(args.seed))
            agent.reset()
            run_single_scene = args.scene_name is not None and scene_name == args.scene_name

            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_runtime_fields(obs_info, reset_info)

            safe_scene = _safe_camera_name(scene_name)
            video_path = output_dir / f"{safe_scene}.mp4"
            stale_patterns = [
                f"{safe_scene}.mp4",
                f"{safe_scene}.avi",
                f"{safe_scene}__*.mp4",
                f"{safe_scene}__*.avi",
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
            steering_trace: list[float] = []
            th_brk_trace: list[float] = []

            try:
                for _ in range(args.max_steps):
                    action = agent.predict_action(obs_img, obs_info)
                    pred_traj = agent.last_trajectory
                    vis_obs_img = _overlay_pred_traj_on_front_image(obs_img, obs_info, pred_traj)
                    writer.write(vis_obs_img)
                    frame_count += 1

                    steering_trace.append(float(action[0]))
                    th_brk_trace.append(float(action[1]))
                    obs, reward, terminated, truncated, step_info = env.step(action)
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
            plot_path = output_dir / f"{safe_scene}__actions.png"
            _save_action_plot(plot_path, steering_trace, th_brk_trace, scene_name, terminal_reason)
            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={terminal_reason} video={video_path} "
                f"plot={plot_path} codec={writer.codec} frames={frame_count}"
            )
            scenes_completed += 1

            if run_single_scene:
                break

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
