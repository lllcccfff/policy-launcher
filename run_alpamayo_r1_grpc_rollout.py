#!/usr/bin/env python3
"""Run Alpamayo R1 on the EasyDrive gRPC environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPAMAYO_ROOT = str(REPO_ROOT / "alpamayo1")
DEFAULT_MODEL_PATH = str(REPO_ROOT / "alpamayo1" / "Alpamayo-R1-10B")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "outputs" / "alpamayo_r1")
CAMERA_SOURCES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
VIDEO_CELL_SIZE = (960, 540)


def _sync_runtime_fields(obs_info: dict, step_info: dict) -> None:
    obs_info["scene_token"] = str(step_info["scene_name"])
    obs_info["relative_timestamp"] = float(step_info["relative_timestamp"]) * 1e-6
    obs_info["timestamp"] = float(step_info["relative_timestamp"]) * 1e-6
    obs_info["current_timestamp"] = float(step_info["current_timestamp"]) * 1e-6


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _terminal_reason(step_info: dict | None, terminated: bool, truncated: bool) -> str:
    if step_info is not None and step_info.get("reason"):
        return _safe_name(str(step_info["reason"]))
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "max_rollout_steps"


def _is_ordered_scenario_exhausted(exc: BaseException) -> bool:
    message = str(exc)
    return "No more scenarios to evaluate" in message or "No completed scene metrics" in message


def _validate_camera_contract(obs_img: dict, obs_info: dict) -> None:
    missing_images = [name for name in CAMERA_SOURCES if name not in obs_img]
    if missing_images:
        raise KeyError(f"Missing Alpamayo camera images: {missing_images}; got {sorted(obs_img)}")

    cam_params = obs_info["cam_params"]
    missing_params = [name for name in CAMERA_SOURCES if name not in cam_params]
    if missing_params:
        raise KeyError(f"Missing Alpamayo camera parameters: {missing_params}; got {sorted(cam_params)}")

    for name in CAMERA_SOURCES:
        image = np.asarray(obs_img[name])
        image = image[-1] if image.ndim == 4 else image
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 RGB HxWx3, got shape={image.shape} dtype={image.dtype}")

        camera = cam_params[name]
        extra = camera.get("extra")
        if not isinstance(extra, dict) or extra.get("type") != "ftheta":
            raise ValueError(f"{name} must provide NuRec ftheta calibration, got extra={extra}")
        if extra.get("logical_id") != name:
            raise ValueError(f"{name} calibration logical_id mismatch: {extra.get('logical_id')!r}")
        if "parameters" not in extra:
            raise ValueError(f"{name} ftheta calibration has no parameters")

        expected_shape = (int(camera["H"]), int(camera["W"]), 3)
        if image.shape != expected_shape:
            raise ValueError(f"{name} image shape {image.shape} does not match calibration {expected_shape}")


def _video_observation(obs_img: dict) -> dict[str, np.ndarray]:
    width, height = VIDEO_CELL_SIZE
    return {
        name: cv2.resize(
            obs_img[name],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        for name in CAMERA_SOURCES
    }


def _save_trajectory_outputs(path_prefix: Path, trajectories: list[np.ndarray]) -> tuple[Path, Path]:
    if not trajectories:
        raise RuntimeError("Cannot save an empty Alpamayo trajectory sequence")
    stacked = np.stack(trajectories, axis=0).astype(np.float32)
    data_path = path_prefix.with_name(path_prefix.name + "__trajectories.npz")
    np.savez_compressed(data_path, trajectories=stacked)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(stacked)))
    for step, (trajectory, color) in enumerate(zip(stacked, colors)):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.35, linewidth=1.0)
        axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=6)
    axis.scatter([0.0], [0.0], color="black", marker="x", s=50, label="ego")
    axis.set_xlabel("x forward (m)")
    axis.set_ylabel("y left (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.set_title(f"Alpamayo predictions across {len(stacked)} rollout steps")
    figure.tight_layout()
    plot_path = path_prefix.with_name(path_prefix.name + "__trajectories.png")
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return data_path, plot_path


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpamayo R1 on the EasyDrive gRPC environment")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpamayo-root", type=str, default=DEFAULT_ALPAMAYO_ROOT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--camera-cross-left", type=str, default=CAMERA_SOURCES[0])
    parser.add_argument("--camera-front-wide", type=str, default=CAMERA_SOURCES[1])
    parser.add_argument("--camera-cross-right", type=str, default=CAMERA_SOURCES[2])
    parser.add_argument("--camera-front-tele", type=str, default=CAMERA_SOURCES[3])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.alpamayo_r1 import AlpamayoR1NativeAgent, create_alpamayo_r1
    from rl_framework.common.video_writer import EpisodeGridVideoWriter

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = create_alpamayo_r1(
        {
            "alpamayo_root": args.alpamayo_root,
            "model_path": args.model_path,
            "device": args.device,
        }
    )
    camera_sources = (
        args.camera_cross_left,
        args.camera_front_wide,
        args.camera_cross_right,
        args.camera_front_tele,
    )
    agent = AlpamayoR1NativeAgent(
        bundle,
        camera_sources=camera_sources,
        top_p=args.top_p,
        temperature=args.temperature,
        max_generation_length=args.max_generation_length,
    )
    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)

    try:
        seen_scenes: set[str] = set()
        completed_scenes = 0
        while True:
            try:
                obs, reset_info = env.reset()
            except Exception as exc:
                if _is_ordered_scenario_exhausted(exc) and completed_scenes > 0:
                    break
                raise

            scene_name = str(reset_info["scene_name"])
            if scene_name in seen_scenes:
                break
            seen_scenes.add(scene_name)
            if args.scene_name is not None and scene_name != args.scene_name:
                continue

            _seed_torch(args.seed)
            agent.reset()
            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_runtime_fields(obs_info, reset_info)
            _validate_camera_contract(obs_img, obs_info)

            safe_scene = _safe_name(scene_name)
            camera_layout = ((camera_sources[0], camera_sources[1]), (camera_sources[2], camera_sources[3]))
            writer = EpisodeGridVideoWriter(output_dir / f"{safe_scene}.mp4", args.step_size, camera_layout)
            trajectories: list[np.ndarray] = []
            total_reward = 0.0
            terminated = False
            truncated = False
            last_step_info = dict(reset_info)
            steps = 0
            try:
                for _ in range(args.max_steps):
                    trajectory = agent.predict_action(obs_img, obs_info)
                    trajectories.append(trajectory)
                    writer.write(_video_observation(obs_img))
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

            reason = _terminal_reason(last_step_info, terminated, truncated)
            trajectory_data, trajectory_plot = _save_trajectory_outputs(output_dir / safe_scene, trajectories)
            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={reason} "
                f"video={writer.output_path} trajectory_data={trajectory_data} trajectory_plot={trajectory_plot}"
            )
            completed_scenes += 1
            if args.scene_name is not None:
                break
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
