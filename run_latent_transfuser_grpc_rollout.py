#!/usr/bin/env python3
"""Run native Latent TransFuser on remote gRPC env and save one video per scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

DEFAULT_TRANSFUSER_ROOT = "TransFuser"
DEFAULT_LATENT_TF_MODEL_DIR = "TransFuser/model_ckpt/models_2022/latentTF"


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


def _rename_with_terminal_reason(path: Path, terminal_reason: str) -> Path:
    final_path = path.with_name(f"{path.stem}__{terminal_reason}{path.suffix}")
    if final_path == path:
        return path
    if final_path.exists():
        final_path.unlink()
    path.rename(final_path)
    return final_path


def _target_point_image_to_rgb(target_point_image: np.ndarray) -> np.ndarray:
    image = np.asarray(target_point_image)
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected draw_target_point output with shape 1xHxW, got {image.shape}")
    grayscale = (image[0] * 255.0).astype(np.uint8)
    return np.repeat(grayscale[:, :, None], 3, axis=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Latent TransFuser on gRPC env and save videos to ./tmp")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step-size", type=float, default=0.1, help="Environment control step size in seconds")
    parser.add_argument("--timeout", type=float, default=30.0, help="gRPC timeout seconds")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="Output video directory")
    parser.add_argument("--device", type=str, default="cuda", help="Latent TransFuser inference device")
    parser.add_argument("--transfuser-root", type=str, default=DEFAULT_TRANSFUSER_ROOT, help="TransFuser checkout root")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_LATENT_TF_MODEL_DIR, help="Latent TransFuser checkpoint dir")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional single checkpoint path")
    parser.add_argument(
        "--route-planner-max-distance",
        type=float,
        default=None,
        help="Override the Latent TransFuser route planner maximum distance",
    )
    parser.add_argument(
        "--render-target-point",
        action="store_true",
        help="Render the draw_target_point model input as a video tile",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.common.video_writer import EpisodeGridVideoWriter
    from rl_framework.latent_transfuser.loader import create_latent_transfuser
    from rl_framework.latent_transfuser.native_agent import LatentTransFuserNativeAgent

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_config = {
        "transfuser_root": args.transfuser_root,
        "model_dir": args.model_dir,
        "device": args.device,
    }
    if args.checkpoint_path:
        latent_config["checkpoint_path"] = args.checkpoint_path
    bundle = create_latent_transfuser(latent_config)
    if args.route_planner_max_distance is not None:
        if args.route_planner_max_distance <= 0.0:
            raise ValueError("--route-planner-max-distance must be > 0")
        bundle.config.route_planner_max_distance = args.route_planner_max_distance
    agent = LatentTransFuserNativeAgent(bundle, capture_target_point_image=args.render_target_point)

    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)
    camera_layout = [
        ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        ["BACK_LEFT", "BACK", "BACK_RIGHT"],
    ]
    if args.render_target_point:
        camera_layout[1] = ["", "TARGET_POINT", ""]

    try:
        first_scene = None
        scenes_completed = 0
        while True:
            try:
                obs, reset_info = env.reset()
            except RuntimeError as exc:
                if scenes_completed > 0 and "No more scenarios to evaluate." in str(exc):
                    break
                raise
            scene_name = str(reset_info.get("scene_name", "scene"))
            if first_scene is None:
                first_scene = scene_name
            elif scene_name == first_scene:
                break

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

            total_reward = 0.0
            steps = 0
            frame_count = 0
            terminated = False
            truncated = False
            last_step_info = dict(reset_info)

            try:
                for _ in range(args.max_steps):
                    action = agent.predict_action(obs_img, obs_info)
                    video_obs_img = obs_img
                    if args.render_target_point:
                        if agent.last_target_point_image is None:
                            raise RuntimeError("Latent TransFuser did not produce a target-point image")
                        video_obs_img = dict(obs_img)
                        video_obs_img["TARGET_POINT"] = _target_point_image_to_rgb(agent.last_target_point_image)
                    writer.write(video_obs_img)
                    frame_count += 1

                    obs, reward, terminated, truncated, step_info = env.step(action)
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

            terminal_reason = _terminal_reason(last_step_info, terminated, truncated)
            video_path = _rename_with_terminal_reason(video_path, terminal_reason)

            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={terminal_reason} video={video_path} "
                f"codec={writer.codec} frames={frame_count}"
            )
            scenes_completed += 1

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
