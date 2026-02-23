#!/usr/bin/env python3
"""Run native VAD on remote gRPC env and save one composed video per episode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

DEFAULT_VAD_CONFIG = "VAD/projects/configs/VAD/VAD_base_stage_2.py"
DEFAULT_VAD_CKPT = "VAD/ckpts/VAD_base.pth"


def _sync_scene_token(obs_info: dict, step_info: dict | None) -> None:
    scene_name = step_info.get("scene_name") if step_info else None
    if scene_name is not None:
        obs_info["scene_token"] = str(scene_name)
    else:
        obs_info.setdefault("scene_token", "grpc_scene")


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        clipped = np.clip(img, 0.0, 1.0) * 255.0 if img.max() <= 1.5 else np.clip(img, 0.0, 255.0)
        return clipped.astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _safe_camera_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _save_obs_images(
    obs_img: dict,
    output_dir: Path,
    preferred_order: Sequence[str],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [k for k in preferred_order if k in obs_img]
    ordered.extend([k for k in sorted(obs_img.keys()) if k not in ordered])

    saved = 0
    for cam_name in ordered:
        if cam_name not in obs_img:
            continue
        img = np.asarray(obs_img[cam_name])
        if img.ndim == 4:
            img = img[-1]
        img = _to_uint8_rgb(img)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        if img.ndim != 3:
            raise ValueError(f"Unsupported image shape for camera {cam_name}: {img.shape}")
        if img.shape[2] > 3:
            img = img[:, :, :3]

        # Overwrite per-camera image every step, so only latest frame is kept.
        filename = f"current_{_safe_camera_name(cam_name)}.png"
        image_path = output_dir / filename
        ok = cv2.imwrite(str(image_path), img[..., ::-1])  # RGB -> BGR
        if not ok:
            raise RuntimeError(f"Failed to save image: {image_path}")
        saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAD on gRPC env and save videos + raw camera images to ./tmp")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--timeout", type=float, default=30.0, help="gRPC timeout seconds")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="Output video directory")
    parser.add_argument("--vad-config", type=str, default=DEFAULT_VAD_CONFIG, help="VAD config path")
    parser.add_argument("--vad-checkpoint", type=str, default=DEFAULT_VAD_CKPT, help="VAD checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0", help="VAD inference device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.vad.loader import create_vad
    from rl_framework.vad.native_agent import FIXED_CAMERAS, VADNativeAgent
    from rl_framework.vad.native_eval import EpisodeGridVideoWriter

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vad_cfg = {
        "config_path": args.vad_config,
        "checkpoint_path": args.vad_checkpoint,
        "device": args.device,
        "runtime_mode": "native_infer",
    }
    vad_model = create_vad(vad_cfg)
    agent = VADNativeAgent(vad_model)

    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)
    preferred_order = FIXED_CAMERAS

    try:
        for ep in range(1, args.episodes + 1):
            (obs_img, obs_info), reset_info = env.reset()
            _sync_scene_token(obs_info, reset_info)

            video_path = output_dir / f"episode_{ep:03d}.mp4"
            for stale_path in (video_path, video_path.with_suffix(".avi")):
                if stale_path.exists():
                    stale_path.unlink()
            writer = EpisodeGridVideoWriter(video_path, fps=args.fps, preferred_order=preferred_order)
            image_dir = output_dir / f"episode_{ep:03d}_images"

            total_reward = 0.0
            steps = 0
            frame_count = 0
            terminated = False
            truncated = False

            try:
                for _ in range(args.max_steps):
                    writer.write(obs_img)
                    _save_obs_images(obs_img, image_dir, preferred_order)
                    frame_count += 1

                    action = agent.predict_action(obs_img, obs_info)
                    (obs_img, obs_info), reward, terminated, truncated, step_info = env.step(action)
                    _sync_scene_token(obs_info, step_info)

                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break
            finally:
                writer.close()
                video_path = writer.output_path

            print(
                f"[Episode {ep}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} video={video_path} "
                f"codec={writer.codec} images={image_dir} frames={frame_count}"
            )

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
