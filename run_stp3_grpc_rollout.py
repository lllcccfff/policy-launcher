#!/usr/bin/env python3
"""Run native ST-P3 on remote gRPC env and save one composed video per scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STP3_ROOT = str(REPO_ROOT / "ST-P3")
DEFAULT_STP3_CKPT = str(REPO_ROOT / "ST-P3" / "ckpts" / "stp3_carla.ckpt")
DEFAULT_OUTPUT_DIR = "./tmp_stp3"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-P3 on gRPC env and save videos to ./tmp_stp3")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step-size", type=float, default=0.02, help="Environment control step size in seconds.")
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
                f"{safe_scene}__*__actions.png",
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
            steering_trace: list[float] = []
            th_brk_trace: list[float] = []

            try:
                for _ in range(args.max_steps):
                    writer.write(obs_img)
                    frame_count += 1

                    action = agent.predict_action(obs_img, obs_info)
                    steering_trace.append(float(action[0]))
                    th_brk_trace.append(float(action[1]))
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
            plot_path = output_dir / f"{video_path.stem}__actions.png"
            _save_action_plot(plot_path, steering_trace, th_brk_trace, scene_name, terminal_reason)

            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={terminal_reason} video={video_path} "
                f"plot={plot_path} codec={writer.codec} frames={frame_count}"
            )
            scenes_completed += 1

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
