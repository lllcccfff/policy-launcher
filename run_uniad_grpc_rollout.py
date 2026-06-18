#!/usr/bin/env python3
"""Run native UniAD on remote gRPC env and save one composed video per scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

DEFAULT_UNIAD_CONFIG = "UniAD_SIM/projects/configs/stage2_e2e/base_e2e.py"
DEFAULT_UNIAD_CKPT = "UniAD_SIM/ckpts/uniad_base_e2e.pth"
CAN_BUS_LABELS = (
    "ego_pos_x",
    "ego_pos_y",
    "ego_pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "yaw_rad",
    "yaw_deg",
)
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


def _rename_with_terminal_reason(path: Path, terminal_reason: str) -> Path:
    final_path = path.with_name(f"{path.stem}__{terminal_reason}{path.suffix}")
    if final_path == path:
        return path
    if final_path.exists():
        final_path.unlink()
    return path.rename(final_path)


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


def _as_numeric_array(value) -> np.ndarray | None:
    import torch

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if arr.dtype == np.object_:
        return None
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        return None
    return arr.astype(np.float32, copy=False)


def _collect_numeric_inputs(value, prefix: str, out: dict[str, np.ndarray]) -> None:
    if prefix in {"img", "raw_imgs"}:
        return
    if prefix.endswith(".img_shape"):
        return

    arr = _as_numeric_array(value)
    if arr is not None and arr.ndim <= 2:
        out[prefix] = arr
        return

    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_numeric_inputs(item, child_prefix, out)
        return

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            _collect_numeric_inputs(item, child_prefix, out)


def _leaf_label(path: str, shape: tuple[int, ...], flat_idx: int) -> str:
    base = path.split(".")[-1]
    if base.isdigit():
        base = path
    if ".lidar2img." in path:
        cam_idx = path.split(".lidar2img.", 1)[1].split(".", 1)[0]
        base = f"lidar2img_cam{cam_idx}"
    if base == "can_bus" and shape == (len(CAN_BUS_LABELS),):
        return CAN_BUS_LABELS[flat_idx]
    if len(shape) == 0:
        return base
    idx = np.unravel_index(flat_idx, shape)
    idx_str = ",".join(str(int(v)) for v in idx)
    return f"{base}[{idx_str}]"


def _safe_plot_stem(path: str) -> str:
    return _safe_camera_name(path.replace(".", "__"))


def _to_k3(cam_info: dict) -> np.ndarray:
    k = np.asarray(cam_info["K"], dtype=np.float32)
    if k.shape == (3, 3):
        return k
    if k.shape == (4, 4):
        return k[:3, :3]
    raise ValueError(f"`K` must be 3x3 or 4x4, got {k.shape}")


def _overlay_pred_traj_on_front_image(
    obs_img: dict,
    obs_info: dict,
    pred_traj: np.ndarray,
) -> dict:
    from metadrive.utils.visualize import project_points_on_image

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

    x_forward = traj[:, 1]
    y_left = -traj[:, 0]
    z_up = np.full_like(x_forward, -PRED_TRAJ_HEIGHT_BELOW_EGO_M)
    ego_points = np.stack([x_forward, y_left, z_up], axis=1)

    h, w = int(base_img.shape[0]), int(base_img.shape[1])
    projected = project_points_on_image(
        base_img,
        k3,
        ego2camera,
        h,
        w,
        ego_points,
        color=(255, 64, 64),
    )

    out_obs_img = dict(obs_img)
    if is_stacked:
        updated = np.array(front_img, copy=True)
        updated[-1] = projected.astype(updated.dtype, copy=False)
        out_obs_img["FRONT"] = updated
    else:
        out_obs_img["FRONT"] = projected.astype(front_img.dtype, copy=False)
    return out_obs_img


class RawInputTraceCollector:
    def __init__(self) -> None:
        self._traces: dict[str, list[np.ndarray]] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

    def update(self, raw_data: dict) -> None:
        leaves: dict[str, np.ndarray] = {}
        _collect_numeric_inputs(raw_data, "", leaves)
        for path, arr in leaves.items():
            arr = np.asarray(arr, dtype=np.float32)
            shape = tuple(arr.shape)
            flat = arr.reshape(-1).copy()
            prev_shape = self._shapes.get(path)
            if prev_shape is None:
                self._shapes[path] = shape
                self._traces[path] = [flat]
                continue
            if prev_shape != shape:
                raise ValueError(f"Raw input shape changed for {path}: {prev_shape} -> {shape}")
            self._traces[path].append(flat)

    def save(self, output_dir: Path, scene_name: str, terminal_reason: str) -> int:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for path in sorted(self._traces.keys()):
            traces = np.stack(self._traces[path], axis=0)
            num_values = traces.shape[1]
            steps = np.arange(traces.shape[0], dtype=np.int32)
            cols = min(4, max(1, int(np.ceil(np.sqrt(num_values)))))
            rows = int(np.ceil(num_values / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), sharex=True)
            axes = np.atleast_1d(axes).reshape(rows, cols)
            shape = self._shapes[path]
            for flat_idx in range(rows * cols):
                ax = axes.flat[flat_idx]
                if flat_idx >= num_values:
                    ax.axis("off")
                    continue
                ax.plot(steps, traces[:, flat_idx], linewidth=1.2)
                ax.set_title(_leaf_label(path, shape, flat_idx), fontsize=9)
                ax.grid(True, alpha=0.3)
                if flat_idx // cols == rows - 1:
                    ax.set_xlabel("step")
            fig.suptitle(f"{scene_name} | terminal={terminal_reason} | {path}")
            fig.tight_layout()
            fig.savefig(output_dir / f"{_safe_plot_stem(path)}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            saved += 1
        return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UniAD on gRPC env and save videos to ./tmp")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step-size", type=float, default=0.1, help="Environment control step size in seconds.")
    parser.add_argument("--timeout", type=float, default=360.0, help="gRPC timeout seconds")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="Output video directory")
    parser.add_argument("--device", type=str, default="cuda", help="UniAD inference device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.uniad.loader import create_uniad
    from rl_framework.uniad.native_agent import UniADNativeAgent
    from rl_framework.vad.native_eval import EpisodeGridVideoWriter

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uniad_cfg = {
        "config_path": DEFAULT_UNIAD_CONFIG,
        "checkpoint_path": DEFAULT_UNIAD_CKPT,
        "device": args.device,
        "AD_root": "./UniAD_SIM",
    }
    uniad_model = create_uniad(uniad_cfg)
    agent = UniADNativeAgent(uniad_model)

    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)
    camera_layout = [
        ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        ["BACK_LEFT", "BACK", "BACK_RIGHT"],
    ]

    try:
        first_scene = None
        while True:
            obs, reset_info = env.reset()
            scene_name = str(reset_info.get("scene_name", "scene"))
            if first_scene is None:
                first_scene = scene_name
            elif scene_name == first_scene:
                break

            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_runtime_fields(obs_info, reset_info)

            safe_scene = _safe_camera_name(scene_name)
            video_path = output_dir / f"{safe_scene}.mp4"
            stale_patterns = [
                f"{safe_scene}.mp4",
                f"{safe_scene}.avi",
                f"{safe_scene}__*.mp4",
                f"{safe_scene}__*.avi",
                f"{safe_scene}__*__actions.png",
                f"{safe_scene}__*__raw_inputs",
            ]
            for stale_path in {path for pattern in stale_patterns for path in output_dir.glob(pattern)}:
                if stale_path.exists():
                    if stale_path.is_dir():
                        shutil.rmtree(stale_path)
                    else:
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
            raw_input_traces = RawInputTraceCollector()

            try:
                for _ in range(args.max_steps):
                    raw_data = agent.prepare_raw_input(obs_img, obs_info, move_to_device=False, keep_raw_imgs=False)
                    raw_input_traces.update(raw_data)
                    pred_traj = agent.predict_traj_from_raw_data(raw_data)
                    vis_obs_img = _overlay_pred_traj_on_front_image(obs_img, obs_info, pred_traj)
                    writer.write(vis_obs_img)
                    frame_count += 1

                    action = agent.predict_action_from_traj(pred_traj, obs_info)
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
            video_path = _rename_with_terminal_reason(video_path, terminal_reason)
            plot_path = output_dir / f"{video_path.stem}__actions.png"
            raw_input_plot_dir = output_dir / f"{video_path.stem}__raw_inputs"
            _save_action_plot(plot_path, steering_trace, th_brk_trace, scene_name, terminal_reason)
            raw_input_plot_count = raw_input_traces.save(raw_input_plot_dir, scene_name, terminal_reason)

            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} reason={terminal_reason} video={video_path} "
                f"plot={plot_path} raw_input_plots={raw_input_plot_dir}({raw_input_plot_count}) "
                f"codec={writer.codec} frames={frame_count}"
            )

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
