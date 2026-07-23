#!/usr/bin/env python3
"""Run native UniAD or VAD on a gRPC environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

MODEL_PATHS = {
    "uniad": {
        "config_path": "UniAD_SIM/projects/configs/stage2_e2e/base_e2e.py",
        "checkpoint_path": "UniAD_SIM/ckpts/uniad_base_e2e.pth",
        "AD_root": "./UniAD_SIM",
    },
    "vad": {
        "config_path": "VAD/projects/configs/VAD/VAD_base_stage_2.py",
        "checkpoint_path": "VAD/ckpts/VAD_base.pth",
        "AD_root": "./VAD",
    },
}
DEFAULT_TIMEOUTS = {"uniad": 360.0, "vad": 30.0}
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
    if prefix == "img":
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
    parser = argparse.ArgumentParser(description="Run UniAD or VAD on a gRPC environment")
    parser.add_argument("--model", choices=("uniad", "vad"), required=True, help="Native planning model to run")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--timeout", type=float, help="gRPC timeout seconds (model-specific default)")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="UniAD raw-input trace directory")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--trace-raw-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save UniAD model-input traces (enabled by default for UniAD)",
    )
    return parser.parse_args()


def _create_agent(model_name: str, device: str):
    model_cfg = {**MODEL_PATHS[model_name], "device": device}
    if model_name == "uniad":
        from rl_framework.uniad.loader import create_uniad
        from rl_framework.uniad.native_agent import UniADNativeAgent

        return UniADNativeAgent(create_uniad(model_cfg))

    from rl_framework.vad.loader import create_vad
    from rl_framework.vad.native_agent import VADNativeAgent

    model_cfg["runtime_mode"] = "native_infer"
    return VADNativeAgent(create_vad(model_cfg))


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    trace_raw_inputs = args.model == "uniad" if args.trace_raw_inputs is None else args.trace_raw_inputs
    if trace_raw_inputs and args.model != "uniad":
        raise ValueError("Raw input tracing is only available for UniAD")

    output_dir = Path(args.output_dir) if trace_raw_inputs else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    agent = _create_agent(args.model, args.device)

    timeout = DEFAULT_TIMEOUTS[args.model] if args.timeout is None else args.timeout
    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=timeout, auto_wait_ready=True)
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

            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_runtime_fields(obs_info, reset_info)

            if trace_raw_inputs:
                safe_scene = _safe_camera_name(scene_name)
                for stale_path in output_dir.glob(f"{safe_scene}__*__raw_inputs"):
                    shutil.rmtree(stale_path)

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            last_step_info = dict(reset_info)
            raw_input_traces = RawInputTraceCollector() if trace_raw_inputs else None

            for _ in range(args.max_steps):
                trajectory = agent.predict_action(obs_img, obs_info)
                if raw_input_traces is not None:
                    raw_input_traces.update(agent.last_raw_data)

                obs, reward, terminated, truncated, step_info = env.step(trajectory)
                last_step_info = step_info
                obs_img, obs_info, _, _ = unpack_ad_observation(obs)
                _sync_runtime_fields(obs_info, step_info)

                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break

            terminal_reason = _terminal_reason(last_step_info, terminated, truncated)
            raw_input_summary = ""
            if raw_input_traces is not None:
                raw_input_plot_dir = output_dir / f"{safe_scene}__{terminal_reason}__raw_inputs"
                raw_input_plot_count = raw_input_traces.save(raw_input_plot_dir, scene_name, terminal_reason)
                raw_input_summary = f" raw_input_plots={raw_input_plot_dir}({raw_input_plot_count})"

            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"model={args.model} terminated={terminated} truncated={truncated} reason={terminal_reason}"
                f"{raw_input_summary} "
            )
            scenes_completed += 1

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
