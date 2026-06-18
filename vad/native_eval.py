from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from rl_framework.common.metrics import EpisodeMetricTracker
from rl_framework.common.grpc_obs_adapter import unpack_ad_observation
from rl_framework.vad.native_agent import FIXED_CAMERAS, VADNativeAgent

METRIC_KEYS = ("NC", "DAC", "TTC", "COM", "RC")


def _sync_scene_token(obs_info: Dict[str, Any], step_info: Dict[str, Any]) -> None:
    obs_info["scene_token"] = str(step_info["scene_name"])


class EpisodeGridVideoWriter:
    def __init__(
        self,
        output_path: Path,
        step_size: float,
        camera_layout: Sequence[Sequence[str]],
    ):
        self.output_path = output_path
        self.fps = self._infer_fps(step_size)
        self.camera_layout = [list(row) for row in camera_layout]
        self._writer: cv2.VideoWriter | None = None
        self._ffmpeg_proc: subprocess.Popen[bytes] | None = None
        self._codec: str | None = None
        self._camera_order: List[str] = []
        self._grid_slots: List[str | None] = []
        self._cols = 1
        self._rows = 1
        self._cell_w = 1
        self._cell_h = 1
        self._frame_w = 2
        self._frame_h = 2
        self._label_h = 28

    @staticmethod
    def _infer_fps(step_size: float) -> int:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"`step_size` must be finite and > 0, got {step_size}")
        return max(1, int(round(1.0 / step)))

    @staticmethod
    def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        if np.issubdtype(img.dtype, np.floating):
            clipped = np.clip(img, 0.0, 1.0) * 255.0 if img.max() <= 1.5 else np.clip(img, 0.0, 255.0)
            return clipped.astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)

    @classmethod
    def _normalize_camera_image(cls, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 4:
            arr = arr[-1]
        arr = cls._to_uint8_rgb(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported camera image shape: {arr.shape}")
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return np.ascontiguousarray(arr)

    def _resolve_grid_slots(self, obs_img: Dict[str, np.ndarray]) -> List[str | None]:
        self._rows = len(self.camera_layout)
        self._cols = max((len(row) for row in self.camera_layout), default=0)
        if self._rows <= 0 or self._cols <= 0:
            raise ValueError("`camera_layout` must be a non-empty 2D matrix of camera names.")

        slots: List[str | None] = []
        for row in range(self._rows):
            row_names = self.camera_layout[row]
            for col in range(self._cols):
                name = row_names[col] if col < len(row_names) else ""
                if name and name in obs_img:
                    slots.append(name)
                else:
                    slots.append(None)

        self._camera_order = [name for name in slots if name is not None]
        return slots

    @staticmethod
    def _even_size(value: int) -> int:
        v = int(max(value, 2))
        return v if (v % 2 == 0) else (v + 1)

    def _codec_candidates(self) -> List[tuple[Path, str]]:
        suffix = self.output_path.suffix.lower()
        if suffix == ".avi":
            return [(self.output_path, "MJPG"), (self.output_path, "XVID"), (self.output_path, "mp4v")]
        if suffix in {".mp4", ".m4v", ".mov"}:
            avi_path = self.output_path.with_suffix(".avi")
            return [
                # Prefer AVC/H.264 before MPEG-4 Part 2 for better native player support.
                (self.output_path, "avc1"),
                (self.output_path, "H264"),
                (self.output_path, "mp4v"),
                (avi_path, "MJPG"),
                (avi_path, "XVID"),
            ]
        avi_path = self.output_path.with_suffix(".avi")
        return [
            (self.output_path, "avc1"),
            (self.output_path, "H264"),
            (self.output_path, "mp4v"),
            (avi_path, "MJPG"),
            (avi_path, "XVID"),
        ]

    def _try_init_ffmpeg_writer(self) -> bool:
        if self.output_path.suffix.lower() not in {".mp4", ".m4v", ".mov"}:
            return False

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return False

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self._frame_w}x{self._frame_h}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False

        if proc.stdin is None:
            try:
                proc.terminate()
            except OSError:
                pass
            return False
        if proc.poll() is not None:
            return False

        self._ffmpeg_proc = proc
        self._codec = "libx264"
        return True

    def _init_writer(self, obs_img: Dict[str, np.ndarray]) -> None:
        self._grid_slots = self._resolve_grid_slots(obs_img)

        max_h, max_w = 1, 1
        for name in self._camera_order:
            img = self._normalize_camera_image(obs_img[name])
            h, w = img.shape[:2]
            max_h, max_w = max(max_h, h), max(max_w, w)
        self._cell_h = max_h + self._label_h
        self._cell_w = max_w

        frame_w = self._cell_w * self._cols
        frame_h = self._cell_h * self._rows
        self._frame_w = self._even_size(frame_w)
        self._frame_h = self._even_size(frame_h)

        if self._try_init_ffmpeg_writer():
            return

        candidates = self._codec_candidates()

        for path, codec in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self._frame_w, self._frame_h))
            if writer is not None and writer.isOpened():
                self._writer = writer
                self._codec = codec
                self.output_path = path
                return
            if writer is not None:
                writer.release()

        attempted = ", ".join([f"{codec}@{path}" for path, codec in candidates])
        raise RuntimeError(f"Failed to open VideoWriter. Attempted: {attempted}")

    def _add_label(self, img_rgb: np.ndarray, name: str) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        canvas = np.zeros((h + self._label_h, w, 3), dtype=np.uint8)
        canvas[self._label_h :, :] = img_rgb
        text_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
        tx = max((w - text_size[0]) // 2, 2)
        ty = max((self._label_h + text_size[1]) // 2, text_size[1] + 2)
        cv2.putText(
            canvas,
            name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return canvas

    def write(self, obs_img: Dict[str, np.ndarray]) -> None:
        if not obs_img:
            return
        if self._writer is None and self._ffmpeg_proc is None:
            self._init_writer(obs_img)
        if self._writer is None and self._ffmpeg_proc is None:
            raise RuntimeError("No available video writer backend.")

        grid_h = self._cell_h * self._rows
        grid_w = self._cell_w * self._cols
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx, name in enumerate(self._grid_slots):
            if name is None or name not in obs_img:
                continue
            img = self._normalize_camera_image(obs_img[name])
            labeled = self._add_label(img, name)
            lh, lw = labeled.shape[:2]
            row, col = divmod(idx, self._cols)
            y0, x0 = row * self._cell_h, col * self._cell_w
            y_off = (self._cell_h - lh) // 2
            x_off = (self._cell_w - lw) // 2
            grid[y0 + y_off : y0 + y_off + lh, x0 + x_off : x0 + x_off + lw] = labeled

        if grid_h != self._frame_h or grid_w != self._frame_w:
            frame = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
            frame[:grid_h, :grid_w] = grid
        else:
            frame = grid

        frame_bgr = np.ascontiguousarray(frame[..., ::-1])  # RGB -> BGR
        if self._ffmpeg_proc is not None:
            stdin = self._ffmpeg_proc.stdin
            if stdin is None:
                raise RuntimeError("FFmpeg writer stdin is unavailable.")
            try:
                stdin.write(frame_bgr.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("FFmpeg video writer failed: broken pipe") from exc
            return

        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._ffmpeg_proc is not None:
            proc = self._ffmpeg_proc
            self._ffmpeg_proc = None
            if proc.stdin is not None:
                proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"FFmpeg video writer exited with code {ret}.")

        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @property
    def codec(self) -> str | None:
        return self._codec


def _as_vec(value: Any, dim: int = 3) -> np.ndarray:
    arr = np.asarray(value if value is not None else [0.0] * dim, dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _compose_ego_state(obs_info: Dict[str, Any]) -> Dict[str, Any]:
    ego_pos = _as_vec(obs_info["ego_pos"], dim=3)
    ego_rot = _as_vec(obs_info["ego_rot"], dim=3)
    linear_vel = _as_vec(obs_info["linear_velocity"], dim=3)
    linear_acc = _as_vec(obs_info["linear_acceleration"], dim=3)
    angular_vel = _as_vec(obs_info["angular_velocity"], dim=3)
    yaw = float(ego_rot[2])

    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    ego_vel = np.array(
        [
            cos_yaw * linear_vel[0] + sin_yaw * linear_vel[1],
            -sin_yaw * linear_vel[0] + cos_yaw * linear_vel[1],
        ],
        dtype=np.float32,
    )
    ego_acc = np.array(
        [
            cos_yaw * linear_acc[0] + sin_yaw * linear_acc[1],
            -sin_yaw * linear_acc[0] + cos_yaw * linear_acc[1],
        ],
        dtype=np.float32,
    )

    return {
        "position_world": ego_pos[:2].astype(np.float32).tolist(),
        "ego_velocity": ego_vel.astype(np.float32).tolist(),
        "ego_acceleration": ego_acc.astype(np.float32).tolist(),
        "yaw": yaw,
        "angular_velocity": float(angular_vel[2]),
        "timestamp": float(obs_info["timestamp"]),
    }


def _extract_surroundings(obs_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    surroundings: List[Dict[str, Any]] = []
    for obj in obs_info["surroundings"]:
        position = _as_vec(obj["position"], dim=2).tolist()
        velocity = _as_vec(obj["velocity"], dim=2).tolist()
        size = _as_vec(obj["size"], dim=2).tolist()
        surroundings.append({"position": position, "velocity": velocity, "size": size})
    return surroundings


def _extract_expert_path(obs_info: Dict[str, Any]) -> Dict[str, Any]:
    pts = np.asarray(obs_info["expert_path"], dtype=np.float32)[:, :2]
    seg_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cumlen = np.concatenate(([0.0], np.cumsum(seg_lengths))).astype(np.float32)
    return {
        "start_xy": pts[0].tolist(),
        "end_xy": pts[-1].tolist(),
        "path_xy": pts.tolist(),
        "path_cumlen": cumlen.tolist(),
    }


def build_metric_payload(obs_info: Dict[str, Any], step_info: Dict[str, Any] | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if step_info:
        if "collision" in step_info:
            payload["collision"] = step_info["collision"]
        if "position_deviation" in step_info:
            payload["position_deviation"] = step_info["position_deviation"]
    payload["ego_state"] = _compose_ego_state(obs_info)
    payload["surroundings"] = _extract_surroundings(obs_info)
    payload["expert_path"] = _extract_expert_path(obs_info)
    return payload


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_vad_native(
    env: Any,
    agent: VADNativeAgent,
    episodes: int = 3,
    max_steps: int = 1000,
    save_video: bool = False,
    output_dir: str | Path = "./tmp",
    save_json: bool = False,
    video_step_size: float = 0.1,
    video_camera_layout: Sequence[Sequence[str]] = (
        ("FRONT_LEFT", "FRONT", "FRONT_RIGHT"),
        ("BACK_LEFT", "BACK", "BACK_RIGHT"),
    ),
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    if save_video or save_json:
        output_path.mkdir(parents=True, exist_ok=True)

    episode_records: List[Dict[str, Any]] = []
    for ep in range(1, episodes + 1):
        obs, reset_info = env.reset()
        obs_img, obs_info, _, _ = unpack_ad_observation(obs)
        _sync_scene_token(obs_info, reset_info)

        tracker = EpisodeMetricTracker()
        tracker.update(build_metric_payload(obs_info, reset_info))

        writer = None
        video_path = None
        if save_video:
            video_path = output_path / f"episode_{ep:03d}.mp4"
            writer = EpisodeGridVideoWriter(
                video_path,
                step_size=video_step_size,
                camera_layout=video_camera_layout,
            )

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        try:
            for _ in range(max_steps):
                if writer is not None:
                    writer.write(obs_img)

                action = agent.predict_action(obs_img, obs_info)
                obs, reward, terminated, truncated, step_info = env.step(action)
                obs_img, obs_info, _, _ = unpack_ad_observation(obs)
                _sync_scene_token(obs_info, step_info)

                tracker.update(build_metric_payload(obs_info, step_info))
                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break
        finally:
            if writer is not None:
                writer.close()
                video_path = writer.output_path

        metrics = tracker.finalize()
        metric_str = ", ".join(f"{k}={metrics[k]:.3f}" for k in METRIC_KEYS)
        print(
            f"[Episode {ep}] Reward={total_reward:.3f}, Length={steps}, "
            f"{metric_str}, terminated={terminated}, truncated={truncated}"
        )

        record = {
            "episode": ep,
            "reward": float(total_reward),
            "length": int(steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "metrics": metrics,
        }
        if video_path is not None:
            record["video"] = str(video_path)
        episode_records.append(record)

    if not episode_records:
        overall = {
            "num_episodes": 0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "mean_metrics": {k: 0.0 for k in METRIC_KEYS},
        }
    else:
        rewards = np.asarray([r["reward"] for r in episode_records], dtype=np.float32)
        lengths = np.asarray([r["length"] for r in episode_records], dtype=np.float32)
        mean_metrics = {
            k: float(np.mean([float(r["metrics"][k]) for r in episode_records])) for k in METRIC_KEYS
        }
        overall = {
            "num_episodes": int(len(episode_records)),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "mean_metrics": mean_metrics,
        }

    overall_metric_str = ", ".join(f"{k}={overall['mean_metrics'][k]:.3f}" for k in METRIC_KEYS)
    print(
        f"[Overall] episodes={overall['num_episodes']}, mean_reward={overall['mean_reward']:.3f}, "
        f"mean_length={overall['mean_length']:.2f}, {overall_metric_str}"
    )

    results = {"episodes": episode_records, "overall": overall}
    if save_json:
        json_path = output_path / "vad_native_eval_results.json"
        txt_path = output_path / "vad_native_eval_results.txt"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(results), f, ensure_ascii=False, indent=2)
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(
                f"episodes={overall['num_episodes']} "
                f"mean_reward={overall['mean_reward']:.3f} "
                f"mean_length={overall['mean_length']:.2f} "
                f"{overall_metric_str}\n"
            )
        print(f"Saved eval results: {json_path}")
        print(f"Saved eval summary: {txt_path}")

    return results
