from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


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
        self._writer: Optional[cv2.VideoWriter] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._codec: Optional[str] = None
        self._camera_order: List[str] = []
        self._grid_slots: List[Optional[str]] = []
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

    def _resolve_grid_slots(self, obs_img: Dict[str, np.ndarray]) -> List[Optional[str]]:
        self._rows = len(self.camera_layout)
        self._cols = max((len(row) for row in self.camera_layout), default=0)
        if self._rows <= 0 or self._cols <= 0:
            raise ValueError("`camera_layout` must be a non-empty 2D matrix of camera names.")

        slots: List[Optional[str]] = []
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

    def _codec_candidates(self) -> List[Tuple[Path, str]]:
        suffix = self.output_path.suffix.lower()
        if suffix == ".avi":
            return [(self.output_path, "MJPG"), (self.output_path, "XVID"), (self.output_path, "mp4v")]
        if suffix in {".mp4", ".m4v", ".mov"}:
            avi_path = self.output_path.with_suffix(".avi")
            return [
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

        frame_bgr = np.ascontiguousarray(frame[..., ::-1])
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
    def codec(self) -> Optional[str]:
        return self._codec
