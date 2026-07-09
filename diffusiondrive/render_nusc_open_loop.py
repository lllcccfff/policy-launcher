#!/usr/bin/env python3
"""Render DiffusionDrive open-loop planning on raw nuScenes images."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from rl_framework.diffusiondrive.compare_inputs import (
    NUSC_CAMERA_ORDER,
    NUSC_TO_STREETWORLD_CAMERA,
    _nusc_lidar_pose,
    _nusc_scene_samples,
    _nusc_sensor2top,
    load_nuscene_raw_sequence,
)
from rl_framework.diffusiondrive.loader import create_diffusiondrive
from rl_framework.diffusiondrive.native_agent import DiffusionDriveNativeAgent
from rl_framework.run_diffusiondrive_grpc_rollout import (
    DEFAULT_DIFFUSIONDRIVE_CKPT,
    DEFAULT_DIFFUSIONDRIVE_ROOT,
    REPO_ROOT,
    _seed_torch,
)

MOSAIC_ORDER = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
)


def _lidar2img_original(cam_info: dict[str, Any]) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64))
    lidar2cam_t = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def _camera_frames(nusc: Any, sample: dict[str, Any], dataroot: Path) -> dict[str, dict[str, Any]]:
    lidar2global, cs_record, pose_record = _nusc_lidar_pose(nusc, sample)
    del lidar2global
    l2e_t = np.asarray(cs_record["translation"], dtype=np.float64)
    l2e_r_mat = Quaternion(cs_record["rotation"]).rotation_matrix
    e2g_t = np.asarray(pose_record["translation"], dtype=np.float64)
    e2g_r_mat = Quaternion(pose_record["rotation"]).rotation_matrix

    frames = {}
    for nusc_camera in NUSC_CAMERA_ORDER:
        cam_token = sample["data"][nusc_camera]
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        image_path = Path(cam_path)
        if not image_path.is_absolute():
            image_path = dataroot / image_path
        cam_info = _nusc_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, nusc_camera)
        cam_info["cam_intrinsic"] = np.asarray(cam_intrinsic, dtype=np.float32)
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        frames[nusc_camera] = {
            "image_rgb": image_rgb,
            "lidar2img": _lidar2img_original(cam_info),
            "streetworld_camera": NUSC_TO_STREETWORLD_CAMERA[nusc_camera],
            "image_path": str(image_path),
        }
    return frames


def _densify(points: np.ndarray, points_per_segment: int = 20) -> np.ndarray:
    dense = []
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        for step in range(points_per_segment):
            alpha = step / float(points_per_segment)
            dense.append(start * (1.0 - alpha) + end * alpha)
    dense.append(points[-1])
    return np.asarray(dense, dtype=np.float32)


def _project_points(points_lidar: np.ndarray, lidar2img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1), dtype=np.float32)], axis=1)
    projected = points_h @ lidar2img.T
    depth = projected[:, 2]
    uv = projected[:, :2] / depth[:, None]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.1)
    return uv, valid


def _draw_traj(image_rgb: np.ndarray, planning_xy: np.ndarray, lidar2img: np.ndarray) -> np.ndarray:
    canvas = np.ascontiguousarray(image_rgb.copy())
    planning_xy = np.asarray(planning_xy, dtype=np.float32)
    plan_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), planning_xy[:, :2]], axis=0)
    points_lidar = np.concatenate([plan_xy, np.full((plan_xy.shape[0], 1), -1.8, dtype=np.float32)], axis=1)
    points_lidar = _densify(points_lidar)
    uv, valid = _project_points(points_lidar, lidar2img)

    color = (255, 150, 0)
    for index in range(len(uv) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        p0 = tuple(np.round(uv[index]).astype(int).tolist())
        p1 = tuple(np.round(uv[index + 1]).astype(int).tolist())
        cv2.line(canvas, p0, p1, color, 6, lineType=cv2.LINE_AA)
    for point, is_valid in zip(uv[::20], valid[::20]):
        if not is_valid:
            continue
        center = tuple(np.round(point).astype(int).tolist())
        cv2.circle(canvas, center, 9, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 11, (20, 20, 20), 2, lineType=cv2.LINE_AA)
    return canvas


def _make_mosaic(frames: dict[str, dict[str, Any]], planning_xy: np.ndarray, scale: float, title: str) -> np.ndarray:
    rows = []
    for row in MOSAIC_ORDER:
        images = []
        for nusc_camera in row:
            rendered = _draw_traj(frames[nusc_camera]["image_rgb"], planning_xy, frames[nusc_camera]["lidar2img"])
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            if scale != 1.0:
                rendered_bgr = cv2.resize(rendered_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(
                rendered_bgr,
                nusc_camera,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            images.append(rendered_bgr)
        rows.append(np.concatenate(images, axis=1))
    mosaic = np.concatenate(rows, axis=0)
    cv2.putText(mosaic, title, (20, mosaic.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    return mosaic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DiffusionDrive open-loop planning on raw nuScenes RGB")
    parser.add_argument("--nuscene-root", type=str, default=str(REPO_ROOT / "data" / "raw" / "nuscenes"))
    parser.add_argument("--nuscene-version", type=str, default="v1.0-trainval")
    parser.add_argument("--nuscene-scene", type=str, default="scene-0008")
    parser.add_argument("--nuscene-start-index", type=int, default=21)
    parser.add_argument("--steps", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusiondrive-root", type=str, default=DEFAULT_DIFFUSIONDRIVE_ROOT)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_DIFFUSIONDRIVE_CKPT)
    parser.add_argument("--output-video", type=str, default=str(REPO_ROOT / "tmp_diffusiondrive" / "0008_3_open_loop.mp4"))
    parser.add_argument("--output-json", type=str, default=str(REPO_ROOT / "tmp_diffusiondrive" / "0008_3_open_loop.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if args.scale <= 0.0:
        raise ValueError(f"--scale must be positive, got {args.scale}")
    if args.fps <= 0.0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    from nuscenes.nuscenes import NuScenes

    dataroot = Path(args.nuscene_root).expanduser().resolve()
    raw_samples, raw_records = load_nuscene_raw_sequence(
        str(dataroot),
        args.nuscene_version,
        args.nuscene_scene,
        args.nuscene_start_index,
        args.steps,
    )
    nusc = NuScenes(version=args.nuscene_version, dataroot=str(dataroot), verbose=False)
    scene_samples = _nusc_scene_samples(nusc, args.nuscene_scene, args.nuscene_start_index, args.steps)

    _seed_torch(args.seed)
    model = create_diffusiondrive(
        {
            "diffusiondrive_root": args.diffusiondrive_root,
            "checkpoint_path": args.checkpoint_path,
            "device": args.device,
        }
    )
    agent = DiffusionDriveNativeAgent(model)
    agent.reset()

    output_video = Path(args.output_video).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video = output_video.with_name(f"{output_video.stem}.raw.mp4")

    writer = None
    summaries = []
    try:
        for index, (raw_data, raw_record, sample) in enumerate(zip(raw_samples, raw_records, scene_samples)):
            step_seed = int(args.seed) + index
            _seed_torch(step_seed)
            img, data = agent._batch_raw_data(raw_data)
            with torch.inference_mode():
                output = model.forward(img, **data)
            planning_xy = output[0]["img_bbox"]["final_planning"].detach().cpu().numpy().astype(np.float32)
            if planning_xy.ndim != 2 or planning_xy.shape[1] != 2:
                raise ValueError(f"final_planning must have shape Nx2, got {planning_xy.shape}")

            frames = _camera_frames(nusc, sample, dataroot)
            title = (
                f"{args.nuscene_scene} index={args.nuscene_start_index + index} "
                f"timestamp={raw_record['timestamp_us']} seed={step_seed}"
            )
            mosaic = _make_mosaic(frames, planning_xy, float(args.scale), title)
            if writer is None:
                height, width = mosaic.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(raw_video), fourcc, float(args.fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {raw_video}")
            writer.write(mosaic)
            summaries.append(
                {
                    "index": int(args.nuscene_start_index + index),
                    "timestamp_us": int(raw_record["timestamp_us"]),
                    "seed": step_seed,
                    "command_onehot": raw_record["command_onehot"],
                    "final_planning_xy_right_forward": planning_xy.astype(float).tolist(),
                }
            )
            print(
                f"frame={index} scene_index={args.nuscene_start_index + index} "
                f"command={raw_record['command_onehot']} final={planning_xy[-1].astype(float).tolist()}"
            )
    finally:
        if writer is not None:
            writer.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "18",
            "-preset",
            "medium",
            str(output_video),
        ],
        check=True,
    )
    raw_video.unlink()

    output_json.write_text(
        json.dumps(
            {
                "nuscene_root": str(dataroot),
                "nuscene_version": args.nuscene_version,
                "nuscene_scene": args.nuscene_scene,
                "nuscene_start_index": int(args.nuscene_start_index),
                "steps": int(args.steps),
                "seed": int(args.seed),
                "video": str(output_video),
                "frames": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_video}")
    print(f"wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
