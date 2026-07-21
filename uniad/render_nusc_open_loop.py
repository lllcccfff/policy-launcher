#!/usr/bin/env python3
"""Render UniAD open-loop planning on raw nuScenes images."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion

from rl_framework.common.trajectory import lidar_to_ego_front_left
from rl_framework.run_uniad_grpc_rollout import DEFAULT_UNIAD_CKPT, DEFAULT_UNIAD_CONFIG
from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4, UNIAD_IMG_NORM_CFG
from rl_framework.uniad.loader import create_uniad


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIAD_ROOT = REPO_ROOT / "UniAD_SIM"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
MOSAIC_ORDER = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
)
DEFAULT_CLIPS = (
    ("0007_4", "scene-0007", 12, 20),
    ("0008_3", "scene-0008", 20, 20),
)
TRAJ_HEIGHT_IN_LIDAR_M = -1.0
PRED_TRAJ_COLOR = (255, 64, 64)
GT_TRAJ_COLOR = (64, 255, 64)


def _load_uniad_converter(uniad_root: Path):
    root = str(uniad_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.data_converter import uniad_nuscenes_converter as converter

    return converter


def _scene_samples(nusc: Any, scene_name: str, start_index: int, steps: int) -> list[dict[str, Any]]:
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    scene = next((item for item in nusc.scene if item["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"Scene not found in nuScenes: {scene_name}")

    samples = []
    sample = nusc.get("sample", scene["first_sample_token"])
    while True:
        samples.append(sample)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])

    end_index = start_index + steps
    if end_index > len(samples):
        raise ValueError(
            f"Requested {scene_name}[{start_index}:{end_index}], but scene has {len(samples)} samples"
        )
    return samples[start_index:end_index]


def _build_converter_info(nusc: Any, nusc_can_bus: Any, converter: Any, sample: dict[str, Any]) -> dict[str, Any]:
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", lidar_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    lidar_path, _, _ = nusc.get_sample_data(lidar_token)

    can_bus = converter._get_can_bus_info(nusc, nusc_can_bus, sample).copy()
    info = {
        "lidar_path": lidar_path,
        "token": sample["token"],
        "prev": sample["prev"],
        "next": sample["next"],
        "can_bus": can_bus,
        "cams": {},
        "scene_token": sample["scene_token"],
        "lidar2ego_translation": cs_record["translation"],
        "lidar2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sample["timestamp"],
    }

    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix

    for camera in CAMERA_ORDER:
        cam_token = sample["data"][camera]
        _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = converter.obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, camera)
        cam_info.update(cam_intrinsic=np.asarray(cam_intrinsic, dtype=np.float32))
        info["cams"][camera] = cam_info

    return info


def _lidar2img_from_converter_cam(cam_info: dict[str, Any]) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64))
    lidar2cam_t = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def _command_from_future_ego(nusc: Any, sample: dict[str, Any], horizon_steps: int = 6) -> int:
    current = sample
    future = sample
    for _ in range(horizon_steps):
        if future["next"] == "":
            break
        future = nusc.get("sample", future["next"])

    cur_lidar = nusc.get("sample_data", current["data"]["LIDAR_TOP"])
    fut_lidar = nusc.get("sample_data", future["data"]["LIDAR_TOP"])
    cur_pose = nusc.get("ego_pose", cur_lidar["ego_pose_token"])
    fut_pose = nusc.get("ego_pose", fut_lidar["ego_pose_token"])

    cur_rot = Quaternion(cur_pose["rotation"]).rotation_matrix
    world_delta = np.asarray(fut_pose["translation"], dtype=np.float64) - np.asarray(cur_pose["translation"], dtype=np.float64)
    lateral_left = float(world_delta @ cur_rot[:, 1])
    if lateral_left <= -2.0:
        return 0
    if lateral_left >= 2.0:
        return 1
    return 2


def _ego_front_left_to_lidar_xy(ego_xy: np.ndarray) -> np.ndarray:
    ego_xy = np.asarray(ego_xy, dtype=np.float32)
    ego_points = np.zeros(ego_xy.shape[:-1] + (4,), dtype=np.float32)
    ego_points[..., :2] = ego_xy[..., :2]
    ego_points[..., 3] = 1.0
    lidar_points = ego_points @ np.linalg.inv(LIDAR_TO_EGO_4X4).T
    return lidar_points[..., :2].astype(np.float32)


def _gt_sdc_traj(nusc: Any, sample: dict[str, Any], steps: int = 6) -> tuple[np.ndarray, np.ndarray]:
    current_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    current_pose = nusc.get("ego_pose", current_lidar["ego_pose_token"])
    current_rot = Quaternion(current_pose["rotation"]).rotation_matrix
    current_translation = np.asarray(current_pose["translation"], dtype=np.float64)

    future = sample
    gt_ego_xy = []
    for _ in range(steps):
        if future["next"] == "":
            break
        future = nusc.get("sample", future["next"])
        future_lidar = nusc.get("sample_data", future["data"]["LIDAR_TOP"])
        future_pose = nusc.get("ego_pose", future_lidar["ego_pose_token"])
        world_delta = np.asarray(future_pose["translation"], dtype=np.float64) - current_translation
        front = float(world_delta @ current_rot[:, 0])
        left = float(world_delta @ current_rot[:, 1])
        gt_ego_xy.append([front, left])

    gt_ego_xy = np.asarray(gt_ego_xy, dtype=np.float32).reshape(-1, 2)
    gt_lidar_xy = _ego_front_left_to_lidar_xy(gt_ego_xy)
    return gt_lidar_xy, gt_ego_xy


def _prepare_model_input(info: dict[str, Any], command: int, device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    img_norm_cfg = UNIAD_IMG_NORM_CFG
    imgs = []
    raw_frames = {}
    lidar2imgs = []
    intrinsics = []
    img_shapes = []

    for camera in CAMERA_ORDER:
        cam_info = info["cams"][camera]
        image_path = Path(cam_info["data_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_bgr = cv2.resize(image_bgr, (1600, 900), interpolation=cv2.INTER_LINEAR)
        raw_frames[camera] = image_bgr

        image_norm = mmcv.imnormalize(image_bgr.astype(np.float32), img_norm_cfg["mean"], img_norm_cfg["std"], img_norm_cfg["to_rgb"])
        img_shapes.append(list(image_norm.shape))
        imgs.append(torch.from_numpy(image_norm).permute(2, 0, 1))

        lidar2img = _lidar2img_from_converter_cam(cam_info)
        lidar2imgs.append(lidar2img)
        intrinsic4 = np.eye(4, dtype=np.float32)
        intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float32)
        intrinsic4[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        intrinsics.append(intrinsic4)

    imgs_tensor = torch.stack(imgs)[None].to(device)

    l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix
    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float32)
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float32)
    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).astype(np.float32)
    l2g_t = (l2e_t @ e2g_r_mat.T + e2g_t).astype(np.float32)

    can_bus = np.asarray(info["can_bus"], dtype=np.float64).copy()
    rotation = Quaternion(info["ego2global_rotation"])
    can_bus[:3] = np.asarray(info["ego2global_translation"], dtype=np.float64)
    can_bus[3:7] = np.asarray(rotation.elements, dtype=np.float64)
    yaw = float(rotation.yaw_pitch_roll[0])
    yaw_deg = yaw / np.pi * 180.0
    if yaw_deg < 0.0:
        yaw_deg += 360.0
    can_bus[-2] = yaw_deg / 180.0 * np.pi
    can_bus[-1] = yaw_deg

    img_metas = {
        "scene_token": str(info["scene_token"]),
        "sample_idx": str(info["token"]),
        "can_bus": can_bus,
        "img_shape": img_shapes,
        "lidar2img": lidar2imgs,
        "l2g_r_mat": l2g_r_mat,
        "intrinsics": intrinsics,
    }
    data = {
        "img": [imgs_tensor],
        "img_metas": [[img_metas]],
        "l2g_t": torch.tensor(l2g_t, dtype=torch.float32, device=device)[None, :],
        "l2g_r_mat": torch.tensor(l2g_r_mat, dtype=torch.float32, device=device)[None, :, :],
        "timestamp": [torch.tensor([float(info["timestamp"]) / 1e6], dtype=torch.float64, device=device)],
        "command": [torch.tensor([int(command)], dtype=torch.long, device=device)],
    }
    return data, raw_frames


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


def _draw_single_traj(canvas: np.ndarray, traj_lidar_xy: np.ndarray, lidar2img: np.ndarray, color: tuple[int, int, int]) -> None:
    traj_lidar_xy = np.asarray(traj_lidar_xy, dtype=np.float32)
    if traj_lidar_xy.ndim != 2 or traj_lidar_xy.shape[1] < 2:
        raise ValueError(f"Unexpected trajectory shape: {traj_lidar_xy.shape}")

    traj_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), traj_lidar_xy[:, :2]], axis=0)
    points_lidar = np.concatenate(
        [traj_xy, np.full((traj_xy.shape[0], 1), TRAJ_HEIGHT_IN_LIDAR_M, dtype=np.float32)],
        axis=1,
    )
    points_lidar = _densify(points_lidar)
    uv, valid = _project_points(points_lidar, lidar2img)

    for index in range(len(uv) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        p0 = tuple(np.round(uv[index]).astype(int).tolist())
        p1 = tuple(np.round(uv[index + 1]).astype(int).tolist())
        cv2.line(canvas, p0, p1, color, 3, lineType=cv2.LINE_AA)
    for point, is_valid in zip(uv[::20], valid[::20]):
        if not is_valid:
            continue
        center = tuple(np.round(point).astype(int).tolist())
        cv2.circle(canvas, center, 5, color, -1, lineType=cv2.LINE_AA)


def _draw_traj(
    image_bgr: np.ndarray,
    plan_lidar_xy: np.ndarray,
    gt_lidar_xy: np.ndarray,
    lidar2img: np.ndarray,
) -> np.ndarray:
    plan_lidar_xy = np.asarray(plan_lidar_xy, dtype=np.float32)
    if plan_lidar_xy.ndim != 2 or plan_lidar_xy.shape[1] < 2:
        raise ValueError(f"Unexpected trajectory shape: {plan_lidar_xy.shape}")

    canvas = np.ascontiguousarray(image_bgr.copy())
    _draw_single_traj(canvas, gt_lidar_xy, lidar2img, GT_TRAJ_COLOR)
    _draw_single_traj(canvas, plan_lidar_xy, lidar2img, PRED_TRAJ_COLOR)
    return canvas


def _make_mosaic(
    raw_frames: dict[str, np.ndarray],
    infos: dict[str, dict[str, Any]],
    plan_lidar_xy: np.ndarray,
    gt_lidar_xy: np.ndarray,
    scale: float,
    title: str,
) -> np.ndarray:
    rows = []
    for row in MOSAIC_ORDER:
        images = []
        for camera in row:
            rendered = _draw_traj(
                raw_frames[camera],
                plan_lidar_xy,
                gt_lidar_xy,
                _lidar2img_from_converter_cam(infos[camera]),
            )
            if scale != 1.0:
                rendered = cv2.resize(rendered, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(rendered, camera, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            images.append(rendered)
        rows.append(np.concatenate(images, axis=1))
    mosaic = np.concatenate(rows, axis=0)
    cv2.putText(mosaic, title, (20, mosaic.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(mosaic, "pred=blue gt=green", (20, mosaic.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    return mosaic


def _parse_clip_specs(values: list[str] | None) -> list[tuple[str, str, int, int]]:
    if not values:
        return list(DEFAULT_CLIPS)
    specs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 4:
            raise ValueError(f"clip spec must be label:scene:start:steps, got {value!r}")
        label, scene_name, start, steps = parts
        specs.append((label, scene_name, int(start), int(steps)))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render UniAD open-loop planning on raw nuScenes RGB")
    parser.add_argument("--nuscene-root", type=str, default=str(REPO_ROOT / "data" / "raw" / "nuscenes"))
    parser.add_argument("--nuscene-version", type=str, default="v1.0-trainval")
    parser.add_argument(
        "--clip",
        action="append",
        help="Clip spec: label:scene:start_index:steps. Defaults to 0007_4 and 0008_3 sim-test spans.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--uniad-root", type=str, default=str(DEFAULT_UNIAD_ROOT))
    parser.add_argument("--config-path", type=str, default=DEFAULT_UNIAD_CONFIG)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_UNIAD_CKPT)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "tmp" / "uniad_nusc_open_loop"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.scale <= 0.0:
        raise ValueError(f"--scale must be positive, got {args.scale}")

    from nuscenes.nuscenes import NuScenes

    dataroot = Path(args.nuscene_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    uniad_root = Path(args.uniad_root).expanduser().resolve()
    converter = _load_uniad_converter(uniad_root)
    nusc = NuScenes(version=args.nuscene_version, dataroot=str(dataroot), verbose=True)
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    nusc_can_bus = NuScenesCanBus(dataroot=str(dataroot))

    model = create_uniad(
        {
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
            "device": args.device,
            "AD_root": str(uniad_root),
        }
    )
    model_impl = model.module if hasattr(model, "module") else model
    device = next(model_impl.parameters()).device

    all_summaries = []
    try:
        for label, scene_name, start_index, steps in _parse_clip_specs(args.clip):
            samples = _scene_samples(nusc, scene_name, start_index, steps)
            raw_video = output_dir / f"{label}_open_loop.raw.mp4"
            output_video = output_dir / f"{label}_open_loop.mp4"
            output_json = output_dir / f"{label}_open_loop.json"
            writer = None
            summaries = []
            try:
                for frame_idx, sample in enumerate(samples):
                    info = _build_converter_info(nusc, nusc_can_bus, converter, sample)
                    command = _command_from_future_ego(nusc, sample)
                    data, raw_frames = _prepare_model_input(info, command, device)
                    with torch.inference_mode():
                        results = model(return_loss=False, rescale=True, feature_extractor=False, **data)

                    plan_lidar_xy = results[0]["planning"]["result_planning"]["sdc_traj"][0].detach().cpu().numpy().astype(np.float32)
                    plan_ego_xy = lidar_to_ego_front_left(plan_lidar_xy, LIDAR_TO_EGO_4X4)
                    gt_lidar_xy, gt_ego_xy = _gt_sdc_traj(nusc, sample, steps=plan_lidar_xy.shape[0])
                    title = (
                        f"{label} {scene_name} sample={start_index + frame_idx} "
                        f"cmd={command} ts={int(sample['timestamp'])}"
                    )
                    mosaic = _make_mosaic(raw_frames, info["cams"], plan_lidar_xy, gt_lidar_xy, float(args.scale), title)
                    if writer is None:
                        height, width = mosaic.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(raw_video), fourcc, float(args.fps), (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video writer: {raw_video}")
                    writer.write(mosaic)

                    summary = {
                        "frame": frame_idx,
                        "scene_sample_index": start_index + frame_idx,
                        "sample_token": sample["token"],
                        "timestamp": int(sample["timestamp"]),
                        "command": int(command),
                        "sdc_traj_lidar_xy": plan_lidar_xy.astype(float).tolist(),
                        "sdc_traj_ego_front_left_xy": plan_ego_xy.astype(float).tolist(),
                        "gt_sdc_traj_valid_steps": int(gt_ego_xy.shape[0]),
                        "gt_sdc_traj_lidar_xy": gt_lidar_xy.astype(float).tolist(),
                        "gt_sdc_traj_ego_front_left_xy": gt_ego_xy.astype(float).tolist(),
                    }
                    summaries.append(summary)
                    gt_final = gt_ego_xy[-1].astype(float).tolist() if gt_ego_xy.shape[0] else []
                    print(
                        f"{label} frame={frame_idx} sample_index={start_index + frame_idx} "
                        f"command={command} final_ego={plan_ego_xy[-1].astype(float).tolist()} "
                        f"gt_final_ego={gt_final}"
                    )
            finally:
                if writer is not None:
                    writer.release()

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
            payload = {
                "nuscene_root": str(dataroot),
                "nuscene_version": args.nuscene_version,
                "label": label,
                "scene": scene_name,
                "start_index": start_index,
                "steps": steps,
                "video": str(output_video),
                "frames": summaries,
            }
            output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            all_summaries.append(payload)
            print(f"wrote {output_video}")
            print(f"wrote {output_json}")
    finally:
        del model

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
