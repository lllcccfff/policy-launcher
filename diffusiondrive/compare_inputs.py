#!/usr/bin/env python3
"""Compare official DiffusionDrive dataset inputs with StreetWorld sim inputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation
from rl_framework.diffusiondrive.dataparser import CAMERA_ORDER, FINAL_HW, IMAGE_AUG_MATRIX, _preprocess_image, parse_diffusiondrive_obs
from rl_framework.diffusiondrive.native_agent import _lidar_top_points_to_ego_xy
from rl_framework.run_diffusiondrive_grpc_rollout import (
    DEFAULT_DIFFUSIONDRIVE_CKPT,
    DEFAULT_DIFFUSIONDRIVE_ROOT,
    REPO_ROOT,
    _is_ordered_scenario_exhausted,
    _seed_torch,
    _sync_runtime_fields,
)

COMPARE_KEYS = (
    "img",
    "timestamp",
    "projection_mat",
    "image_wh",
    "ego_status",
    "gt_ego_fut_trajs",
    "gt_ego_fut_masks",
    "gt_ego_fut_cmd",
)
META_KEYS = ("T_global", "T_global_inv", "timestamp")
DEFAULT_PROCESSED_SCENE_DIR = str(REPO_ROOT / "data" / "processed" / "uniad_train" / "0008_3")
DEFAULT_DATA_RAW_DIR = str(REPO_ROOT / "data" / "raw")
EGO_ORIGIN_DELTA_Z = 1.19 / 2.0
NUSC_CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
NUSC_TO_STREETWORLD_CAMERA = {
    "CAM_FRONT": "FRONT",
    "CAM_FRONT_RIGHT": "FRONT_RIGHT",
    "CAM_FRONT_LEFT": "FRONT_LEFT",
    "CAM_BACK": "BACK",
    "CAM_BACK_LEFT": "BACK_LEFT",
    "CAM_BACK_RIGHT": "BACK_RIGHT",
}


def _add_diffusiondrive_root(diffusiondrive_root: str) -> Path:
    root = Path(diffusiondrive_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DiffusionDrive root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _unwrap(value: Any) -> Any:
    if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
        return _unwrap(value.data)
    return value


def _to_numpy(value: Any) -> np.ndarray:
    value = _unwrap(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (float, int, np.floating, np.integer)):
        return np.asarray(value)
    return np.asarray(value)


def _numeric_summary(value: Any) -> dict[str, Any]:
    arr = _to_numpy(value)
    flat = arr.reshape(-1) if arr.shape else arr.reshape(1)
    finite = flat[np.isfinite(flat)]
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "first_values": flat[: min(12, flat.size)].astype(float).tolist(),
    }
    if finite.size:
        summary.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        )
    return summary


def _matrix_payload(value: Any) -> dict[str, Any]:
    arr = _to_numpy(value).astype(np.float32)
    return {
        "shape": list(arr.shape),
        "values": arr.astype(float).tolist(),
    }


def _camera_params_payload(camera_params: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for camera in CAMERA_ORDER:
        params = camera_params[camera]
        payload[camera] = {
            "K": _matrix_payload(params["K"]),
            "ego2camera": _matrix_payload(params["ego2camera"]),
            "H": int(params["H"]),
            "W": int(params["W"]),
        }
    return payload


def _compare_values(left: Any, right: Any) -> dict[str, Any]:
    left_arr = _to_numpy(left).astype(np.float64)
    right_arr = _to_numpy(right).astype(np.float64)
    result: dict[str, Any] = {
        "official_shape": list(left_arr.shape),
        "sim_shape": list(right_arr.shape),
        "same_shape": left_arr.shape == right_arr.shape,
    }
    if left_arr.shape != right_arr.shape:
        return result
    diff = right_arr - left_arr
    flat = diff.reshape(-1) if diff.shape else diff.reshape(1)
    result.update(
        {
            "max_abs_diff": float(np.max(np.abs(flat))),
            "mean_abs_diff": float(np.mean(np.abs(flat))),
            "diff_first_values": flat[: min(12, flat.size)].astype(float).tolist(),
        }
    )
    return result


def _compare_camera_params(reference_records: list[dict[str, Any]], sim_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = min(len(reference_records), len(sim_records))
    comparisons = []
    for index in range(count):
        reference_params = reference_records[index]["camera_params"]
        sim_params = sim_records[index]["camera_params"]
        item = {}
        for camera in CAMERA_ORDER:
            item[camera] = {
                "K": _compare_values(reference_params[camera]["K"]["values"], sim_params[camera]["K"]["values"]),
                "ego2camera": _compare_values(
                    reference_params[camera]["ego2camera"]["values"],
                    sim_params[camera]["ego2camera"]["values"],
                ),
                "H": {
                    "reference": reference_params[camera]["H"],
                    "sim": sim_params[camera]["H"],
                    "same": reference_params[camera]["H"] == sim_params[camera]["H"],
                },
                "W": {
                    "reference": reference_params[camera]["W"],
                    "sim": sim_params[camera]["W"],
                    "same": reference_params[camera]["W"] == sim_params[camera]["W"],
                },
            }
        comparisons.append(item)
    return comparisons


def _extract_raw_fields(sample: dict[str, Any]) -> dict[str, Any]:
    raw = {key: _unwrap(sample[key]) for key in COMPARE_KEYS}
    metas = _unwrap(sample["img_metas"])
    raw["img_metas"] = {key: metas[key] for key in META_KEYS}
    return raw


def load_official_sequence(
    diffusiondrive_root: str,
    official_ann_file: str,
    official_workdir: str,
    official_start_index: int,
    steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = _add_diffusiondrive_root(diffusiondrive_root)
    ann_file = Path(official_ann_file).expanduser().resolve()
    if not ann_file.is_file():
        raise FileNotFoundError(f"Official ann_file not found: {ann_file}")
    workdir = Path(official_workdir).expanduser().resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(f"Official workdir not found: {workdir}")

    from mmcv import Config
    from mmdet.datasets import build_dataset

    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        import projects.mmdet3d_plugin  # noqa: F401

        cfg = Config.fromfile(str(root / "projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"))
        dataset_cfg = cfg.data.test.copy()
        dataset_cfg.ann_file = str(ann_file)
        dataset_cfg.test_mode = True
        dataset = build_dataset(dataset_cfg)
        samples = []
        records = []
        for offset in range(steps):
            index = official_start_index + offset
            sample = dataset[index]
            info = dataset.data_infos[index]
            samples.append(_extract_raw_fields(sample))
            records.append(
                {
                    "index": index,
                    "token": str(info["token"]),
                    "timestamp": float(info["timestamp"]) / 1e6,
                    "camera_keys": list(info["cams"].keys()),
                    "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], dtype=np.float32).tolist(),
                    "official_workdir": str(workdir),
                }
            )
    finally:
        os.chdir(cwd)
    return samples, records


def _nusc_lidar_pose(nusc: Any, sample: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    lidar_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", lidar_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    lidar2ego = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)
    ego2global = transform_matrix(pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False)
    return (ego2global @ lidar2ego).astype(np.float32), cs_record, pose_record


def _nusc_sensor2top(
    nusc: Any,
    sensor_token: str,
    l2e_t: np.ndarray,
    l2e_r_mat: np.ndarray,
    e2g_t: np.ndarray,
    e2g_r_mat: np.ndarray,
    sensor_type: str,
) -> dict[str, Any]:
    from pyquaternion import Quaternion

    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    sweep = {
        "data_path": str(nusc.get_sample_data_path(sd_rec["token"])),
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = np.asarray(sweep["sensor2ego_translation"], dtype=np.float64)
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = np.asarray(sweep["ego2global_translation"], dtype=np.float64)

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    inv_e2g_r_mat_t = np.linalg.inv(e2g_r_mat).T
    inv_l2e_r_mat_t = np.linalg.inv(l2e_r_mat).T
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (inv_e2g_r_mat_t @ inv_l2e_r_mat_t)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (inv_e2g_r_mat_t @ inv_l2e_r_mat_t)
    T -= e2g_t @ (inv_e2g_r_mat_t @ inv_l2e_r_mat_t) + l2e_t @ inv_l2e_r_mat_t
    sweep["sensor2lidar_rotation"] = R.T.astype(np.float32)
    sweep["sensor2lidar_translation"] = T.astype(np.float32)
    return sweep


def _nusc_lidar2img(cam_info: dict[str, Any]) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64))
    lidar2cam_t = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return (IMAGE_AUG_MATRIX.astype(np.float64) @ (viewpad @ lidar2cam_rt.T)).astype(np.float32)


def _nusc_future_traj_and_cmd(
    nusc: Any,
    sample: dict[str, Any],
    pose_record: dict[str, Any],
    cs_record: dict[str, Any],
    ego_fut_ts: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3), dtype=np.float64)
    ego_fut_masks = np.zeros((ego_fut_ts + 1,), dtype=np.float32)
    sample_cur = sample
    for i in range(ego_fut_ts + 1):
        lidar_sample_data = nusc.get("sample_data", sample_cur["data"]["LIDAR_TOP"])
        fut_pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        fut_cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
        ego2global = transform_matrix(
            fut_pose_record["translation"],
            Quaternion(fut_pose_record["rotation"]),
            inverse=False,
        )
        sensor2ego = transform_matrix(
            fut_cs_record["translation"],
            Quaternion(fut_cs_record["rotation"]),
            inverse=False,
        )
        ego_fut_trajs[i] = (ego2global @ sensor2ego)[:3, 3]
        ego_fut_masks[i] = 1.0
        if sample_cur["next"] == "":
            ego_fut_trajs[i + 1 :] = ego_fut_trajs[i]
            break
        sample_cur = nusc.get("sample", sample_cur["next"])

    ego_fut_trajs = ego_fut_trajs - np.asarray(pose_record["translation"], dtype=np.float64)
    rot_mat = Quaternion(pose_record["rotation"]).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs = ego_fut_trajs - np.asarray(cs_record["translation"], dtype=np.float64)
    rot_mat = Quaternion(cs_record["rotation"]).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

    if ego_fut_trajs[-1][0] >= 2.0:
        command = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif ego_fut_trajs[-1][0] <= -2.0:
        command = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        command = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
    return ego_fut_trajs[:, :2].astype(np.float32), ego_fut_masks[1:].astype(np.float32), command


def _nusc_scene_samples(nusc: Any, scene_name: str, start_index: int, steps: int) -> list[dict[str, Any]]:
    scene = next((item for item in nusc.scene if item["name"] == scene_name), None)
    if scene is None:
        raise KeyError(f"nuScenes scene not found: {scene_name}")
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    selected = samples[start_index : start_index + steps]
    if len(selected) < steps:
        raise ValueError(f"Requested {steps} raw nuScenes frames from {scene_name}[{start_index}], only {len(selected)} available")
    return selected


def load_nuscene_raw_sequence(
    nuscene_root: str,
    nuscene_version: str,
    scene_name: str,
    start_index: int,
    steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    root = Path(nuscene_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"nuScenes root not found: {root}")
    nusc = NuScenes(version=nuscene_version, dataroot=str(root), verbose=False)
    selected = _nusc_scene_samples(nusc, scene_name, start_index, steps)

    samples = []
    records = []
    image_wh = np.tile(np.array([[float(FINAL_HW[1]), float(FINAL_HW[0])]], dtype=np.float32), (len(CAMERA_ORDER), 1))
    ego_status = np.zeros((10,), dtype=np.float32)
    for offset, sample in enumerate(selected):
        lidar2global, cs_record, pose_record = _nusc_lidar_pose(nusc, sample)
        l2e_t = np.asarray(cs_record["translation"], dtype=np.float64)
        l2e_r_mat = Quaternion(cs_record["rotation"]).rotation_matrix
        e2g_t = np.asarray(pose_record["translation"], dtype=np.float64)
        e2g_r_mat = Quaternion(pose_record["rotation"]).rotation_matrix
        ego_fut_trajs, ego_fut_masks, command = _nusc_future_traj_and_cmd(nusc, sample, pose_record, cs_record)

        images = []
        projections = []
        image_paths = {}
        camera_infos = {}
        for nusc_camera in NUSC_CAMERA_ORDER:
            streetworld_camera = NUSC_TO_STREETWORLD_CAMERA[nusc_camera]
            cam_token = sample["data"][nusc_camera]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            image_path = Path(cam_path)
            if not image_path.is_absolute():
                image_path = root / image_path
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            cam_info = _nusc_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, nusc_camera)
            cam_info["cam_intrinsic"] = np.asarray(cam_intrinsic, dtype=np.float32)
            images.append(_preprocess_image(image))
            projections.append(_nusc_lidar2img(cam_info))
            image_paths[streetworld_camera] = str(image_path)
            camera_infos[streetworld_camera] = {
                "nuscenes_camera": nusc_camera,
                "sample_data_token": cam_token,
                "cam_intrinsic": np.asarray(cam_intrinsic, dtype=np.float32).astype(float).tolist(),
                "sensor2lidar_rotation": cam_info["sensor2lidar_rotation"].astype(float).tolist(),
                "sensor2lidar_translation": cam_info["sensor2lidar_translation"].astype(float).tolist(),
            }

        timestamp = float(sample["timestamp"]) / 1e6
        raw_sample = {
            "img": torch.from_numpy(np.stack(images, axis=0)).float(),
            "projection_mat": torch.from_numpy(np.stack(projections, axis=0)).float(),
            "image_wh": torch.from_numpy(image_wh.copy()).float(),
            "timestamp": torch.tensor(timestamp, dtype=torch.float32),
            "T_global": torch.from_numpy(lidar2global).float(),
            "T_global_inv": torch.from_numpy(np.linalg.inv(lidar2global).astype(np.float32)).float(),
            "img_metas": {
                "T_global": lidar2global,
                "T_global_inv": np.linalg.inv(lidar2global).astype(np.float32),
                "timestamp": timestamp,
            },
            "ego_status": torch.from_numpy(ego_status.copy()).float(),
            "gt_ego_fut_cmd": torch.from_numpy(command).float(),
            "gt_ego_fut_trajs": torch.from_numpy(ego_fut_trajs).float(),
            "gt_ego_fut_masks": torch.from_numpy(ego_fut_masks).float(),
        }
        samples.append(raw_sample)
        records.append(
            {
                "source": "nuscene_raw",
                "rgb_source": "nuscenes_samples",
                "scene_name": scene_name,
                "scene_offset": int(start_index + offset),
                "token": str(sample["token"]),
                "timestamp": timestamp,
                "timestamp_us": int(sample["timestamp"]),
                "command": int(np.argmax(command)),
                "command_onehot": command.astype(float).tolist(),
                "ego_status_source": "official_converter_zero_vector_without_can_bus",
                "lidar2ego_translation": np.asarray(cs_record["translation"], dtype=np.float32).astype(float).tolist(),
                "ego2global_translation": np.asarray(pose_record["translation"], dtype=np.float32).astype(float).tolist(),
                "image_paths": image_paths,
                "camera_infos": camera_infos,
            }
        )
    return samples, records


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _processed_timestamps(scene_dir: Path) -> list[int]:
    by_camera = []
    for camera in CAMERA_ORDER:
        pattern = re.compile(rf"^{re.escape(camera)}_(\d+)\.jpg$")
        timestamps = {
            int(match.group(1))
            for path in (scene_dir / "camera" / "rgb").iterdir()
            if (match := pattern.match(path.name)) is not None
        }
        if not timestamps:
            raise FileNotFoundError(f"No RGB images for {camera} under {scene_dir / 'camera' / 'rgb'}")
        by_camera.append(timestamps)
    return sorted(set.intersection(*by_camera))


def _nearest_timestamp(timestamps: list[int], target: int) -> int:
    return min(timestamps, key=lambda ts: abs(ts - target))


def _processed_images(scene_dir: Path, timestamp: int) -> dict[str, np.ndarray]:
    images = {}
    for camera in CAMERA_ORDER:
        path = scene_dir / "camera" / "rgb" / f"{camera}_{timestamp}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        images[camera] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return images


def _k_from_intrinsics(intrinsics: dict[str, Any], camera: str) -> np.ndarray:
    item = intrinsics[camera]
    return np.array(
        [
            [float(item["fx"]), 0.0, float(item["cx"])],
            [0.0, float(item["fy"]), float(item["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _processed_camera_params(scene_dir: Path, reference_timestamp: int) -> dict[str, dict[str, Any]]:
    intrinsics = _read_json(scene_dir / "camera" / "intri.json")
    extrinsics = _read_json(scene_dir / "camera" / "extri.json")
    ego_poses = _read_json(scene_dir / "ego_pose.json")
    ego_origin_delta = np.eye(4, dtype=np.float32)
    ego_origin_delta[2, 3] = EGO_ORIGIN_DELTA_Z
    params = {}
    for camera in CAMERA_ORDER:
        raw_timestamps = sorted(int(ts) for ts in extrinsics[camera].keys())
        ts = _nearest_timestamp(raw_timestamps, reference_timestamp)
        world2camera = np.asarray(extrinsics[camera][str(ts)], dtype=np.float32)
        ego2world = np.asarray(ego_poses[str(ts)], dtype=np.float32)
        ego2camera = (world2camera @ ego2world @ ego_origin_delta).astype(np.float32)
        k = _k_from_intrinsics(intrinsics, camera)
        params[camera] = {
            "K": k,
            "H": int(intrinsics[camera]["h"]),
            "W": int(intrinsics[camera]["w"]),
            "ego2camera": ego2camera,
        }
    return params


def _read_scene_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Scene config must be a mapping: {path}")
    return config


def _resolve_config_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve() if (config_path.parent / path).exists() else (REPO_ROOT / path).resolve()


def _colmap_camera_num_params(model_id: int) -> int:
    counts = {
        0: 3,
        1: 4,
        2: 4,
        3: 5,
        4: 8,
        5: 8,
        6: 12,
        7: 5,
        8: 4,
        9: 5,
        10: 12,
    }
    if model_id not in counts:
        raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
    return counts[model_id]


def _colmap_k(model_id: int, params: tuple[float, ...]) -> np.ndarray:
    if model_id == 0:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model_id in (1, 4, 5, 6):
        fx, fy, cx, cy = params[:4]
    elif model_id in (2, 3, 7, 8, 9):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _read_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras = {}
    with path.open("rb") as handle:
        camera_count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(camera_count):
            camera_id, model_id = struct.unpack("<ii", handle.read(8))
            width, height = struct.unpack("<QQ", handle.read(16))
            num_params = _colmap_camera_num_params(model_id)
            params = struct.unpack("<" + "d" * num_params, handle.read(8 * num_params))
            cameras[int(camera_id)] = {
                "K": _colmap_k(int(model_id), params),
                "H": int(height),
                "W": int(width),
            }
    return cameras


def _qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * w * z, 2.0 * z * x + 2.0 * w * y],
            [2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * w * x],
            [2.0 * z * x - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float32,
    )


def _read_null_terminated(handle: Any) -> str:
    data = bytearray()
    while True:
        char = handle.read(1)
        if char == b"":
            raise EOFError("Unexpected EOF while reading COLMAP image name")
        if char == b"\x00":
            return data.decode("utf-8")
        data.extend(char)


def _read_colmap_images(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    images: dict[str, dict[int, dict[str, Any]]] = {}
    pattern = re.compile(r"^(.+)_(\d+)$")
    with path.open("rb") as handle:
        image_count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(image_count):
            _ = struct.unpack("<i", handle.read(4))[0]
            qvec = struct.unpack("<dddd", handle.read(32))
            tvec = struct.unpack("<ddd", handle.read(24))
            camera_id = struct.unpack("<i", handle.read(4))[0]
            name = _read_null_terminated(handle)
            point_count = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(point_count * 24, os.SEEK_CUR)

            match = pattern.match(Path(name).stem)
            if match is None:
                raise ValueError(f"COLMAP image name must end with _timestamp: {name}")
            camera_name = match.group(1)
            timestamp = int(match.group(2))
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = _qvec_to_rotmat(qvec)
            w2c[:3, 3] = np.asarray(tvec, dtype=np.float32)
            images.setdefault(camera_name, {})[timestamp] = {
                "camera_id": int(camera_id),
                "w2c": w2c,
                "name": name,
            }
    return images


def _colmap_camera_params(
    scene_dir: Path,
    colmap_path: Path,
    timestamp_range: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    cameras = _read_colmap_cameras(colmap_path / "cameras.bin")
    images = _read_colmap_images(colmap_path / "images.bin")
    raw_ego_poses = _read_json(scene_dir / "ego_pose.json")
    ego_origin_delta = np.eye(4, dtype=np.float32)
    ego_origin_delta[2, 3] = EGO_ORIGIN_DELTA_Z

    params = {}
    start, end = timestamp_range
    for camera in CAMERA_ORDER:
        if camera not in images:
            raise KeyError(f"COLMAP images missing camera {camera}")
        timestamps = sorted(ts for ts in images[camera].keys() if start <= ts < end)
        if not timestamps:
            raise KeyError(f"COLMAP images for {camera} have no timestamps in range {timestamp_range}")
        timestamp = timestamps[0]
        if str(timestamp) not in raw_ego_poses:
            raise KeyError(f"Missing ego pose for COLMAP timestamp {timestamp}")
        image = images[camera][timestamp]
        camera_info = cameras[image["camera_id"]]
        ego2world = np.asarray(raw_ego_poses[str(timestamp)], dtype=np.float32)
        params[camera] = {
            "K": camera_info["K"],
            "H": camera_info["H"],
            "W": camera_info["W"],
            "ego2camera": (image["w2c"] @ ego2world @ ego_origin_delta).astype(np.float32),
        }
    return params


def _pose_heading(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def _velocity_and_yaw_rate(poses: dict[int, np.ndarray], timestamps: list[int], index: int) -> tuple[np.ndarray, np.ndarray]:
    if index == 0:
        prev_index = 0
        curr_index = 1
    else:
        prev_index = index - 1
        curr_index = index
    prev_ts = timestamps[prev_index]
    curr_ts = timestamps[curr_index]
    dt = (curr_ts - prev_ts) * 1e-6
    if dt <= 0:
        raise ValueError(f"Non-positive timestamp delta: prev={prev_ts}, curr={curr_ts}")
    prev_pose = poses[prev_ts]
    curr_pose = poses[curr_ts]
    velocity = np.zeros(3, dtype=np.float32)
    velocity[:3] = (curr_pose[:3, 3] - prev_pose[:3, 3]) / dt
    prev_heading = _pose_heading(prev_pose)
    curr_heading = _pose_heading(curr_pose)
    delta_heading = (curr_heading - prev_heading + np.pi) % (2.0 * np.pi) - np.pi
    angular_velocity = np.zeros(3, dtype=np.float32)
    angular_velocity[2] = float(delta_heading / dt)
    return velocity, angular_velocity


def _acceleration(poses: dict[int, np.ndarray], timestamps: list[int], index: int) -> np.ndarray:
    if len(timestamps) < 3:
        return np.zeros(3, dtype=np.float32)
    prev_index = max(index - 1, 0)
    next_index = min(index + 1, len(timestamps) - 1)
    if prev_index == index:
        prev_vel, _ = _velocity_and_yaw_rate(poses, timestamps, index)
        next_vel, _ = _velocity_and_yaw_rate(poses, timestamps, index + 1)
        dt = (timestamps[index + 1] - timestamps[index]) * 1e-6
    elif next_index == index:
        prev_vel, _ = _velocity_and_yaw_rate(poses, timestamps, index - 1)
        next_vel, _ = _velocity_and_yaw_rate(poses, timestamps, index)
        dt = (timestamps[index] - timestamps[index - 1]) * 1e-6
    else:
        prev_vel, _ = _velocity_and_yaw_rate(poses, timestamps, index)
        next_vel, _ = _velocity_and_yaw_rate(poses, timestamps, index + 1)
        dt = (timestamps[index + 1] - timestamps[index - 1]) * 1e-6
    if dt <= 0:
        raise ValueError(f"Non-positive acceleration dt at index={index}")
    return ((next_vel - prev_vel) / dt).astype(np.float32)


def _processed_command(poses: dict[int, np.ndarray], timestamps: list[int], index: int, horizon_steps: int = 6) -> int:
    target_index = min(index + horizon_steps, len(timestamps) - 1)
    current_pose = poses[timestamps[index]]
    future_pose = poses[timestamps[target_index]]
    world_delta = future_pose[:3, 3] - current_pose[:3, 3]
    left = float(world_delta @ current_pose[:3, 1])
    if left <= -2.0:
        return 0
    if left >= 2.0:
        return 1
    return 2


def load_processed_sequence(
    processed_scene_dir: str,
    start_timestamp: int | None,
    steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    scene_dir = Path(processed_scene_dir).expanduser().resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Processed scene dir not found: {scene_dir}")

    timestamps = _processed_timestamps(scene_dir)
    first_timestamp = _nearest_timestamp(timestamps, int(start_timestamp)) if start_timestamp is not None else timestamps[0]
    first_index = timestamps.index(first_timestamp)
    selected = timestamps[first_index : first_index + steps]
    if len(selected) < steps:
        raise ValueError(f"Requested {steps} processed frames from {first_timestamp}, only {len(selected)} available")

    raw_ego_poses = _read_json(scene_dir / "ego_pose.json")
    ego_origin_delta = np.eye(4, dtype=np.float32)
    ego_origin_delta[2, 3] = EGO_ORIGIN_DELTA_Z
    poses = {
        int(ts): (np.asarray(pose, dtype=np.float32) @ ego_origin_delta).astype(np.float32)
        for ts, pose in raw_ego_poses.items()
    }
    pose_timestamps = sorted(ts for ts in timestamps if ts in poses)
    if pose_timestamps != timestamps:
        missing = sorted(set(timestamps) - set(pose_timestamps))
        raise KeyError(f"Missing ego poses for processed image timestamps: {missing[:10]}")

    camera_params = _processed_camera_params(scene_dir, selected[0])
    samples = []
    records = []
    for timestamp in selected:
        index = timestamps.index(timestamp)
        pose = poses[timestamp]
        linear_velocity, angular_velocity = _velocity_and_yaw_rate(poses, timestamps, index)
        acceleration = _acceleration(poses, timestamps, index)
        obs_info = {
            "scene_token": scene_dir.name,
            "cam_params": camera_params,
            "ego_rot": np.array([0.0, 0.0, _pose_heading(pose)], dtype=np.float32),
            "heading_theta": _pose_heading(pose),
            "accelerate": acceleration,
            "linear_acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "linear_velocity": linear_velocity,
            "ego_steer": 0.0,
            "ego_pos": pose[:3, 3].astype(np.float32),
            "current_timestamp": float(timestamp) * 1e-6,
            "command": _processed_command(poses, timestamps, index),
        }
        samples.append(parse_diffusiondrive_obs(_processed_images(scene_dir, timestamp), obs_info))
        records.append(
            {
                "timestamp_us": int(timestamp),
                "timestamp": float(timestamp) * 1e-6,
                "processed_scene_dir": str(scene_dir),
                "command": int(obs_info["command"]),
                "ego_pos": obs_info["ego_pos"].astype(float).tolist(),
                "heading_theta": float(obs_info["heading_theta"]),
                "camera_params": _camera_params_payload(obs_info["cam_params"]),
            }
        )
    return samples, records, selected


def load_scene_config_sequence(
    scene_config: str,
    processed_scene_dir: str,
    start_timestamp: int | None,
    steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    scene_dir = Path(processed_scene_dir).expanduser().resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Processed scene dir not found: {scene_dir}")
    config_path = Path(scene_config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Scene config not found: {config_path}")
    scene_cfg = _read_scene_config(config_path)
    dataset_cfg = scene_cfg["dataloader_cfg"]["dataset_cfg"]
    timestamp_range = (int(dataset_cfg["timestamp_range"][0]), int(dataset_cfg["timestamp_range"][1]))
    colmap_path = _resolve_config_path(config_path, str(dataset_cfg["colmap_path"]))

    raw_ego_poses = _read_json(scene_dir / "ego_pose.json")
    ego_origin_delta = np.eye(4, dtype=np.float32)
    ego_origin_delta[2, 3] = EGO_ORIGIN_DELTA_Z - float(scene_cfg.get("ego_center_height", 0))
    poses = {
        int(timestamp): (np.asarray(pose, dtype=np.float32) @ ego_origin_delta).astype(np.float32)
        for timestamp, pose in raw_ego_poses.items()
        if timestamp_range[0] <= int(timestamp) < timestamp_range[1]
    }
    camera_params = _colmap_camera_params(scene_dir, colmap_path, timestamp_range)
    timestamps = sorted(poses.keys())
    first_timestamp = _nearest_timestamp(timestamps, int(start_timestamp)) if start_timestamp is not None else timestamps[0]
    first_index = timestamps.index(first_timestamp)
    selected = timestamps[first_index : first_index + steps]
    if len(selected) < steps:
        raise ValueError(f"Requested {steps} scene-config frames from {first_timestamp}, only {len(selected)} available")

    samples = []
    records = []
    for timestamp in selected:
        index = timestamps.index(timestamp)
        pose = poses[timestamp]
        linear_velocity, angular_velocity = _velocity_and_yaw_rate(poses, timestamps, index)
        acceleration = _acceleration(poses, timestamps, index)
        obs_info = {
            "scene_token": str(dataset_cfg["scene_name"]),
            "cam_params": camera_params,
            "ego_rot": np.array([0.0, 0.0, _pose_heading(pose)], dtype=np.float32),
            "heading_theta": _pose_heading(pose),
            "accelerate": acceleration,
            "linear_acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "linear_velocity": linear_velocity,
            "ego_steer": 0.0,
            "ego_pos": pose[:3, 3].astype(np.float32),
            "current_timestamp": float(timestamp) * 1e-6,
            "command": _processed_command(poses, timestamps, index),
        }
        samples.append(parse_diffusiondrive_obs(_processed_images(scene_dir, timestamp), obs_info))
        records.append(
            {
                "timestamp_us": int(timestamp),
                "timestamp": float(timestamp) * 1e-6,
                "processed_scene_dir": str(scene_dir),
                "scene_config": str(config_path),
                "colmap_path": str(colmap_path),
                "scene_name": str(dataset_cfg["scene_name"]),
                "command": int(obs_info["command"]),
                "ego_pos": obs_info["ego_pos"].astype(float).tolist(),
                "heading_theta": float(obs_info["heading_theta"]),
                "camera_params": _camera_params_payload(obs_info["cam_params"]),
            }
        )
    return samples, records, selected


def load_sim_sequence(
    host: str,
    port: int,
    timeout: float,
    scene_name: str,
    steps: int,
    diffusiondrive_root: str,
    checkpoint_path: str,
    device: str,
    seed: int,
    target_timestamps_us: list[int] | None = None,
    timestamp_tolerance_us: int = 50000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.diffusiondrive.loader import create_diffusiondrive
    from rl_framework.diffusiondrive.native_agent import DiffusionDriveNativeAgent

    _seed_torch(seed)
    model = create_diffusiondrive(
        {
            "diffusiondrive_root": diffusiondrive_root,
            "checkpoint_path": checkpoint_path,
            "device": device,
        }
    )
    agent = DiffusionDriveNativeAgent(model)

    env = GrpcClientEnv(host=host, port=port, timeout_sec=timeout, auto_wait_ready=True)
    try:
        seen_scenes: set[str] = set()
        while True:
            try:
                obs, reset_info = env.reset()
            except Exception as exc:
                if _is_ordered_scenario_exhausted(exc):
                    raise RuntimeError(f"Scene {scene_name!r} was not found before ordered scenario exhaustion.") from exc
                raise
            current_scene = str(reset_info.get("scene_name", "scene"))
            if current_scene in seen_scenes:
                raise RuntimeError(f"Scene {scene_name!r} was not found before scenario order repeated.")
            seen_scenes.add(current_scene)
            if current_scene == scene_name:
                break

        agent.reset()
        obs_img, obs_info, _, _ = unpack_ad_observation(obs)
        _sync_runtime_fields(obs_info, reset_info)
        samples = []
        records = []
        step = 0

        def append_current(target_timestamp_us: int | None) -> None:
            raw_data = agent.prepare_raw_input(obs_img, obs_info)
            samples.append(raw_data)
            records.append(
                {
                    "step": step,
                    "scene_name": current_scene,
                    "timestamp": float(obs_info["current_timestamp"]),
                    "timestamp_us": int(round(float(obs_info["current_timestamp"]) * 1e6)),
                    "target_timestamp_us": target_timestamp_us,
                    "command": int(obs_info["command"]),
                    "camera_keys": list(CAMERA_ORDER),
                    "camera_params": _camera_params_payload(obs_info["cam_params"]),
                }
            )

        def advance_once() -> None:
            nonlocal obs, obs_img, obs_info, step
            raw_data = agent.prepare_raw_input(obs_img, obs_info)
            pred_traj = agent.predict_traj_from_raw_data(raw_data)
            action = agent.predict_action_from_traj(pred_traj, obs_info)
            obs, _, terminated, truncated, step_info = env.step(action)
            step += 1
            if terminated or truncated:
                raise RuntimeError(f"Simulation terminated before requested compare inputs: terminated={terminated}, truncated={truncated}, step={step}")
            obs_img, obs_info, _, _ = unpack_ad_observation(obs)
            _sync_runtime_fields(obs_info, step_info)

        if target_timestamps_us is None:
            for _ in range(steps):
                append_current(None)
                if len(samples) < steps:
                    advance_once()
        else:
            for target_timestamp_us in target_timestamps_us:
                while int(round(float(obs_info["current_timestamp"]) * 1e6)) < int(target_timestamp_us) - timestamp_tolerance_us:
                    advance_once()
                current_timestamp_us = int(round(float(obs_info["current_timestamp"]) * 1e6))
                if abs(current_timestamp_us - int(target_timestamp_us)) > timestamp_tolerance_us:
                    raise RuntimeError(
                        f"Could not align sim timestamp to target: current={current_timestamp_us}, "
                        f"target={target_timestamp_us}, tolerance={timestamp_tolerance_us}"
                    )
                append_current(int(target_timestamp_us))
    finally:
        env.close()
    return samples, records


def summarize_sequence(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for sample in samples:
        entry = {key: _numeric_summary(sample[key]) for key in COMPARE_KEYS}
        entry["img_metas"] = {key: _numeric_summary(sample["img_metas"][key]) for key in META_KEYS}
        summaries.append(entry)
    return summaries


def compare_sequences(official: list[dict[str, Any]], sim: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = min(len(official), len(sim))
    comparisons = []
    for index in range(count):
        item = {key: _compare_values(official[index][key], sim[index][key]) for key in COMPARE_KEYS}
        item["img_metas"] = {
            key: _compare_values(official[index]["img_metas"][key], sim[index]["img_metas"][key]) for key in META_KEYS
        }
        comparisons.append(item)
    return comparisons


def _trajectory_payload(value: Any) -> dict[str, Any]:
    arr = _to_numpy(value).astype(np.float32)
    return {
        "shape": list(arr.shape),
        "values": arr.astype(float).tolist(),
        "final_point": arr[-1].astype(float).tolist() if arr.ndim >= 2 and arr.shape[0] else [],
    }


def run_inference_sequence(
    samples: list[dict[str, Any]],
    diffusiondrive_model: torch.nn.Module,
    seed: int,
) -> list[dict[str, Any]]:
    from rl_framework.diffusiondrive.native_agent import DiffusionDriveNativeAgent

    agent = DiffusionDriveNativeAgent(diffusiondrive_model)
    agent.reset()
    outputs = []
    for index, raw_data in enumerate(samples):
        step_seed = int(seed) + index
        _seed_torch(step_seed)
        img, data = agent._batch_raw_data(raw_data)
        with torch.inference_mode():
            output = diffusiondrive_model.forward(img, **data)
        raw_planning = output[0]["img_bbox"]["final_planning"].detach().cpu().numpy().astype(np.float32)
        if raw_planning.ndim != 2 or raw_planning.shape[1] != 2:
            raise ValueError(f"DiffusionDrive final_planning must have shape Nx2, got {raw_planning.shape}")
        front_left_planning = _lidar_top_points_to_ego_xy(raw_planning)
        outputs.append(
            {
                "step": index,
                "seed": step_seed,
                "timestamp": float(_to_numpy(raw_data["timestamp"])),
                "command_onehot": _to_numpy(raw_data["gt_ego_fut_cmd"]).astype(float).tolist(),
                "ego_status": _to_numpy(raw_data["ego_status"]).astype(float).tolist(),
                "raw_final_planning_xy_right_forward": _trajectory_payload(raw_planning),
                "converted_final_planning_x_front_y_left": _trajectory_payload(front_left_planning),
            }
        )
    return outputs


def compare_inference_outputs(reference: list[dict[str, Any]], sim: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = min(len(reference), len(sim))
    comparisons = []
    for index in range(count):
        comparisons.append(
            {
                "step": index,
                "reference_seed": reference[index]["seed"],
                "sim_seed": sim[index]["seed"],
                "same_seed": reference[index]["seed"] == sim[index]["seed"],
                "command_onehot": _compare_values(reference[index]["command_onehot"], sim[index]["command_onehot"]),
                "ego_status": _compare_values(reference[index]["ego_status"], sim[index]["ego_status"]),
                "raw_final_planning_xy_right_forward": _compare_values(
                    reference[index]["raw_final_planning_xy_right_forward"]["values"],
                    sim[index]["raw_final_planning_xy_right_forward"]["values"],
                ),
                "converted_final_planning_x_front_y_left": _compare_values(
                    reference[index]["converted_final_planning_x_front_y_left"]["values"],
                    sim[index]["converted_final_planning_x_front_y_left"]["values"],
                ),
            }
        )
    return comparisons


def inspect_data_raw(data_raw_dir: str) -> dict[str, Any]:
    root = Path(data_raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data/raw dir not found: {root}")
    image_files = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
    pkl_files = list(root.rglob("*.pkl"))
    nuscenes_root = root / "nuscenes"
    return {
        "path": str(root),
        "exists": True,
        "has_nuscenes_dir": nuscenes_root.is_dir(),
        "pkl_files": [str(path) for path in pkl_files[:20]],
        "pkl_count": len(pkl_files),
        "image_count": len(image_files),
        "top_level_dirs": sorted(path.name for path in root.iterdir() if path.is_dir()),
        "liauto_front_image_shape": _image_shape(root / "liauto_0" / "inputs" / "images" / "cam1" / "000000_00_cropped.png"),
    }


def _image_shape(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    image = Image.open(path)
    width, height = image.size
    return [height, width, len(image.getbands())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare official DiffusionDrive inputs with StreetWorld sim inputs")
    parser.add_argument("--official-ann-file", type=str, default=None, help="Official DiffusionDrive nuscenes_infos_*.pkl")
    parser.add_argument(
        "--official-workdir",
        type=str,
        default=DEFAULT_DIFFUSIONDRIVE_ROOT,
        help="Working directory used to resolve relative image paths in official metadata",
    )
    parser.add_argument("--official-start-index", type=int, default=None, help="First official dataset index to compare")
    parser.add_argument("--nuscene-root", type=str, default=str(REPO_ROOT / "data" / "raw" / "nuscenes"), help="nuScenes raw dataroot")
    parser.add_argument("--nuscene-version", type=str, default="v1.0-trainval", help="nuScenes version under --nuscene-root")
    parser.add_argument("--nuscene-scene", type=str, default=None, help="nuScenes raw scene name, for example scene-0008")
    parser.add_argument("--nuscene-start-index", type=int, default=0, help="First raw nuScenes sample index inside --nuscene-scene")
    parser.add_argument("--processed-scene-dir", type=str, default=None, help="Direct processed scene directory to compare")
    parser.add_argument("--scene-config", type=str, default=None, help="Scene config for COLMAP-based direct processed loading")
    parser.add_argument("--processed-start-timestamp", type=int, default=None, help="First processed timestamp in microseconds")
    parser.add_argument("--data-raw-dir", type=str, default=DEFAULT_DATA_RAW_DIR, help="Raw data directory to inspect")
    parser.add_argument("--sim-scene-name", type=str, default="0008_3", help="StreetWorld ordered scenario name")
    parser.add_argument("--steps", type=int, default=8, help="Number of sequential inputs to compare")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--timeout", type=float, default=360.0, help="gRPC timeout seconds")
    parser.add_argument("--device", type=str, default="cuda", help="DiffusionDrive inference device")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed for deterministic DiffusionDrive sampling")
    parser.add_argument("--compare-inference", action="store_true", help="Run DiffusionDrive forward on both collected input sequences")
    parser.add_argument("--diffusiondrive-root", type=str, default=DEFAULT_DIFFUSIONDRIVE_ROOT, help="DiffusionDrive checkout root")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_DIFFUSIONDRIVE_CKPT, help="DiffusionDrive checkpoint path")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "outputs" / "diffusiondrive" / "input_compare.json"),
        help="Comparison report path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")

    if args.scene_config is not None and args.processed_scene_dir is None:
        raise ValueError("--scene-config requires --processed-scene-dir")
    if args.nuscene_scene is not None and args.official_ann_file is not None:
        raise ValueError("Use either --nuscene-scene or --official-ann-file, not both")
    if args.nuscene_scene is None:
        if args.processed_scene_dir is not None and args.official_ann_file is not None:
            raise ValueError("Use either --processed-scene-dir or --official-ann-file, not both")
        if args.processed_scene_dir is None and args.official_ann_file is None:
            raise ValueError("One of --nuscene-scene, --processed-scene-dir, or --official-ann-file is required")
    if args.nuscene_start_index < 0:
        raise ValueError(f"--nuscene-start-index must be non-negative, got {args.nuscene_start_index}")

    target_timestamps_us = None
    sim_target_records = None
    source_name = "official"
    if args.nuscene_scene is not None:
        official_samples, official_records = load_nuscene_raw_sequence(
            args.nuscene_root,
            args.nuscene_version,
            args.nuscene_scene,
            args.nuscene_start_index,
            args.steps,
        )
        source_name = "nuscene_raw"
        if args.processed_scene_dir is not None:
            if args.scene_config is not None:
                _, sim_target_records, target_timestamps_us = load_scene_config_sequence(
                    args.scene_config,
                    args.processed_scene_dir,
                    args.processed_start_timestamp,
                    args.steps,
                )
            else:
                _, sim_target_records, target_timestamps_us = load_processed_sequence(
                    args.processed_scene_dir,
                    args.processed_start_timestamp,
                    args.steps,
                )
    elif args.processed_scene_dir is not None:
        if args.scene_config is not None:
            official_samples, official_records, target_timestamps_us = load_scene_config_sequence(
                args.scene_config,
                args.processed_scene_dir,
                args.processed_start_timestamp,
                args.steps,
            )
            source_name = "scene_config"
        else:
            official_samples, official_records, target_timestamps_us = load_processed_sequence(
                args.processed_scene_dir,
                args.processed_start_timestamp,
                args.steps,
            )
            source_name = "processed"
    else:
        if args.official_start_index is None:
            raise ValueError("--official-start-index is required with --official-ann-file")
        official_samples, official_records = load_official_sequence(
            args.diffusiondrive_root,
            args.official_ann_file,
            args.official_workdir,
            args.official_start_index,
            args.steps,
        )

    sim_samples, sim_records = load_sim_sequence(
        args.host,
        args.port,
        args.timeout,
        args.sim_scene_name,
        args.steps,
        args.diffusiondrive_root,
        args.checkpoint_path,
        args.device,
        args.seed,
        target_timestamps_us=target_timestamps_us,
    )

    report = {
        "source": source_name,
        "seed": int(args.seed),
        "inference_seed_rule": "seed + step" if args.compare_inference else None,
        "data_raw": inspect_data_raw(args.data_raw_dir),
        "reference_records": official_records,
        "sim_records": sim_records,
        "reference_summary": summarize_sequence(official_samples),
        "sim_summary": summarize_sequence(sim_samples),
        "comparisons": compare_sequences(official_samples, sim_samples),
    }
    if sim_target_records is not None:
        report["sim_target_records"] = sim_target_records
    if official_records and "camera_params" in official_records[0]:
        report["camera_param_comparisons"] = _compare_camera_params(official_records, sim_records)
    if args.compare_inference:
        from rl_framework.diffusiondrive.loader import create_diffusiondrive

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = create_diffusiondrive(
            {
                "diffusiondrive_root": args.diffusiondrive_root,
                "checkpoint_path": args.checkpoint_path,
                "device": args.device,
            }
        )
        try:
            reference_inference = run_inference_sequence(official_samples, model, args.seed)
            sim_inference = run_inference_sequence(sim_samples, model, args.seed)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["reference_inference"] = reference_inference
        report["sim_inference"] = sim_inference
        report["inference_comparisons"] = compare_inference_outputs(reference_inference, sim_inference)

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for index, comparison in enumerate(report["comparisons"]):
        cmd_reference = _to_numpy(official_samples[index]["gt_ego_fut_cmd"]).astype(float).tolist()
        cmd_sim = _to_numpy(sim_samples[index]["gt_ego_fut_cmd"]).astype(float).tolist()
        ego_reference = _to_numpy(official_samples[index]["ego_status"]).astype(float).tolist()
        ego_sim = _to_numpy(sim_samples[index]["ego_status"]).astype(float).tolist()
        img_mean_diff = comparison["img"].get("mean_abs_diff")
        proj_mean_diff = comparison["projection_mat"].get("mean_abs_diff")
        line = (
            f"step={index} reference_cmd={cmd_reference} sim_cmd={cmd_sim} "
            f"reference_ego_status={ego_reference} sim_ego_status={ego_sim} "
            f"img_mean_abs_diff={img_mean_diff} projection_mean_abs_diff={proj_mean_diff}"
        )
        if args.compare_inference:
            reference_final = report["reference_inference"][index]["converted_final_planning_x_front_y_left"]["final_point"]
            sim_final = report["sim_inference"][index]["converted_final_planning_x_front_y_left"]["final_point"]
            line += f" reference_final_x_front_y_left={reference_final} sim_final_x_front_y_left={sim_final}"
        print(line)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
