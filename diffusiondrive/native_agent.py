from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from policy_launcher.common.trajectory import lidar_to_ego_front_left
from policy_launcher.diffusiondrive.dataparser import FIXED_LIDAR_TOP_TO_EGO, parse_diffusiondrive_obs


class DiffusionDriveNativeAgent:
    def __init__(self, model_or_device: torch.nn.Module | str):
        if isinstance(model_or_device, str):
            from policy_launcher.diffusiondrive.loader import create_diffusiondrive

            repo_root = Path(__file__).resolve().parents[2]
            self.diffusiondrive_model = create_diffusiondrive(
                {
                    "diffusiondrive_root": str(repo_root / "DiffusionDrive"),
                    "checkpoint_path": str(repo_root / "DiffusionDrive" / "ckpts" / "diffusiondrive_nusc_stage2.pth"),
                    "device": model_or_device,
                }
            )
        else:
            self.diffusiondrive_model = model_or_device
        self.model_device = self._infer_model_device()
        self._scene_token = None

    def _infer_model_device(self) -> torch.device:
        return next(self.diffusiondrive_model.parameters()).device

    def reset(self) -> None:
        head = self.diffusiondrive_model.head
        head.det_head.instance_bank.reset()
        head.map_head.instance_bank.reset()
        head.motion_plan_head.instance_queue.reset()
        self._scene_token = None

    def _reset_for_scene(self, scene_token: str) -> None:
        if scene_token != self._scene_token:
            self.reset()
            self._scene_token = scene_token
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

    def _batch_raw_data(self, raw_data: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        img = raw_data["img"].to(self.model_device).unsqueeze(0)
        data: Dict[str, Any] = {}
        for key, value in raw_data.items():
            if key == "img":
                continue
            if key == "img_metas":
                data[key] = [value]
            elif torch.is_tensor(value):
                data[key] = value.to(self.model_device).unsqueeze(0)
            else:
                data[key] = value
        return img, data

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> np.ndarray:
        self._reset_for_scene(str(obs_info["scene_token"]))
        raw_data = parse_diffusiondrive_obs(obs_img, obs_info)
        img, data = self._batch_raw_data(raw_data)
        with torch.inference_mode():
            output = self.diffusiondrive_model.forward(img, **data)
        planning = output[0]["img_bbox"]["final_planning"].detach().cpu().numpy().astype(np.float32)
        return lidar_to_ego_front_left(planning, FIXED_LIDAR_TOP_TO_EGO)
