from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

HISTORY_STEPS = 16
IMAGE_HISTORY_STEPS = 4
DEFAULT_CAMERA_SOURCES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)


@dataclass(frozen=True)
class EgoFrame:
    position: np.ndarray
    rotation: np.ndarray


class AlpamayoR1NativeAgent:
    def __init__(self, device: str) -> None:
        from rl_framework.alpamayo_r1.loader import create_alpamayo_r1

        repo_root = Path(__file__).resolve().parents[2]
        bundle = create_alpamayo_r1(
            {
                "alpamayo_root": str(repo_root / "alpamayo1"),
                "model_path": str(repo_root / "alpamayo1" / "Alpamayo-R1-10B"),
                "device": device,
            }
        )
        self.model = bundle.model
        self.processor = bundle.processor
        self.device = bundle.device
        self.camera_sources = DEFAULT_CAMERA_SOURCES
        self.top_p = 0.98
        self.temperature = 0.6
        self.max_generation_length = 256
        self._image_history: Deque[Dict[str, np.ndarray]] = deque(maxlen=HISTORY_STEPS)
        self._ego_history: Deque[EgoFrame] = deque(maxlen=HISTORY_STEPS)
        self._scene_token: str | None = None

    def reset(self) -> None:
        self._image_history.clear()
        self._ego_history.clear()
        self._scene_token = None
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def _reset_for_scene(self, scene_token: str) -> None:
        if scene_token != self._scene_token:
            self.reset()
            self._scene_token = scene_token

    def predict_action(
        self,
        obs_img: Dict[str, Any],
        obs_info: Dict[str, Any],
    ) -> np.ndarray:
        self._reset_for_scene(str(obs_info["scene_token"]))
        self._append_observation(obs_img, obs_info)
        image_frames = self._image_frames_tensor()
        ego_history_xyz, ego_history_rot = self._ego_history_tensors()

        from alpamayo_r1 import helper

        messages = helper.create_message(image_frames.flatten(0, 1))
        tokenized_data = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized_data,
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            },
            self.device,
        )

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_xyz, _ = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=self.top_p,
                temperature=self.temperature,
                num_traj_samples=1,
                max_generation_length=self.max_generation_length,
            )

        trajectory = pred_xyz[0, 0, 0].detach().float().cpu().numpy().astype(np.float32)
        if trajectory.shape != (64, 3):
            raise ValueError(f"Alpamayo R1 must predict a 64x3 trajectory, got {trajectory.shape}")
        return trajectory[:, :2]

    def _append_observation(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> None:
        frames = {name: obs_img[name] for name in self.camera_sources}
        position = obs_info["ego_pos"]
        rotation = SCR.from_euler("XYZ", obs_info["ego_rot"], degrees=False).as_matrix().astype(np.float32)
        if not self._ego_history:
            first_frame = EgoFrame(position=position, rotation=rotation)
            for _ in range(HISTORY_STEPS):
                self._image_history.append(frames)
                self._ego_history.append(first_frame)
            return
        self._image_history.append(frames)
        self._ego_history.append(EgoFrame(position=position, rotation=rotation))

    def _image_frames_tensor(self) -> torch.Tensor:
        if len(self._image_history) < IMAGE_HISTORY_STEPS:
            raise RuntimeError(
                f"Alpamayo R1 requires {IMAGE_HISTORY_STEPS} image history steps, "
                f"got {len(self._image_history)}"
            )
        frames = list(self._image_history)[-IMAGE_HISTORY_STEPS:]
        camera_frames = [
            np.stack([entry[camera_name] for entry in frames], axis=0)
            for camera_name in self.camera_sources
        ]
        images = np.stack(camera_frames, axis=0)
        return torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()

    def _ego_history_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self._ego_history) < HISTORY_STEPS:
            raise RuntimeError(
                f"Alpamayo R1 requires {HISTORY_STEPS} egomotion history steps, "
                f"got {len(self._ego_history)}"
            )
        history = list(self._ego_history)[-HISTORY_STEPS:]
        positions = np.stack([frame.position for frame in history], axis=0)
        rotations = np.stack([frame.rotation for frame in history], axis=0)
        current = history[-1]
        world_to_current = current.rotation.T
        xyz_local = (world_to_current @ (positions - current.position).T).T.astype(np.float32)
        rot_local = np.einsum("ij,tjk->tik", world_to_current, rotations).astype(np.float32)
        return (
            torch.from_numpy(xyz_local).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(rot_local).unsqueeze(0).unsqueeze(0),
        )
