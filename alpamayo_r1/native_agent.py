from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

from rl_framework.alpamayo_r1.loader import AlpamayoR1Bundle
from rl_framework.common.trajectory import traj2control


HISTORY_STEPS = 16
IMAGE_HISTORY_STEPS = 4
PREDICTION_HORIZON_S = 6.4
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


def _vec3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{name} must contain 3 values, got shape {np.asarray(value).shape}")
    return array


class AlpamayoR1NativeAgent:
    def __init__(
        self,
        bundle: AlpamayoR1Bundle,
        camera_sources: Sequence[str] = DEFAULT_CAMERA_SOURCES,
        top_p: float = 0.98,
        temperature: float = 0.6,
        max_generation_length: int = 256,
    ) -> None:
        camera_sources = tuple(str(name) for name in camera_sources)
        if len(camera_sources) != 4:
            raise ValueError(f"Alpamayo R1 requires exactly 4 camera sources, got {camera_sources}")
        if len(set(camera_sources)) != 4:
            raise ValueError(f"Alpamayo R1 requires 4 unique camera sources, got {camera_sources}")
        if camera_sources != DEFAULT_CAMERA_SOURCES:
            raise ValueError(
                "Alpamayo R1 requires canonical PhysicalAI camera order "
                f"{DEFAULT_CAMERA_SOURCES}, got {camera_sources}"
            )
        self.model = bundle.model
        self.processor = bundle.processor
        self.device = bundle.device
        self.camera_sources = camera_sources
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.max_generation_length = int(max_generation_length)
        self._image_history: Deque[Dict[str, np.ndarray]] = deque(maxlen=HISTORY_STEPS)
        self._ego_history: Deque[EgoFrame] = deque(maxlen=HISTORY_STEPS)

    def reset(self) -> None:
        self._image_history.clear()
        self._ego_history.clear()

    def predict(
        self,
        obs_img: Dict[str, Any],
        obs_info: Dict[str, Any],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
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
        acc, steer = traj2control(trajectory[:, :2], obs_info, horizon=PREDICTION_HORIZON_S)
        return trajectory, (float(steer), float(acc))

    def _append_observation(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> None:
        frames = {name: self._latest_rgb(obs_img, name) for name in set(self.camera_sources)}
        position = _vec3(obs_info["ego_pos"], "ego_pos")
        euler_xyz = _vec3(obs_info["ego_rot"], "ego_rot")
        rotation = SCR.from_euler("XYZ", euler_xyz, degrees=False).as_matrix().astype(np.float32)
        if not self._ego_history:
            first_frame = EgoFrame(position=position, rotation=rotation)
            for _ in range(HISTORY_STEPS):
                self._image_history.append(frames)
                self._ego_history.append(first_frame)
            return
        self._image_history.append(frames)
        self._ego_history.append(EgoFrame(position=position, rotation=rotation))

    @staticmethod
    def _latest_rgb(obs_img: Dict[str, Any], camera_name: str) -> np.ndarray:
        if camera_name not in obs_img:
            raise KeyError(f"Alpamayo R1 requires camera {camera_name!r}; got {sorted(obs_img)}")
        frame = np.asarray(obs_img[camera_name])
        if frame.ndim == 4:
            frame = frame[-1]
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{camera_name} image must have shape HxWx3 or TxHxWx3, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise TypeError(f"{camera_name} image must be uint8 RGB, got {frame.dtype}")
        return np.ascontiguousarray(frame.copy())

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
