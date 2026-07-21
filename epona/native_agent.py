from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

from rl_framework.common.trajectory import traj2control
from rl_framework.epona.loader import EponaBundle


class EponaNativeAgent:
    def __init__(self, bundle: EponaBundle, camera: str = "FRONT"):
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.args = bundle.args
        self.device = bundle.device
        self.camera = str(camera)
        self.condition_frames = int(self.args.condition_frames)
        self.traj_len = int(self.args.traj_len)
        self.horizon = float(self.traj_len) / float(self.args.downsample_fps)
        self._images: list[np.ndarray] = []
        self._poses: list[np.ndarray] = []
        self.last_trajectory: np.ndarray | None = None

    def reset(self) -> None:
        self._images.clear()
        self._poses.clear()
        self.last_trajectory = None

    def predict_action(
        self,
        obs_img: Dict[str, Any],
        obs_info: Dict[str, Any],
    ) -> Tuple[float, float]:
        self._append_observation(obs_img, obs_info)
        img = self._condition_images_tensor()
        rel_pose, rel_yaw = self._condition_pose_tensors()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            start_latents = self.tokenizer.encode_to_z(img)
            predict_traj, _ = self.model.step_eval(
                start_latents,
                rel_pose,
                rel_yaw,
                self_pred_traj=False,
                traj_only=True,
            )

        pred_traj = predict_traj[0, :, :2].detach().float().cpu().numpy()
        acc, steer = traj2control(pred_traj, obs_info, horizon=self.horizon)
        self.last_trajectory = pred_traj
        return float(steer), float(acc)

    def _append_observation(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> None:
        self._images.append(obs_img[self.camera])
        self._poses.append(self._ego_pose_matrix(obs_info))
        if len(self._images) > self.condition_frames:
            del self._images[:-self.condition_frames]
            del self._poses[:-self.condition_frames]

    def _condition_images_tensor(self) -> torch.Tensor:
        images = self._padded(self._images)
        resized = [self._resize_center_crop(image) for image in images]
        tensors = [torch.from_numpy(image).permute(2, 0, 1) for image in resized]
        img = torch.stack(tensors, dim=0).float() / 255.0
        img = (img - 0.5) * 2.0
        return img.unsqueeze(0).to(self.device, non_blocking=True)

    def _condition_pose_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        poses = np.stack(self._padded(self._poses), axis=0)
        poses = np.concatenate([poses[:1], poses, poses[-1:]], axis=0)
        pose_t = torch.from_numpy(poses).to(self.device).unsqueeze(0)

        rot_a = pose_t[:, :-1, :3, :3]
        rot_b = pose_t[:, 1:, :3, :3]
        trans_a = pose_t[:, :-1, :3, 3:]
        trans_b = pose_t[:, 1:, :3, 3:]
        rel_rot = torch.linalg.inv(rot_a) @ rot_b
        rel_trans = torch.linalg.inv(rot_a) @ (trans_b - trans_a)
        rel_pose = rel_trans[..., :2, 0]
        rel_yaw = torch.atan2(rel_rot[..., 1, 0], rel_rot[..., 0, 0]).unsqueeze(-1) * (180.0 / torch.pi)
        return rel_pose, rel_yaw

    def _padded(self, values: list[Any]) -> list[Any]:
        if not values:
            raise RuntimeError("Epona observation history is empty.")
        missing = self.condition_frames - len(values)
        if missing <= 0:
            return values[-self.condition_frames:]
        return [values[0]] * missing + values

    def _resize_center_crop(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = int(self.args.image_size[0]), int(self.args.image_size[1])
        h, w = image.shape[:2]
        if 2 * h < w:
            resized_w = round(w / h * target_h)
            image = cv2.resize(image, (resized_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized_h = round(h / w * target_w)
            image = cv2.resize(image, (target_w, resized_h), interpolation=cv2.INTER_CUBIC)

        h, w = image.shape[:2]
        y0 = (h - target_h) // 2
        x0 = (w - target_w) // 2
        cropped = image[y0:y0 + target_h, x0:x0 + target_w]
        return cropped

    @staticmethod
    def _ego_pose_matrix(obs_info: Dict[str, Any]) -> np.ndarray:
        pos = obs_info["ego_pos"]
        rot = obs_info["ego_rot"]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = SCR.from_euler("XYZ", rot, degrees=False).as_matrix()
        pose[:3, 3] = pos
        return pose
