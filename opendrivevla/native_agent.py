from __future__ import annotations

import copy
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

from rl_framework.common.tensor import to_device
from rl_framework.common.trajectory import lidar_to_ego_front_left, traj2control
from rl_framework.uniad.dataparser import LIDAR_TO_EGO_4X4, UNIAD_IMG_NORM_CFG, parse_uniad_obs

FIXED_CAMERAS = ("FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT")
HISTORY_POINTS = 4
HISTORY_INTERVAL_S = 0.5


@dataclass
class EgoFrame:
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    timestamp: float

def _command_to_goal(command: int) -> str:
    if command == 0:
        return "turn right"
    if command == 1:
        return "turn left"
    if command == 2:
        return "keep forward"
    raise ValueError(f"OpenDriveVLA command must be 0/1/2, got {command}")


def _strict_parse_traj(answer: str) -> np.ndarray:
    match = re.search(r"<traj_start>(.*?)<traj_end>", answer, flags=re.DOTALL)
    text = match.group(1) if match is not None else answer
    pairs = re.findall(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", text)
    if len(pairs) != 6:
        raise ValueError(f"OpenDriveVLA output must contain exactly 6 waypoints, got {len(pairs)}: {answer!r}")
    return np.array([(float(x), float(y)) for x, y in pairs])


class OpenDriveVLANativeAgent:
    def __init__(
        self,
        tokenizer: Any,
        model: torch.nn.Module,
        cameras: Sequence[str] = FIXED_CAMERAS,
        max_new_tokens: int = 512,
        bf16: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.cameras = tuple(cameras)
        self.max_new_tokens = int(max_new_tokens)
        self.bf16 = bool(bf16)
        self.model_device = self._infer_model_device()
        self.history: Deque[EgoFrame] = deque(maxlen=128)
        self.last_answer: str | None = None
        self.last_trajectory: np.ndarray | None = None
        self.last_debug: Dict[str, Any] = {}

    def _infer_model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def reset(self) -> None:
        self.history.clear()
        self.last_answer = None
        self.last_trajectory = None
        self.last_debug = {}
        vision_tower = self.model.get_vision_tower()
        if vision_tower is None or vision_tower.vision_tower is None:
            raise RuntimeError("OpenDriveVLA model has no loaded UniAD vision tower")
        vision_model = vision_tower.vision_tower.vision_model
        vision_model.prev_frame_info["prev_bev"] = None
        vision_model.prev_frame_info["scene_token"] = None
        vision_model.prev_frame_info["prev_pos"] = 0
        vision_model.prev_frame_info["prev_angle"] = 0

    def _append_history(self, obs_info: Dict[str, Any]) -> None:
        self.history.append(
            EgoFrame(
                position=obs_info["ego_pos"],
                rotation=obs_info["ego_rot"],
                velocity=obs_info["linear_velocity"],
                timestamp=float(obs_info["relative_timestamp"]),
            )
        )

    @staticmethod
    def _frame_at_or_before(frames: Sequence[EgoFrame], timestamp: float) -> EgoFrame:
        selected = frames[0]
        for frame in frames:
            if frame.timestamp <= timestamp + 1e-6:
                selected = frame
            else:
                break
        return selected

    def _history_samples_lidar_xy(self, obs_info: Dict[str, Any]) -> np.ndarray:
        current_pos = obs_info["ego_pos"]
        current_rot = obs_info["ego_rot"]
        current_ts = float(obs_info["relative_timestamp"])
        lidar_to_ego = LIDAR_TO_EGO_4X4
        current_ego_to_world = np.eye(4)
        current_ego_to_world[:3, :3] = SCR.from_euler("XYZ", current_rot).as_matrix()
        current_ego_to_world[:3, 3] = current_pos
        current_lidar_to_world = current_ego_to_world @ lidar_to_ego
        world_to_current_lidar = np.linalg.inv(current_lidar_to_world)
        frames = list(self.history)
        if not frames:
            raise RuntimeError("OpenDriveVLA history is empty")
        target_offsets = [HISTORY_INTERVAL_S * i for i in range(HISTORY_POINTS, -1, -1)]
        selected = [self._frame_at_or_before(frames, current_ts - offset) for offset in target_offsets]
        lidar_points = []
        for frame in selected:
            history_ego_to_world = np.eye(4)
            history_ego_to_world[:3, :3] = SCR.from_euler("XYZ", frame.rotation).as_matrix()
            history_ego_to_world[:3, 3] = frame.position
            history_lidar_to_world = history_ego_to_world @ lidar_to_ego
            history_lidar_to_current_lidar = world_to_current_lidar @ history_lidar_to_world
            lidar_points.append(history_lidar_to_current_lidar[:2, 3])
        return np.asarray(lidar_points)

    # 已经过严格验证
    def _build_question(self, obs_info: Dict[str, Any]) -> str:
        command = int(obs_info["command"])
        ego_rot = obs_info["ego_rot"]
        velocity = obs_info["linear_velocity"][:2] * HISTORY_INTERVAL_S
        r_world_ego = SCR.from_euler("XYZ", ego_rot).as_matrix()
        can_bus_acceleration = (r_world_ego.T @ obs_info["accelerate"])[:2]
        angular_velocity = obs_info["angular_velocity"]
        history_samples = self._history_samples_lidar_xy(obs_info)
        history_points = history_samples[:HISTORY_POINTS]
        history_diff = history_samples[1:] - history_samples[:-1]
        acceleration = history_diff[-1] - history_diff[-2]
        history_text = ",".join(f"({x:.2f},{y:.2f})" for x, y in history_points)
        heading_speed = float(obs_info["ego_velo"]) * HISTORY_INTERVAL_S
        ego_message = (
            f"- Velocity (vx,vy): ({velocity[0]:.2f},{velocity[1]:.2f})"
            f" - Heading Angular Velocity (v_yaw): ({angular_velocity[2]:.2f})"
            f" - Acceleration (ax,ay): ({acceleration[0]:.2f},{acceleration[1]:.2f})"
            f" - Can Bus: ({can_bus_acceleration[0]:.2f},{can_bus_acceleration[1]:.2f})"
            f" - Heading Speed: ({heading_speed:.2f})"
            f" - Steering: ({2.0 * float(obs_info['ego_steer']) / 2.588:.2f})"
        )
        return (
            "Scene information: <scene_start><SCENE><scene_end>\n"
            "Object-wise tracking information: <track_start><TRACK><track_end>\n"
            "Map information: <map_start><MAP><map_end>\n"
            f"Ego states: {ego_message}\n"
            f"Historical trajectory (last 2 seconds): [{history_text}]\n"
            f"Mission goal: {_command_to_goal(command)}\n"
            "Planning trajectory: <trajectory>"
        )

    def _build_input_ids(self, question: str) -> torch.Tensor:
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_uniad_token

        conv = conv_templates["qwen_planning_oriented_vlm"].copy()
        conv.clear_conversation()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return tokenizer_uniad_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.model_device)

    def _prepare_uniad_data(self, obs_img: Dict[str, np.ndarray], obs_info: Dict[str, Any]) -> Dict[str, Any]:
        bgr_obs = {
            name: image[..., ::-1]
            for name, image in obs_img.items()
        }
        raw_data = parse_uniad_obs(bgr_obs, obs_info)
        img_metas = raw_data["img_metas"][0][0]
        sample_idx = f"{obs_info['scene_token']}:{float(obs_info['relative_timestamp']):.3f}"
        img_metas["sample_idx"] = sample_idx
        img_metas["pts_filename"] = f"{sample_idx}.bin"
        img_metas["img_norm_cfg"] = UNIAD_IMG_NORM_CFG
        img_metas["ori_shape"] = copy.deepcopy(img_metas["img_shape"])
        img_metas["pad_shape"] = copy.deepcopy(img_metas["img_shape"])
        img_metas["scale_factor"] = 1.0
        img_metas["flip"] = False
        return to_device(raw_data, self.model_device)

    def predict_action(
        self,
        obs_img: Dict[str, np.ndarray],
        obs_info: Dict[str, Any],
    ) -> Tuple[float, float]:
        self._append_history(obs_info)
        question = self._build_question(obs_info)
        input_ids = self._build_input_ids(question)
        uniad_data = self._prepare_uniad_data(obs_img, obs_info)

        autocast_dtype = torch.bfloat16 if self.bf16 else torch.float16
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                output_ids = self.model.generate(
                    input_ids,
                    uniad_data=uniad_data,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                )
        if output_ids.shape[1] > input_ids.shape[1]:
            generated_ids = output_ids[:, input_ids.shape[1]:]
        else:
            generated_ids = output_ids
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        traj_lidar_xy = _strict_parse_traj(answer)
        pred_traj = lidar_to_ego_front_left(traj_lidar_xy, LIDAR_TO_EGO_4X4)
        acc, steer = traj2control(pred_traj, obs_info, horizon=3.0)
        action = (float(steer), float(acc))
        self.last_answer = answer
        self.last_trajectory = pred_traj
        self.last_debug = {
            "command": int(obs_info["command"]),
            "mission_goal": _command_to_goal(int(obs_info["command"])),
            "question": question,
            "answer": answer,
            "traj_lidar_xy": traj_lidar_xy.tolist(),
            "traj_front_left": pred_traj.tolist(),
        }
        return action
