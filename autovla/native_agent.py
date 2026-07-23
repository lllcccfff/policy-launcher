from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from rl_framework.autovla.loader import AutoVLABundle

CAMERA_MAP = {
    "front_camera": "FRONT",
    "front_left_camera": "FRONT_LEFT",
    "front_right_camera": "FRONT_RIGHT",
}


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame, mode="RGB")


def _command_to_instruction(command: int) -> str:
    if command == 0:
        return "Turn Right"
    if command == 1:
        return "Turn Left"
    if command == 2:
        return "Go Straight"
    return "Go Straight"


class AutoVLANativeAgent:
    def __init__(
        self,
        bundle: AutoVLABundle,
        history_frames: int = 4,
        frame_interval: float = 0.5,
        camera_map: Mapping[str, str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_length: int | None = None,
    ):
        self.bundle = bundle
        self.model = bundle.model
        self.processor = bundle.model.processor
        self.camera_map = dict(CAMERA_MAP if camera_map is None else camera_map)
        self.history_frames = int(history_frames)
        if self.history_frames <= 0:
            raise ValueError(f"history_frames must be positive, got {history_frames}")
        self.frame_interval = float(frame_interval)
        self.video_conf = dict(bundle.config["model"]["video"])
        self.gen_conf = dict(bundle.config["inference"]["sample"])
        if temperature is not None:
            self.gen_conf["temperature"] = float(temperature)
        if top_k is not None:
            self.gen_conf["top_k"] = int(top_k)
        if top_p is not None:
            self.gen_conf["top_p"] = float(top_p)
        if max_length is not None:
            self.gen_conf["max_length"] = int(max_length)
        self._frames: Dict[str, Deque[np.ndarray]] = {
            logical: deque(maxlen=self.history_frames) for logical in self.camera_map
        }
        self._last_frame_ts: Dict[str, float | None] = {logical: None for logical in self.camera_map}
        self.last_debug: Dict[str, Any] = {}

    def reset(self) -> None:
        for frames in self._frames.values():
            frames.clear()
        for key in self._last_frame_ts:
            self._last_frame_ts[key] = None
        self.last_debug = {}

    def _update_image_history(self, obs_img: Mapping[str, np.ndarray], timestamp: float) -> None:
        for logical_name, env_name in self.camera_map.items():
            if env_name not in obs_img:
                raise KeyError(f"AutoVLA requires camera {env_name!r}; got {sorted(obs_img)}")
            frame = obs_img[env_name]
            frames = self._frames[logical_name]
            last_ts = self._last_frame_ts[logical_name]
            should_append = (
                not frames
                or self.frame_interval <= 0.0
                or last_ts is None
                or timestamp - last_ts >= self.frame_interval - 1e-6
            )
            if not frames:
                for _ in range(self.history_frames):
                    frames.append(frame.copy())
                self._last_frame_ts[logical_name] = timestamp
            elif should_append:
                frames.append(frame.copy())
                self._last_frame_ts[logical_name] = timestamp

    def _video_frames(self, logical_name: str) -> list[Image.Image]:
        frames = list(self._frames[logical_name])
        if not frames:
            raise RuntimeError(f"AutoVLA image history for {logical_name} is empty")
        while len(frames) < self.history_frames:
            frames.insert(0, frames[0])
        return [_frame_to_pil(frame) for frame in frames[-self.history_frames:]]

    def _build_messages(self, obs_info: Mapping[str, Any]) -> list[dict[str, Any]]:
        min_pixels = self.video_conf["min_pixels"]
        max_pixels = self.video_conf["max_pixels"]
        velocity = float(obs_info["ego_velo"])
        acceleration = float(np.linalg.norm(np.asarray(obs_info["accelerate"], dtype=np.float32).reshape(-1)[:2]))
        instruction = _command_to_instruction(int(obs_info["command"])).lower()

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": self._video_frames("front_camera")
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": self._video_frames("front_left_camera")
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": self._video_frames("front_right_camera")
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
                )
            },
        ]

        if bool(self.bundle.config["model"]["use_cot"]):
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
                "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                "Structure your reasoning as follows:\n"
                "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
                "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
                "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
                "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                "Present the final action clearly after your reasoning steps."
            )
        else:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def _prepare_model_inputs(self, messages: Sequence[dict[str, Any]]) -> Dict[str, torch.Tensor]:
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.bundle.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    def _decode_generated_trajectory(self, generated_ids: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        outputs_trimmed = generated_ids[:-1].detach().cpu()
        action_tokens = outputs_trimmed[outputs_trimmed >= int(self.model.action_start_id)]
        trajectory = self.model.action_tokenizer.decode_token_ids_to_trajectory(action_tokens)[0, 1:, :2]
        trajectory_np = trajectory.detach().cpu().numpy().astype(np.float32)
        return trajectory_np, action_tokens

    def predict_action(
        self,
        obs_img: Dict[str, np.ndarray],
        obs_info: Dict[str, Any],
    ) -> np.ndarray:
        obs_info["ego_velo"] = float(obs_info["ego_velo"])
        obs_info["ego_steer"] = float(obs_info["ego_steer"])
        timestamp = obs_info["relative_timestamp"]
        self._update_image_history(obs_img, timestamp)

        messages = self._build_messages(obs_info)
        model_inputs = self._prepare_model_inputs(messages)
        max_length = int(self.gen_conf["max_length"])
        generation_kwargs: Dict[str, Any] = {
            "max_length": max_length,
            "do_sample": True,
            "temperature": float(self.gen_conf["temperature"]),
            "top_k": int(self.gen_conf["top_k"]),
            "top_p": float(self.gen_conf["top_p"]),
        }

        with torch.inference_mode():
            outputs = self.model.vlm.generate(**model_inputs, **generation_kwargs)
        input_len = int(model_inputs["input_ids"].shape[1])
        generated_ids = outputs[0, input_len:]
        output_text = self.processor.decode(generated_ids.detach().cpu(), skip_special_tokens=False)

        pred_traj, action_tokens = self._decode_generated_trajectory(generated_ids)

        self.last_debug = {
            "command": int(obs_info["command"]),
            "instruction": _command_to_instruction(int(obs_info["command"])),
            "output_text": output_text,
            "action_tokens": [int(x) for x in action_tokens.detach().cpu().tolist()],
            "traj_front_left": pred_traj.astype(float).tolist(),
        }
        return pred_traj
