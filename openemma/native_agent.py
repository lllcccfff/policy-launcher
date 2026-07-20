from __future__ import annotations

import re
from collections import deque
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np

from rl_framework.common.trajectory import traj2control
from rl_framework.openemma.backbone import OpenEMMABackendName, OpenEMMAOfficialVLM
from rl_framework.openemma.dataparser import (
    OpenEMMAHistoryFrame,
    history_frame_from_observation,
    integrate_speed_curvature,
)


class OpenEMMANativeAgent:
    _PAIR_RE = re.compile(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]")

    def __init__(
        self,
        *,
        model: OpenEMMABackendName = "gpt",
        model_id: str | None = None,
        history_len: int = 10,
        history_stride: int = 5,
        motion_dt: float = 0.1,
        prediction_dt: float = 0.5,
        camera: str = "FRONT",
        warmup_action: Tuple[float, float] = (0.0, 0.0),
    ):
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        if history_stride <= 0:
            raise ValueError(f"history_stride must be positive, got {history_stride}")
        if motion_dt <= 0.0:
            raise ValueError(f"motion_dt must be positive, got {motion_dt}")
        self.model = model
        self.vlm = OpenEMMAOfficialVLM(model, model_id=model_id)
        self.history_len = int(history_len)
        self.history_stride = int(history_stride)
        self.motion_dt = float(motion_dt)
        self.prediction_dt = float(prediction_dt)
        self.camera = camera
        self.warmup_action = (float(warmup_action[0]), float(warmup_action[1]))
        self._history: Deque[OpenEMMAHistoryFrame] = deque(maxlen=self.history_len * self.history_stride)
        self._motion_xy: Deque[np.ndarray] = deque(maxlen=self.history_len * self.history_stride)
        self._motion_speeds: Deque[float] = deque(maxlen=self.history_len * self.history_stride)
        self._motion_headings: Deque[float] = deque(maxlen=self.history_len * self.history_stride)
        self._step_index = 0
        self._prev_intent: str | None = None
        self._prev_intent_history_seconds: float | None = None
        self._prev_navigation_hint: str | None = None
        self._prev_navigation_history_seconds: float | None = None
        self._last_action = self.warmup_action
        self.last_trajectory: np.ndarray | None = None
        self.last_debug: Dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return bool(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._motion_xy.clear()
        self._motion_speeds.clear()
        self._motion_headings.clear()
        self._step_index = 0
        self._prev_intent = None
        self._prev_intent_history_seconds = None
        self._prev_navigation_hint = None
        self._prev_navigation_history_seconds = None
        self._last_action = self.warmup_action
        self.last_trajectory = None
        self.last_debug = {}

    def _append_history(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> bool:
        step_index = self._step_index
        ego_xy = np.asarray(obs_info["ego_pos"], dtype=np.float32).reshape(-1)[:2]
        self._motion_xy.append(ego_xy.astype(np.float32))
        self._motion_speeds.append(float(obs_info["ego_velo"]))
        self._motion_headings.append(float(obs_info["heading_theta"]))

        frame = history_frame_from_observation(obs_img, obs_info, camera=self.camera)
        self._history.append(frame)
        should_sample = step_index % self._history_sample_stride() == 0
        self._step_index += 1
        return should_sample

    def _chat(self, text: str, images: Sequence[str], sys_message: str | None = None) -> str:
        return self.vlm.infer(text, images, sys_message=sys_message)

    def _scene_description(self, images: Sequence[str]) -> str:
        history_seconds = self._history_seconds()
        image_interval = self._image_interval()
        if self.vlm.backend == "llava":
            prompt = (
                "You are an autonomous driving labeller. You have access to these front-view camera images of a car "
                f"taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} "
                "seconds. Imagine you are driving the car. Provide a concise description of the driving scene according "
                "to traffic lights, movements of other cars or pedestrians and lane markings."
            )
        else:
            prompt = f"You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."
        return self._chat(prompt, images)

    def _object_description(self, images: Sequence[str]) -> str:
        history_seconds = self._history_seconds()
        image_interval = self._image_interval()
        prompt = f"You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."
        return self._chat(prompt, images)

    def _navigation_hint(self, obs_info: Dict[str, Any]) -> str:
        command = int(obs_info["command"])
        if command == 1:
            return "turn left"
        elif command == 0:
            return "turn right"
        elif command == 2:
            return "keep straight"
        else:
            raise ValueError(f"Unsupported navigation command: {command}")

    def _intent_description(
        self, images: Sequence[str], obs_info: Dict[str, Any]
    ) -> Tuple[str, str, str | None, float | None, str | None, float | None]:
        history_seconds = self._history_seconds()
        elapsed_seconds = self._elapsed_seconds()
        image_interval = self._image_interval()
        navigation_hint = self._navigation_hint(obs_info)
        previous_intent = self._prev_intent
        previous_intent_history_seconds = self._prev_intent_history_seconds
        previous_navigation_hint = self._prev_navigation_hint
        previous_navigation_history_seconds = self._prev_navigation_history_seconds
        if previous_intent is None:
            if self.vlm.backend == "llava":
                prompt = (
                    "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
                    f"taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} seconds. Imagine "
                    "you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, "
                    "provide a concise description of the desired intent of  the ego car. Is it going to follow the lane "
                    f"to turn left, turn right, or go straight? The navigation hint is: {navigation_hint}. Should it "
                    "maintain the current speed or slow down or speed up?"
                )
            else:
                prompt = f"You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. The navigation hint is: {navigation_hint}. Should it maintain the current speed or slow down or speed up?"
        else:
            if previous_intent_history_seconds is None:
                raise ValueError("previous_intent_history_seconds is missing while previous_intent is set")
            previous_seconds_ago = elapsed_seconds - previous_intent_history_seconds
            if previous_seconds_ago <= 0.0:
                raise ValueError(f"previous intent timestamp must be before current timestamp, got {previous_seconds_ago}")
            if previous_navigation_hint is None:
                raise ValueError("previous_navigation_hint is missing while previous_intent is set")
            if previous_navigation_history_seconds is None:
                raise ValueError("previous_navigation_history_seconds is missing while previous_navigation_hint is set")
            if previous_navigation_history_seconds != previous_intent_history_seconds:
                raise ValueError(
                    "previous navigation timestamp must match previous intent timestamp, "
                    f"got {previous_navigation_history_seconds} and {previous_intent_history_seconds}"
                )
            if self.vlm.backend == "llava":
                prompt = (
                    "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
                    f"taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} seconds. Imagine "
                    f"you are driving the car. {previous_seconds_ago:.1f} seconds ago your intent was to {previous_intent}. "
                    f"At that time the navigation hint was: {previous_navigation_hint}. The current navigation hint is: "
                    f"{navigation_hint}. Based on the updated lane markings and the updated movement of other cars and "
                    "pedestrians, do you keep your intent or do you change it? Provide a concise description explanation "
                    "of your current intent: "
                )
            else:
                prompt = f"You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a {image_interval:.1f} second interval over the past {history_seconds:.1f} seconds. Imagine you are driving the car. {previous_seconds_ago:.1f} seconds ago your intent was to {previous_intent}. At that time the navigation hint was: {previous_navigation_hint}. The current navigation hint is: {navigation_hint}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: "

        return (
            self._chat(prompt, images),
            navigation_hint,
            previous_navigation_hint,
            previous_navigation_history_seconds,
            previous_intent,
            previous_intent_history_seconds,
        )

    def _image_interval(self) -> float:
        return self._history_sample_stride() * self.motion_dt

    def _history_seconds(self) -> float:
        return len(self._motion_xy) * self.motion_dt

    def _elapsed_seconds(self) -> float:
        return self._step_index * self.motion_dt

    def _history_sample_stride(self) -> int:
        history_seconds = self._history_seconds()
        if history_seconds < 0.5:
            return 1
        if history_seconds < 1.0:
            return 2
        if history_seconds < 1.5:
            return 3
        if history_seconds < 2.0:
            return 4
        return self.history_stride

    def _history_sample_indices(self) -> list[int]:
        stride = self._history_sample_stride()
        return list(range(0, len(self._history), stride))[-self.history_len :]

    def _history_speed_curvature(self) -> Tuple[np.ndarray, np.ndarray | None]:
        sample_indices = self._history_sample_indices()
        speeds = np.asarray(list(self._motion_speeds), dtype=np.float32)[sample_indices]
        if len(self._motion_xy) < 3:
            return speeds, None
        motion_points = np.stack(list(self._motion_xy), axis=0)
        headings = np.unwrap(np.asarray(list(self._motion_headings), dtype=np.float32))
        segment_distances = np.linalg.norm(np.diff(motion_points[:, :2], axis=0), axis=1)
        center_distances = segment_distances[:-1] + segment_distances[1:]
        if np.any(center_distances <= 0.0):
            raise ValueError(f"Cannot estimate curvature with zero-distance motion segments: {center_distances}")
        curvatures = np.zeros(motion_points.shape[0], dtype=np.float32)
        curvatures[1:-1] = (headings[2:] - headings[:-2]) / center_distances
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        return speeds, curvatures[sample_indices]

    def _generate_motion(self, obs_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        sample_indices = self._history_sample_indices()
        images = [list(self._history)[idx].image_base64 for idx in sample_indices]
        speeds, curvatures = self._history_speed_curvature()
        if curvatures is None:
            speed_curvature_prompt = ", ".join(f"[{speed:.1f},N/A]" for speed in speeds)
        else:
            speed_curvature_prompt = ", ".join(
                f"[{speed:.1f},{curvature * 100.0:.1f}]" for speed, curvature in zip(speeds, curvatures)
            )
        history_seconds = self._history_seconds()
        elapsed_seconds = self._elapsed_seconds()
        image_interval = self._image_interval()

        scene_description = self._scene_description(images)
        object_description = self._object_description(images)
        (
            intent_description,
            navigation_hint,
            previous_navigation_hint,
            previous_navigation_history_seconds,
            previous_intent,
            previous_intent_history_seconds,
        ) = self._intent_description(images, obs_info)
        self._prev_intent = intent_description
        self._prev_intent_history_seconds = elapsed_seconds
        self._prev_navigation_hint = navigation_hint
        self._prev_navigation_history_seconds = elapsed_seconds

        sys_message = (
            "You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n"
        )
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a {image_interval:.1f} second interval. 
        The scene is described as follows: {scene_description}.
        The identified critical objects are {object_description}.
        The car's intent is {intent_description}.
        The {history_seconds:.1f} second historical velocities and curvatures of the ego car are {speed_curvature_prompt}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""

        response = self._chat(prompt, images, sys_message=sys_message)
        pairs = self._PAIR_RE.findall(response.replace("Future speeds and curvatures:", ""))
        if not pairs:
            raise ValueError(f"OpenEMMA response did not contain speed-curvature pairs: {response!r}")
        raw_speed_curvature = np.asarray([[float(v), float(k)] for v, k in pairs], dtype=np.float32)[:10]
        speed_curvature = raw_speed_curvature.copy()
        speed_curvature[:, 1] = speed_curvature[:, 1] / 100.0
        traj = integrate_speed_curvature(speed_curvature, dt=self.prediction_dt)
        debug = {
            "scene_description": scene_description,
            "object_description": object_description,
            "intent_description": intent_description,
            "previous_intent": previous_intent,
            "previous_intent_history_seconds": previous_intent_history_seconds,
            "navigation_hint": navigation_hint,
            "previous_navigation_hint": previous_navigation_hint,
            "previous_navigation_history_seconds": previous_navigation_history_seconds,
            "history_size": len(self._history),
            "sampled_history_size": len(sample_indices),
            "history_sample_stride": self._history_sample_stride(),
            "history_sample_indices": sample_indices,
            "motion_history_size": len(self._motion_xy),
            "history_speed_curvature": speed_curvature_prompt,
            "response": response,
            "raw_speed_curvature": raw_speed_curvature.astype(float).tolist(),
            "trajectory": traj.astype(float).tolist(),
        }
        return traj, debug

    def predict_traj(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> np.ndarray | None:
        sample_added = self._append_history(obs_img, obs_info)
        if not self.ready:
            self.last_debug = {
                "ready": False,
                "history_size": len(self._history),
                "history_len": self.history_len,
                "action": list(self.warmup_action),
            }
            return None
        motion_history_size = len(self._motion_xy)
        motion_bootstrap_update = motion_history_size <= 3 and self.last_debug.get("motion_history_size") != motion_history_size
        if sample_added or self.last_trajectory is None or motion_bootstrap_update:
            traj, debug = self._generate_motion(obs_info)
            self.last_trajectory = traj
            self.last_debug = {"ready": True, **debug}
        return self.last_trajectory

    def predict_action_from_traj(self, traj: np.ndarray, obs_info: Dict[str, Any]) -> Tuple[float, float]:
        horizon = float(traj.shape[0]) * self.prediction_dt
        acc, steer = traj2control(traj, obs_info, horizon=horizon)
        return float(steer), float(acc)

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> Tuple[float, float]:
        traj = self.predict_traj(obs_img, obs_info)
        if traj is None:
            self._last_action = self.warmup_action
            return self._last_action
        self._last_action = self.predict_action_from_traj(traj, obs_info)
        self.last_debug["action"] = [float(self._last_action[0]), float(self._last_action[1])]
        return self._last_action
