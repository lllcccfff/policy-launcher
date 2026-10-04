from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np

from policy_launcher.openemma.backbone import OpenEMMABackendName, OpenEMMAOfficialVLM
from policy_launcher.openemma.dataparser import (
    OpenEMMAHistoryFrame,
    history_frame_from_observation,
    integrate_speed_curvature,
    parse_speed_curvature,
)


class OpenEMMANativeAgent:
    """Build driving context from recent history and predict an ego-frame trajectory."""

    # Based on taco-group/openemma main.py at 8403ea636696c5c10e8fdeca566410de0a07e449,
    # with navigation hints and timing from the configured observation interval.
    def __init__(
        self,
        *,
        model: OpenEMMABackendName = "gpt",
        model_id: str | None = None,
        prediction_dt: float = 0.5,
        camera: str = "FRONT",
        warmup_action: Tuple[float, float] = (0.0, 0.0),
    ):
        self.model = model
        self.vlm = OpenEMMAOfficialVLM(model, model_id=model_id)
        self.history_len = 10
        self.motion_dt = 0.5
        self.prediction_dt = float(prediction_dt)
        self.camera = camera
        self.warmup_action = (float(warmup_action[0]), float(warmup_action[1]))
        self._history: Deque[OpenEMMAHistoryFrame] = deque(maxlen=self.history_len)
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        self._prev_intent: str | None = None
        self._prev_navigation_hint: str | None = None
        self._scene_token: str | None = None

    def predict_action(self, obs_img: Dict[str, Any], obs_info: Dict[str, Any]) -> np.ndarray:
        scene_token = str(obs_info["scene_token"])
        if scene_token != self._scene_token:
            self.reset()
            self._scene_token = scene_token

        self._history.append(history_frame_from_observation(obs_img, obs_info, camera=self.camera))
        if len(self._history) < self.history_len:
            return np.asarray(self.warmup_action, dtype=np.float32).reshape(1, 2)

        if self._prev_intent is None:
            print(
                f"[OpenEMMA warmup] scene={scene_token} "
                f"history_frames={self.history_len} "
                f"first_inference_step={self.history_len - 1} "
                f"history_interval={self.motion_dt:g}s",
                flush=True,
            )
        return self._generate_motion(obs_info)

    def _generate_motion(self, obs_info: Dict[str, Any]) -> np.ndarray:
        navigation_hint = {0: "turn right", 1: "turn left", 2: "keep straight"}[int(obs_info["command"])]
        history = list(self._history)
        images = [frame.image_base64 for frame in history]
        speeds, curvatures = self._history_speed_curvature(history)
        curvatures = curvatures * 100
        speed_curvature_prompt = ", ".join(
            f"[{speed:.1f},{curvature:.1f}]" for speed, curvature in zip(speeds, curvatures)
        )
        history_seconds = len(history) * self.motion_dt

        sys_message = (
            "You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, "
            "a sequence of past speeds, a sequence of past curvatures, and a driving rationale. "
            "Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds "
            "to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is "
            "turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the "
            "vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. "
            "You should try to stay in the middle of your lane. You should maintain necessary distance from the "
            "leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to "
            "predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent "
            "inferred from the image. Make a best guess if the problem is too difficult for you. "
            "If you cannot provide a response people will get injured.\n"
        )
        for attempt in range(1, 4):
            scene_description = self._scene_description(images)
            object_description = self._object_description(images)
            intent_description = self._intent_description(images, navigation_hint)
            self._prev_intent = intent_description
            self._prev_navigation_hint = navigation_hint

            prompt = (
                "These are frames from a video taken by a camera mounted in the front of a car. "
                f"The images are taken at a {self.motion_dt:g} second interval. \n"
                f"        The scene is described as follows: {scene_description}. \n"
                f"        The identified critical objects are {object_description}. \n"
                f"        The car's intent is {intent_description}. \n"
                f"        The {history_seconds:g} second historical velocities and curvatures of the ego car are "
                f"{speed_curvature_prompt}. \n"
                "        Infer the association between these numbers and the image sequence. "
                "Generate the predicted future speeds and curvatures in the format "
                "[speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. "
                "Write the raw text not markdown or latex. Future speeds and curvatures:"
            )

            for generation_attempt in range(1, 4):
                response = self.vlm.infer(prompt, images, sys_message=sys_message)
                if "unable" not in response and "sorry" not in response and "[" in response:
                    break
                if generation_attempt < 3:
                    print(
                        f"[OpenEMMA generation retry] scene={self._scene_token} "
                        f"motion_attempt={attempt}/3 generation_attempt={generation_attempt}/3 "
                        f"response={response!r}", flush=True,
                    )
            try:
                speed_curvature = parse_speed_curvature(response)
            except ValueError as exc:
                if attempt == 3:
                    raise ValueError(
                        f"OpenEMMA failed to generate a valid trajectory after 3 attempts: {response!r}"
                    ) from exc
                print(
                    f"[OpenEMMA retry] scene={self._scene_token} attempt={attempt}/3 "
                    f"reason={exc} response={response!r}", flush=True,
                )
            else:
                return integrate_speed_curvature(speed_curvature, dt=self.prediction_dt)

    def _scene_description(self, images: Sequence[str]) -> str:
        history_seconds = len(self._history) * self.motion_dt
        if self.vlm.backend == "llava":
            prompt = (
                "You are an autonomous driving labeller. You have access to these front-view camera images of a car "
                f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} "
                "seconds. Imagine you are driving the car. Provide a concise description of the driving scene according "
                "to traffic lights, movements of other cars or pedestrians and lane markings."
            )
        else:
            prompt = (
                "You are a autonomous driving labeller. You have access to these front-view camera images of a car "
                f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} seconds. "
                "Imagine you are driving the car. Describe the driving scene according to traffic lights, "
                "movements of other cars or pedestrians and lane markings."
            )
        return self.vlm.infer(prompt, images, sys_message=None)

    def _object_description(self, images: Sequence[str]) -> str:
        history_seconds = len(self._history) * self.motion_dt
        prompt = (
            "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
            f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} seconds. "
            "Imagine you are driving the car. What other road users should you pay attention to in the driving scene? "
            "List two or three of them, specifying its location within the image of the driving scene and provide "
            "a short description of the that road user on what it is doing, and why it is important to you."
        )
        return self.vlm.infer(prompt, images, sys_message=None)

    def _intent_description(self, images: Sequence[str], navigation_hint: str) -> str:
        history_seconds = len(self._history) * self.motion_dt
        previous_intent = self._prev_intent
        if previous_intent is None:
            if self.vlm.backend == "llava":
                prompt = (
                    "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
                    f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} seconds. Imagine "
                    "you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, "
                    "provide a concise description of the desired intent of  the ego car. Is it going to follow the lane "
                    f"to turn left, turn right, or go straight? The navigation hint is: {navigation_hint}. Should it "
                    "maintain the current speed or slow down or speed up?"
                )
            else:
                prompt = (
                    "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
                    f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} seconds. "
                    "Imagine you are driving the car. Based on the lane markings and the movement of other cars "
                    "and pedestrians, describe the desired intent of the ego car. "
                    "Is it going to follow the lane to turn left, turn right, or go straight? "
                    f"The navigation hint is: {navigation_hint}. "
                    "Should it maintain the current speed or slow down or speed up?"
                )
        else:
            if self.vlm.backend == "llava":
                prompt = (
                    "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
                    f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} seconds. Imagine "
                    f"you are driving the car. {self.motion_dt:g} seconds ago your intent was to {previous_intent}. "
                    f"At that time the navigation hint was: {self._prev_navigation_hint}. "
                    f"The current navigation hint is: {navigation_hint}. "
                    "Based on the updated lane markings and the updated movement of other cars and "
                    "pedestrians, do you keep your intent or do you change it? Provide a concise description explanation "
                    "of your current intent: "
                )
            else:
                prompt = (
                    "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
                    f"taken at a {self.motion_dt:g} second interval over the past {history_seconds:g} seconds. "
                    f"Imagine you are driving the car. {self.motion_dt:g} seconds ago your intent was to {previous_intent}. "
                    f"At that time the navigation hint was: {self._prev_navigation_hint}. "
                    f"The current navigation hint is: {navigation_hint}. "
                    "Based on the updated lane markings and the "
                    "updated movement of other cars and pedestrians, do you keep your intent or do you change it? "
                    "Explain your current intent: "
                )

        return self.vlm.infer(prompt, images, sys_message=None)

    def _history_speed_curvature(
        self, history: Sequence[OpenEMMAHistoryFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        speeds = np.asarray([frame.speed for frame in history], dtype=np.float32)
        motion_points = np.stack([frame.ego_xy for frame in history], axis=0)
        headings = np.unwrap(np.asarray([frame.heading_theta for frame in history], dtype=np.float32))
        segment_distances = np.linalg.norm(np.diff(motion_points[:, :2], axis=0), axis=1)
        center_distances = segment_distances[:-1] + segment_distances[1:]
        if np.any(center_distances <= 0.0):
            raise ValueError(f"Cannot estimate curvature with zero-distance motion segments: {center_distances}")
        curvatures = np.zeros(motion_points.shape[0], dtype=np.float32)
        curvatures[1:-1] = (headings[2:] - headings[:-2]) / center_distances
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        return speeds, curvatures


class OpenEMMANativeAgentGPT(OpenEMMANativeAgent):
    def __init__(self):
        super().__init__(model="gpt")


class OpenEMMANativeAgentQwen(OpenEMMANativeAgent):
    def __init__(self):
        super().__init__(model="qwen")


class OpenEMMANativeAgentLLaVA(OpenEMMANativeAgent):
    def __init__(self):
        super().__init__(model="llava")


class OpenEMMANativeAgentLlama(OpenEMMANativeAgent):
    def __init__(self):
        super().__init__(model="llama")
