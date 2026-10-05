import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from policy_launcher.openemma.native_agent import (
    OpenEMMANativeAgentLlama,
    OpenEMMANativeAgentLLaVA,
    OpenEMMANativeAgentQwen,
)
from policy_launcher.openemma.dataparser import OpenEMMAHistoryFrame


class OpenEmmaWarmupTest(unittest.TestCase):
    def observation(self, step, scene="scene-a", dt=0.5):
        return {"FRONT": np.full((8, 8, 3), step, dtype=np.uint8)}, {
            "scene_token": scene,
            "ego_pos": [4.0 * step * dt, 0.0],
            "ego_velo": 4.0,
            "heading_theta": 0.0,
            "command": 2,
        }

    def test_all_benchmark_backends_wait_for_ten_frames_and_reset_per_scene(self):
        for agent_class, backend in [
            (OpenEMMANativeAgentLlama, "llama"),
            (OpenEMMANativeAgentQwen, "qwen"),
            (OpenEMMANativeAgentLLaVA, "llava"),
        ]:
            with self.subTest(backend=backend):
                vlm = SimpleNamespace(backend=backend, infer=Mock(return_value=", ".join(["[4.0, 0.0]"] * 10)))
                with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
                    agent = agent_class()
                for scene in ["scene-a", "scene-b"]:
                    vlm.infer.reset_mock()
                    for step in range(9):
                        agent.predict_action(*self.observation(step, scene))
                        vlm.infer.assert_not_called()
                    trajectory = agent.predict_action(*self.observation(9, scene))
                    self.assertEqual(vlm.infer.call_count, 4)
                    self.assertEqual(trajectory.shape, (10, 2))
                    np.testing.assert_allclose(trajectory[:, 0], np.arange(1, 11) * 2.0)
                    prompt, images = vlm.infer.call_args.args
                    self.assertEqual(len(images), 10)
                    self.assertIn("0.5 second interval", prompt)
                    self.assertNotIn("N/A", prompt)
                    self.assertNotIn("not yet available", prompt)

    def test_infers_each_step_and_updates_intent_after_history_rolls(self):
        vlm = SimpleNamespace(backend="qwen", infer=Mock(side_effect=[
            "scene", "lead car", "follow the lane", ", ".join(["[4.0, 0.0]"] * 10),
            "updated scene", "pedestrian", "slow down", ", ".join(["[2.0, 0.0]"] * 10),
        ]))
        with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
            agent = OpenEMMANativeAgentQwen()
        for step in range(10):
            agent.predict_action(*self.observation(step))
        self.assertEqual(vlm.infer.call_count, 4)
        first_images = vlm.infer.call_args.args[1]

        obs_img, obs_info = self.observation(10)
        obs_info["command"] = 1
        second = agent.predict_action(obs_img, obs_info)
        self.assertEqual(vlm.infer.call_count, 8)
        np.testing.assert_allclose(second[:, 0], np.arange(1, 11))
        intent_prompt = vlm.infer.call_args_list[6].args[0]
        self.assertIn("0.5 seconds ago your intent was to follow the lane", intent_prompt)
        self.assertIn("At that time the navigation hint was: keep straight", intent_prompt)
        self.assertIn("The current navigation hint is: turn left", intent_prompt)
        self.assertEqual(len(vlm.infer.call_args.args[1]), 10)
        second_images = vlm.infer.call_args.args[1]
        self.assertEqual(second_images[:-1], first_images[1:])
        self.assertNotEqual(second_images[-1], first_images[-1])

    def test_invalid_motion_response_raises(self):
        vlm = SimpleNamespace(backend="llama", infer=Mock(side_effect=[
            "scene", "objects", "intent", *(["no numeric prediction"] * 3),
        ] * 3))
        with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
            agent = OpenEMMANativeAgentLlama()
        for step in range(9):
            agent.predict_action(*self.observation(step))
        with self.assertRaisesRegex(ValueError, "after 3 attempts"):
            agent.predict_action(*self.observation(9))
        self.assertEqual(vlm.infer.call_count, 18)

    def test_all_benchmark_backends_infer_with_stationary_history(self):
        for agent_class, backend in [
            (OpenEMMANativeAgentLlama, "llama"),
            (OpenEMMANativeAgentQwen, "qwen"),
            (OpenEMMANativeAgentLLaVA, "llava"),
        ]:
            with self.subTest(backend=backend):
                vlm = SimpleNamespace(backend=backend, infer=Mock(return_value=", ".join(["[0.0,0.0]"] * 10)))
                with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
                    agent = agent_class()
                with np.errstate(divide="raise", invalid="raise"):
                    for step in range(10):
                        obs_img, obs_info = self.observation(0)
                        obs_info["ego_velo"] = 0.0
                        obs_info["heading_theta"] = 0.01 * step
                        trajectory = agent.predict_action(obs_img, obs_info)
                        self.assertEqual(vlm.infer.call_count, 0 if step < 9 else 4)
                prompt = vlm.infer.call_args.args[0]
                self.assertIn(
                    "historical velocities and curvatures of the ego car are " + ", ".join(["[0.0,0.0]"] * 10),
                    prompt,
                )
                np.testing.assert_array_equal(trajectory, np.zeros((10, 2), dtype=np.float32))

    def test_mixed_stationary_and_moving_history_preserves_moving_curvature(self):
        positions = [0, 0, 0, 1, 2, 2, 2, 2, 3, 4]
        speeds = [0, 0, 0, 2, 2, 0, 0, 0, 2, 2]
        history = [
            OpenEMMAHistoryFrame("", np.asarray([x, 0], dtype=np.float32), i * 0.1, speed)
            for i, (x, speed) in enumerate(zip(positions, speeds))
        ]
        with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM"):
            agent = OpenEMMANativeAgentQwen()
        with np.errstate(divide="raise", invalid="raise"):
            actual_speeds, curvatures = agent._history_speed_curvature(history)
        np.testing.assert_array_equal(actual_speeds, speeds)
        np.testing.assert_allclose(curvatures, [0, 0, 0.2, 0.1, 0.2, 0, 0, 0.2, 0.1, 0.1], rtol=1e-5)

    def test_small_nonzero_motion_still_uses_measured_distance(self):
        history = [
            OpenEMMAHistoryFrame("", np.asarray([i * 1e-8, 0], dtype=np.float32), i * 1e-9, 1e-8)
            for i in range(10)
        ]
        with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM"):
            agent = OpenEMMANativeAgentQwen()
        with np.errstate(divide="raise", invalid="raise"):
            _, curvatures = agent._history_speed_curvature(history)
        np.testing.assert_allclose(curvatures, 0.1, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
