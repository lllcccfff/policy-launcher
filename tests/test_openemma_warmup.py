import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from policy_launcher.openemma.native_agent import (
    OpenEMMANativeAgentLlama,
    OpenEMMANativeAgentLLaVA,
    OpenEMMANativeAgentQwen,
)


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

    def test_stationary_history_raises_before_inference(self):
        vlm = SimpleNamespace(backend="qwen", infer=Mock())
        with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
            agent = OpenEMMANativeAgentQwen()
        for _ in range(9):
            agent.predict_action(*self.observation(0))
        with self.assertRaisesRegex(ValueError, "zero-distance motion segments"):
            agent.predict_action(*self.observation(0))
        vlm.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
