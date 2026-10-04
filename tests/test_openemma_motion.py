import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rl_framework.openemma.dataparser import parse_speed_curvature
from rl_framework.openemma.native_agent import OpenEMMANativeAgentLlama


FIXTURE = json.loads((Path(__file__).parent / "fixtures/openemma_motion_responses.json").read_text())
RAW_PREDICTION = ", ".join(["[4.0, 0.0]"] * 10)


class OpenEmmaMotionTest(unittest.TestCase):
    def test_recorded_responses_take_the_original_first_pairs(self):
        for case in FIXTURE["cases"]:
            with self.subTest(case=case["name"]):
                expected = np.asarray(case["first_pairs"], dtype=np.float32)
                expected[:, 1] /= 100.0
                np.testing.assert_array_equal(parse_speed_curvature(case["response"]), expected)

    def test_raw_list_and_prediction_with_trailing_explanation(self):
        for response in [
            RAW_PREDICTION,
            RAW_PREDICTION + "\n\nNote: The future speeds and curvatures are based on the assumption "
            "that the car maintains its current speed and direction.",
            "The predicted future speeds and curvatures are: " + RAW_PREDICTION
            + ". The future speeds and curvatures remain constant.",
        ]:
            with self.subTest(response=response):
                np.testing.assert_array_equal(parse_speed_curvature(response), [[4.0, 0.0]] * 10)

    def test_short_predictions_preserve_the_original_variable_horizon(self):
        for count in (1, 8, 9):
            response = "Predicted future speeds and curvatures:\n" + "\n".join(
                f"{i + 1}. [4.0, 0.0]" for i in range(count)
            )
            with self.subTest(count=count):
                np.testing.assert_array_equal(parse_speed_curvature(response), [[4.0, 0.0]] * count)

    def test_extra_pairs_and_historical_labels_follow_the_original_parser(self):
        responses = [
            "Historical speeds and curvatures: " + RAW_PREDICTION,
            RAW_PREDICTION + ", " + RAW_PREDICTION,
            RAW_PREDICTION + "\nHistorical speeds and curvatures: [4.0, 0.0]",
            "Future speeds and curvatures: " + RAW_PREDICTION + ", [4.0, 0.0]",
            "Future speeds and curvatures: " + RAW_PREDICTION
            + "\nAlternative future speeds and curvatures: " + RAW_PREDICTION,
        ]
        for response in responses:
            with self.subTest(response=response):
                np.testing.assert_array_equal(parse_speed_curvature(response), [[4.0, 0.0]] * 10)

    def test_responses_without_original_numeric_matches_are_rejected(self):
        for response in (FIXTURE["invalid_object_description"], "[ 4.0 , 0.0 ]"):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    parse_speed_curvature(response)

    def test_invalid_response_regenerates_only_the_same_motion_request(self):
        vlm = SimpleNamespace(backend="llama", infer=Mock(side_effect=[
            "scene", "objects", "keep straight", FIXTURE["invalid_object_description"], RAW_PREDICTION,
        ]))
        with patch("rl_framework.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
            agent = OpenEMMANativeAgentLlama()
        for step in range(10):
            trajectory = agent.predict_action({"FRONT": np.zeros((8, 8, 3), dtype=np.uint8)}, {
                "scene_token": "0001", "ego_pos": [2.0 * step, 0.0], "ego_velo": 4.0,
                "heading_theta": 0.0, "command": 2,
            })
        self.assertEqual(vlm.infer.call_count, 5)
        self.assertEqual(vlm.infer.call_args_list[3], vlm.infer.call_args_list[4])
        np.testing.assert_allclose(trajectory[:, 0], np.arange(1, 11) * 2.0)
        self.assertEqual(len(agent._history), 10)
        self.assertEqual(agent._prev_intent, "keep straight")

    def test_original_outer_retry_refreshes_context_after_three_missing_predictions(self):
        invalid = FIXTURE["invalid_object_description"]
        vlm = SimpleNamespace(backend="llama", infer=Mock(side_effect=[
            "first scene", "first objects", "keep straight", invalid, invalid, invalid,
            "updated scene", "updated objects", "keep straight", RAW_PREDICTION,
        ]))
        with patch("rl_framework.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
            agent = OpenEMMANativeAgentLlama()
        for step in range(10):
            trajectory = agent.predict_action({"FRONT": np.zeros((8, 8, 3), dtype=np.uint8)}, {
                "scene_token": "0001", "ego_pos": [2.0 * step, 0.0], "ego_velo": 4.0,
                "heading_theta": 0.0, "command": 2,
            })
        self.assertEqual(vlm.infer.call_count, 10)
        calls = vlm.infer.call_args_list
        self.assertEqual(calls[3], calls[4])
        self.assertEqual(calls[4], calls[5])
        self.assertIn("updated scene", calls[9].args[0])
        self.assertIn("updated objects", calls[9].args[0])
        self.assertEqual(calls[3].args[1], calls[9].args[1])
        self.assertEqual(len(agent._history), 10)
        np.testing.assert_allclose(trajectory[:, 0], np.arange(1, 11) * 2.0)


if __name__ == "__main__":
    unittest.main()
