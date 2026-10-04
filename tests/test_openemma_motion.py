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
    def test_recorded_responses_exclude_echoed_history(self):
        for case in FIXTURE["cases"]:
            with self.subTest(case=case["name"]):
                expected = np.asarray(case["future_pairs"], dtype=np.float32)
                expected[:, 1] /= 100.0
                np.testing.assert_array_equal(parse_speed_curvature(case["response"]), expected)

    def test_raw_list_and_prediction_with_trailing_explanation(self):
        for response in [
            RAW_PREDICTION,
            ",\n".join(["[ 4.0 , -0.0 ]"] * 10),
            "The predicted future speeds and curvatures are: " + RAW_PREDICTION
            + ". The future speeds and curvatures remain constant.",
        ]:
            with self.subTest(response=response):
                np.testing.assert_array_equal(parse_speed_curvature(response), [[4.0, 0.0]] * 10)

    def test_missing_incomplete_and_ambiguous_predictions_are_rejected(self):
        invalid = [
            FIXTURE["invalid_object_description"],
            "Historical speeds and curvatures: " + RAW_PREDICTION,
            RAW_PREDICTION + ", " + RAW_PREDICTION,
            "Future speeds and curvatures: " + ", ".join(["[4.0, 0.0]"] * 9),
            "Future speeds and curvatures: " + RAW_PREDICTION + ", [4.0, 0.0]",
            "Future speeds and curvatures: " + RAW_PREDICTION
            + "\nAlternative future speeds and curvatures: " + RAW_PREDICTION,
        ]
        for response in invalid:
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


if __name__ == "__main__":
    unittest.main()
