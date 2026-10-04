import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from policy_launcher.openemma.native_agent import (
    OpenEMMANativeAgentGPT,
    OpenEMMANativeAgentLlama,
    OpenEMMANativeAgentLLaVA,
    OpenEMMANativeAgentQwen,
)


class OpenEmmaPromptTest(unittest.TestCase):
    def test_only_navigation_and_configured_timing_differ_from_official_prompts(self):
        # Captured from the pinned official functions, not from the agent under test.
        reference = json.loads((Path(__file__).parent / "fixtures/openemma_prompts_8403ea6.json").read_text())
        for agent_class, backend in (
            (OpenEMMANativeAgentGPT, "gpt"),
            (OpenEMMANativeAgentLlama, "llama"),
            (OpenEMMANativeAgentQwen, "qwen"),
            (OpenEMMANativeAgentLLaVA, "llava"),
        ):
            for interval in (0.5, 0.2, 0.05):
                with self.subTest(backend=backend, interval=interval):
                    vlm = SimpleNamespace(backend=backend, infer=Mock(side_effect=reference["responses"]))
                    with patch("policy_launcher.openemma.native_agent.OpenEMMAOfficialVLM", return_value=vlm):
                        agent = agent_class()
                    # A different fixed configuration for each run; never change it during rollout.
                    agent.motion_dt = interval
                    speeds = np.asarray(reference["speeds"], dtype=np.float32)
                    curvatures = np.asarray(reference["curvatures"], dtype=np.float32)
                    with patch.object(agent, "_history_speed_curvature", return_value=(speeds, curvatures)):
                        for step in range(11):
                            agent.predict_action({"FRONT": np.full((8, 8, 3), step, dtype=np.uint8)}, {
                                "scene_token": "scene-a", "ego_pos": [step * 4 * interval, 0],
                                "ego_velo": 4.0, "heading_theta": 0.0,
                                "command": 0 if step < 10 else 1,
                            })
                    self.assertEqual(agent.motion_dt, interval)
                    expected = reference["calls"]["llava" if backend == "llava" else "standard"]
                    self.assertEqual(vlm.infer.call_count, len(expected))
                    for index, (actual, original) in enumerate(zip(vlm.infer.call_args_list, expected)):
                        with self.subTest(call=index):
                            text = original["text"].replace("0.5 second interval", f"{interval:g} second interval")
                            text = text.replace("past 5 seconds", f"past {10 * interval:g} seconds")
                            text = text.replace("The 5 second historical", f"The {10 * interval:g} second historical")
                            if index == 2:
                                text = text.replace("Should it ", "The navigation hint is: turn right. Should it ")
                            elif index == 6:
                                text = text.replace("Half a second ago", f"{interval:g} seconds ago")
                                text = text.replace(
                                    "Based on the updated",
                                    "At that time the navigation hint was: turn right. "
                                    "The current navigation hint is: turn left. Based on the updated",
                                )
                            self.assertEqual(actual.args[0].encode("utf-8"), text.encode("utf-8"))
                            self.assertEqual(actual.kwargs["sys_message"], original["sys_message"])
                            self.assertEqual(len(actual.args[1]), 10)


if __name__ == "__main__":
    unittest.main()
