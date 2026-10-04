import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from rl_framework.openemma.backbone import OpenEMMAOfficialVLM


class Inputs(dict):
    def to(self, device):
        return self


class OpenEmmaLlamaDecodeTest(unittest.TestCase):
    def test_generated_answer_does_not_require_chat_end_markers(self):
        vlm = OpenEMMAOfficialVLM.__new__(OpenEMMAOfficialVLM)
        vlm._decode_image = Mock(return_value=object())
        vlm._processor = Mock(return_value=Inputs(input_ids=np.array([[11, 12, 13]])))
        vlm._processor.decode.return_value = "  a generated answer without chat markers  "
        vlm._model = SimpleNamespace(device="cpu", generate=Mock(return_value=np.array([[11, 12, 13, 21, 22]])))

        self.assertEqual(vlm._infer_llama("prompt", ["image"]), "a generated answer without chat markers")
        call = vlm._processor.decode.call_args
        self.assertEqual(call.args[0].tolist(), [21, 22])
        self.assertEqual(call.kwargs, {"skip_special_tokens": True})


if __name__ == "__main__":
    unittest.main()
