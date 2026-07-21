from typing import Dict

import numpy as np
import torch

from rl_framework.uniad.dataparser import CAMERA_ORDER, build_image_input

VAD_IMG_NORM_CFG = {
    "mean": np.array([103.530, 116.280, 123.675], dtype=np.float32),
    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "to_rgb": False,
}


def command_to_one_hot(command: int, device: torch.device) -> torch.Tensor:
    # Simulator convention: 0=right, 1=left, 2=straight.

    one_hot = torch.zeros(3, dtype=torch.float32, device=device)
    one_hot[command] = 1.0
    return one_hot


def parse_vad_obs(obs, info) -> Dict[str, object]:
    img_tensor, img_meta = build_image_input(
        obs,
        info,
        CAMERA_ORDER,
        VAD_IMG_NORM_CFG,
        image_scale=0.8,
    )

    cmd_vec = command_to_one_hot(int(info["command"]), img_tensor.device)

    return {
        "img": img_tensor,
        # Keep nested format expected by VAD forward_test.
        "img_metas": [[img_meta]],
        "ego_his_trajs": None,
        "ego_fut_cmd": cmd_vec.view(1, 1, 1, 3),
    }
