from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from rl_framework.uniad.dataparser import CAMERA_ORDER, build_image_input


GENAD_IMG_NORM_CFG = {
    "mean": np.array([123.675, 116.280, 103.530], dtype=np.float32),
    "std": np.array([58.395, 57.120, 57.375], dtype=np.float32),
    # EasyDrive provides RGB; GenAD's file pipeline converts BGR to RGB before normalization.
    "to_rgb": False,
}


def parse_genad_obs(
    obs: Dict[str, np.ndarray],
    info: Dict[str, Any],
) -> Dict[str, object]:
    img_tensor, img_meta = build_image_input(
        obs,
        info,
        CAMERA_ORDER,
        GENAD_IMG_NORM_CFG,
        image_scale=0.4,
    )

    height, width = img_tensor.shape[-2:]
    padded_height = (height + 31) // 32 * 32
    padded_width = (width + 31) // 32 * 32
    img_tensor = F.pad(img_tensor, (0, padded_width - width, 0, padded_height - height))
    img_meta["pad_shape"] = [[padded_height, padded_width, 3] for _ in CAMERA_ORDER]

    command = int(info["command"])
    command_one_hot = torch.zeros(3, dtype=torch.float32, device=img_tensor.device)
    command_one_hot[command] = 1.0

    return {
        "img": img_tensor,
        "img_metas": [[img_meta]],
        "ego_his_trajs": None,
        "ego_lcf_feat": None,
        "ego_fut_cmd": command_one_hot.view(1, 1, 1, 3),
    }
