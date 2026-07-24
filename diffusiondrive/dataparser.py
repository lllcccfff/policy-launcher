from __future__ import annotations

from typing import Any, Dict

from rl_framework.sparsedrive.dataparser import FIXED_LIDAR_TOP_TO_EGO, parse_sparsedrive_obs


def parse_diffusiondrive_obs(
    obs_img: Dict[str, Any],
    obs_info: Dict[str, Any],
) -> Dict[str, Any]:
    return parse_sparsedrive_obs(obs_img, obs_info)
