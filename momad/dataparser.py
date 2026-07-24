from __future__ import annotations

from typing import Any, Dict

from rl_framework.diffusiondrive.dataparser import parse_diffusiondrive_obs


def parse_momad_obs(
    obs_img: Dict[str, Any],
    obs_info: Dict[str, Any],
) -> Dict[str, Any]:
    return parse_diffusiondrive_obs(obs_img, obs_info)
