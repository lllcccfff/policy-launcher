from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _to_vec3(value: Any) -> List[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(3, dtype=np.float32)
    n = min(3, arr.shape[0])
    out[:n] = arr[:n]
    return out.tolist()


def turn_signal_to_command(turn_signal: Any) -> int:
    sign = int(np.sign(float(turn_signal)))
    mapping = {-1: 0, 0: 2, 1: 1}  # right, straight, left
    return mapping[sign]


def unpack_ad_observation(
    obs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Strict adapter for new assembly observation only.
    """
    gaussian = obs["gaussian"]
    states = obs["states"]
    navigation = obs["navigation"]
    surrounding = obs["surrounding"]

    obs_img = gaussian["image"]
    obs_info = dict(states)
    obs_info["cam_params"] = gaussian["camera_info"]
    obs_info["command"] = turn_signal_to_command(navigation["turn_signal"])
    obs_info["expert_path"] = navigation["waypoint"]
    obs_info["surroundings"] = surrounding
    obs_info["ego_pos"] = _to_vec3(obs_info["ego_pos"])
    obs_info["ego_rot"] = _to_vec3(obs_info["ego_rot"])
    obs_info["linear_velocity"] = _to_vec3(obs_info["linear_velocity"])
    obs_info["linear_acceleration"] = _to_vec3(obs_info["linear_acceleration"])
    obs_info["angular_velocity"] = _to_vec3(obs_info["angular_velocity"])
    return obs_img, obs_info, navigation, surrounding

