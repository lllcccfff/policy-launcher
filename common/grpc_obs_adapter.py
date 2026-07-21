from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
    turn_signal = navigation["turn_signal"]
    if turn_signal == -1:
        command = 0
    elif turn_signal == 1:
        command = 1
    else:
        command = 2
    obs_info["command"] = command
    obs_info["route_type"] = navigation["navigating_type"]
    obs_info["route_waypoints"] = navigation["waypoint"]
    obs_info["target_waypoint"] = navigation["target_waypoint"]
    obs_info["surroundings"] = surrounding
    return obs_img, obs_info, navigation, surrounding
