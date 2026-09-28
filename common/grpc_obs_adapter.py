from __future__ import annotations

from typing import Any, Dict, List, Tuple


def unpack_ad_observation(
    obs: Dict[str, Any],
    step_info: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Strict adapter for a gRPC observation and its matching step metadata.
    """
    gaussian = obs["gaussian"]
    states = obs["states"]
    navigation = obs["navigation"]
    surrounding = obs["surrounding"]

    obs_img = gaussian["image"]
    obs_info = dict(states)
    obs_info["scene_token"] = str(step_info["scene_name"])
    obs_info["relative_timestamp"] = float(step_info["relative_timestamp"]) * 1e-6
    obs_info["timestamp"] = float(step_info["relative_timestamp"]) * 1e-6
    obs_info["current_timestamp"] = float(step_info["current_timestamp"]) * 1e-6
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
    obs_info["location"] = navigation["location"]
    obs_info["route_waypoints"] = navigation["waypoint"]
    obs_info["target_waypoint"] = navigation["target_waypoint"]
    if "carla_style_target" in navigation:
        obs_info["carla_style_target"] = navigation["carla_style_target"]
    obs_info["surroundings"] = surrounding
    return obs_img, obs_info, navigation, surrounding
