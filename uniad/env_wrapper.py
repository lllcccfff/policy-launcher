import gymnasium as gym
from gymnasium import spaces
import numpy as np
import colorsys
import torch
import os
import sys
from pathlib import Path
from collections import defaultdict
import cv2
from scipy.spatial.transform import Rotation as SCR
from typing import Any, Dict, Tuple, Optional, Callable
import numbers
from rl_framework.common.trajectory import traj2control
import colorsys
import matplotlib.pyplot as plt

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "simulator/GaussianDrive"))
sys.path.insert(0, os.path.join(project_root, "simulator/EasyDrive"))

from metadrive.utils.visualize import project_points_on_image, render_vehicle_trajectories

save_dir = os.path.join('/home/caohejiakang/ADPolicy', 'output_test', 'cirr4_test')
map_save_dir = os.path.join('/nas2/caohejiakang/ADPolicy_backup', 'output_test', 'topdown_maps')

def save_obs_images(obs_img, save_dir, step=0, prefix='', save_all_stack=False, camera_filter=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []

    if isinstance(obs_img, dict) and 'image' in obs_img:
        images = obs_img['image']
    else:
        images = obs_img
    
    for cam_name, img_stack in images.items():
        if camera_filter is not None and cam_name not in camera_filter:
            continue

        if save_all_stack:
            for stack_idx in range(img_stack.shape[0]):
                img = img_stack[stack_idx]  # (H, W, 3)

                img_to_save = process_image_for_saving(img)

                filename = f"{prefix}{cam_name}_step{step:06d}_stack{stack_idx}.png"
                filepath = save_dir / filename

                cv2.imwrite(str(filepath), img_to_save)
                saved_files.append(str(filepath))
        else:
            img = img_stack[-1]  # (H, W, 3)

            img_to_save = process_image_for_saving(img)

            filename = f"{prefix}{cam_name}_step{step:06d}.png"
            filepath = save_dir / filename

            cv2.imwrite(str(filepath), img_to_save)
            saved_files.append(str(filepath))
    
    return saved_files

def process_image_for_saving(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (img * 255).astype(np.uint8)
    elif img.dtype == np.uint8:
        img_uint8 = img
    else:
        img_uint8 = img.astype(np.uint8)

    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    return img_bgr


class UniADWrapper(gym.Wrapper):
    """
    Process:
    1. 获取环境的 obs 和 info
    2. 使用 dataparser 将 obs 和 info 转换为 raw_data
    3. 将 raw_data 输入 UniAD 获取 plan_query 和 bev_feat
    4. 返回处理后的观测
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        uniad_model: torch.nn.Module,
        dataparser: Callable,
        embed_dims: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        planning_steps: int = 6,
        uniad_device: str = 'cuda:0',
        policy_device: str = 'cuda:1'
    ):
        """
        Args:
            env: 基础环境
            uniad_model: UniAD模型实例
            dataparser: 数据解析函数，接收(obs, info)返回raw_data
            device: 推理设备
        """
        super().__init__(env)
        self.uniad_model = uniad_model
        self.dataparser = dataparser
        self.uniad_device = uniad_device
        self.policy_device = policy_device
        self.planning_steps = planning_steps
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.uniad_model.eval()
        for param in self.uniad_model.parameters():
            param.requires_grad = False
        
        self.observation_space = spaces.Dict({
            'plan_query': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, self.embed_dims),
                dtype=np.float32
            ),
            'bev_feat': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.bev_h * self.bev_w, self.embed_dims),
                dtype=np.float32
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.planning_steps * 2,),
            dtype=np.float32
        )

        lateral_bounds = (-0.6, 0.6)
        longitudinal_bounds = (0.0, 4.0)
        lat_scale = torch.full((self.planning_steps,), (lateral_bounds[1] - lateral_bounds[0]) * 0.5, dtype=torch.float32)
        lat_bias = torch.full((self.planning_steps,), (lateral_bounds[1] + lateral_bounds[0]) * 0.5, dtype=torch.float32)
        lon_scale = torch.full((self.planning_steps,), (longitudinal_bounds[1] - longitudinal_bounds[0]) * 0.5, dtype=torch.float32)
        lon_bias = torch.full((self.planning_steps,), (longitudinal_bounds[1] + longitudinal_bounds[0]) * 0.5, dtype=torch.float32)
        lat_min = torch.full((self.planning_steps,), lateral_bounds[0], dtype=torch.float32)
        lat_max = torch.full((self.planning_steps,), lateral_bounds[1], dtype=torch.float32)
        lon_min = torch.full((self.planning_steps,), longitudinal_bounds[0], dtype=torch.float32)
        lon_max = torch.full((self.planning_steps,), longitudinal_bounds[1], dtype=torch.float32)
        self._action_scale = torch.stack((lat_scale, lon_scale), dim=-1).view(-1)
        self._action_bias = torch.stack((lat_bias, lon_bias), dim=-1).view(-1)
        self._action_min = torch.stack((lat_min, lon_min), dim=-1).view(-1)
        self._action_max = torch.stack((lat_max, lon_max), dim=-1).view(-1)

        self.step_count = 0
        self._expert_path_info = None
        self.traj_img = None
        self.camera_k = None
        self.uniad_color, self.uniad_color_end = self._hsl_gradient_colors((255, 255, 0), lighten_only=True)
        self.ft_color, self.ft_color_end = self._hsl_gradient_colors((28, 244, 167))
        self.trajectory_start_offset = 0.3
        self.latest_uniad_world_traj: Optional[np.ndarray] = None
        self._ttc_warn_horizon = self._infer_ttc_warn_horizon()
        self._reset_episode_stats()

    def reset(
        self, 
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
            processed_obs: 处理后的观测，包含 plan_query 和 bev_feat
            info: 环境信息
        """
        self._reset_episode_stats()
        reset_output = self.env.reset(seed=seed, **kwargs)
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, reset_info = reset_output
        else:
            obs, reset_info = reset_output, {}
        obs_img, obs_info = obs
        self._expert_path_info = self._extract_expert_path_info()
        scene_name = reset_info.get("scene_name") if isinstance(reset_info, dict) else None
        if scene_name is None:
            scene_name = getattr(self.env, "scene_name", None)
        if scene_name is not None:
            obs_info['scene_token'] = str(scene_name)
        processed_obs = self._process_observation(obs_img, obs_info) # plan_query and bev_feat
        return processed_obs, obs_info
    
    def step(
        self, 
        action: Any
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Args:
            action: 环境动作
            
        Returns:
            processed_obs: 处理后的观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 环境信息
        """
        action = torch.as_tensor(action, dtype=torch.float32)
        single_action = action.dim() == 1
        if single_action:
            action = action.unsqueeze(0)

        physical_action = self._denormalize_action(action)
        increments = physical_action.reshape(action.shape[0], self.planning_steps, 2)
        trajectory = torch.cumsum(increments, dim=1)

        vehicle = self.env.actor_controller
        plan_info = {
            'ego_velo': vehicle.speed,
            'ego_steer': vehicle.steering * np.deg2rad(vehicle.max_steering)
        }
    
        acc, steer = traj2control(trajectory[0].cpu().numpy(), plan_info)
        control_action = (steer, acc) # control action != acc and steer in info

        obs, reward, terminated, truncated, info = self.env.step(control_action)
        obs_img, obs_info = obs
        scene_name = info.get("scene_name") if isinstance(info, dict) else None
        if scene_name is None:
            scene_name = getattr(self.env, "scene_name", None)
        if scene_name is not None:
            obs_info['scene_token'] = str(scene_name)

        processed_obs = self._process_observation(obs_img, obs_info)

        if info is None:
            info = {}
        self._update_episode_stats(info, terminated, truncated)
        info.update(self._build_metric_payload(obs_info))

        return processed_obs, reward, terminated, truncated, info

    @property
    def action_scale(self) -> torch.Tensor:
        """Expose env action scale for policy use."""
        return self._action_scale

    @property
    def action_bias(self) -> torch.Tensor:
        """Expose env action bias for policy use."""
        return self._action_bias
    
    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        scale = self._action_scale.to(action.device)
        bias = self._action_bias.to(action.device)
        min_val = self._action_min.to(action.device)
        max_val = self._action_max.to(action.device)
        normalized = torch.clamp(action, -1.0, 1.0)
        physical = normalized * scale + bias
        return torch.max(torch.min(physical, max_val), min_val)
    
    def _process_observation(
        self, 
        obs_img: Dict[str, Any], 
        obs_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Args:
            obs_img and obs_info
            
        Returns:
            plan_query and bev_feat
        """
        img_norm_cfg = {
        'mean': np.array([103.530, 116.280, 123.675]), 
        'std': np.array([1.0, 1.0, 1.0]), 
        'to_rgb': False
        }

        cameras = {'FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT'}
        
        raw_data = self.dataparser(obs_img, obs_info, cameras, img_norm_cfg)

        if isinstance(raw_data, dict):
            raw_data = self._to_device(raw_data, self.uniad_device)

        del raw_data['raw_imgs']

        with torch.no_grad():
            model_output = self.uniad_model(return_loss=False, rescale=True, feature_extractor=True, **raw_data)

        plan_query = model_output.get('plan_query', None)
        bev_feat = model_output.get('bev_feat', None)

        # [P, 1, C] -> [P, C]
        plan_query = plan_query.squeeze(1).to(self.policy_device)  
        # [H*W, 1, C] -> [H*W, C]
        bev_feat = bev_feat.squeeze(1).to(self.policy_device)

        processed_obs = {
            'plan_query': plan_query.cpu().numpy().astype(np.float32),
            'bev_feat': bev_feat.cpu().numpy().astype(np.float32),
        }
        
        assert processed_obs['plan_query'].shape == (1, self.embed_dims), \
            f"plan_query shape mismatch: {processed_obs['plan_query'].shape}"
        assert processed_obs['bev_feat'].shape == (self.bev_h * self.bev_w, self.embed_dims), \
            f"bev_feat shape mismatch: {processed_obs['bev_feat'].shape}"
        
        return processed_obs
    
    def _to_device(
        self, 
        data: Any, 
        device: str
    ) -> Any:

        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._to_device(item, device) for item in data)
        else:
            return data

    def _extract_expert_path_info(self):
        actor_mgr = getattr(self.env, 'agent_managers', {}).get('actor', None) if hasattr(self.env, 'agent_managers') else None
        observer = getattr(actor_mgr, 'observer', None) if actor_mgr is not None else None
        nav_obs = None
        if observer is None:
            return None
        if hasattr(observer, '_observers'):
            nav_obs = observer._observers.get('navigation')
        else:
            nav_obs = observer
        if nav_obs is None:
            return None
        start = None
        end = None
        path_xy = getattr(nav_obs, '_path_xy', None)
        path_points = None
        cumlen = None

        if path_xy is not None and len(path_xy) > 0:
            path_points = np.asarray(path_xy, dtype=np.float32)
            start = path_points[0]
            end = path_points[-1]
            if path_points.shape[0] >= 2:
                seg_lengths = np.linalg.norm(path_points[1:] - path_points[:-1], axis=1)
                cumlen = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        init_state = getattr(nav_obs, 'init_state', {})
        if start is None and isinstance(init_state, dict):
            spawn_pos = init_state.get('spawn_position')
            if spawn_pos is not None:
                start = np.asarray(spawn_pos, dtype=np.float32)
        destination = getattr(nav_obs, 'destination', None)
        if end is None:
            if destination is not None:
                end = np.asarray(destination, dtype=np.float32)
            elif isinstance(init_state, dict):
                dest = init_state.get('destination')
                if dest is not None:
                    end = np.asarray(dest, dtype=np.float32)
        if path_points is None and start is not None and end is not None:
            path_points = np.stack([start, end], axis=0)
            seg_len = np.linalg.norm(end - start)
            cumlen = np.array([0.0, seg_len], dtype=np.float32)
        if start is None or end is None:
            return None
        info = {
            'start_xy': start[:2].astype(np.float32).tolist(),
            'end_xy': end[:2].astype(np.float32).tolist(),
        }
        if path_points is not None and cumlen is not None:
            info['path_xy'] = path_points[:, :2].astype(np.float32).tolist()
            info['path_cumlen'] = cumlen.astype(np.float32).tolist()
        return info

    def _build_metric_payload(self, obs_info: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            'ego_state': self._compose_ego_state(obs_info),
            'surroundings': self._compose_surroundings()
        }
        if self._expert_path_info is not None:
            payload['expert_path'] = self._expert_path_info
        return payload
    
    def _update_episode_stats(self, info: Dict[str, Any], terminated: bool, truncated: bool) -> None:
        if not isinstance(info, dict):
            return
        reward_components = info.get("reward_components") or {}
        for name, value in reward_components.items():
            if isinstance(value, numbers.Number):
                self._episode_reward_sums[name] += float(value)
        diag = info.get("diag") or {}
        stalled_flag = diag.get("stalled", info.get("stalled", False))
        if isinstance(stalled_flag, (bool, numbers.Number)) and bool(stalled_flag):
            self._episode_counts["stall_steps"] += 1
        ttc_value = diag.get("ttc")
        if isinstance(ttc_value, numbers.Number) and ttc_value <= self._ttc_warn_horizon:
            self._episode_counts["ttc_warning_steps"] += 1
        collision_flag = info.get("collision")
        if isinstance(collision_flag, numbers.Number) and collision_flag >= 1:
            self._episode_counts["collision_happened"] = 1
        self._episode_len += 1
        if terminated or truncated:
            summary = self._build_episode_summary()
            if summary:
                env_summary = info.get("env")
                if not isinstance(env_summary, dict):
                    env_summary = {}
                env_summary.update(summary)
                info["env"] = env_summary
                episode_info = info.setdefault("episode", {})
                if isinstance(episode_info, dict):
                    episode_env = episode_info.get("env")
                    if not isinstance(episode_env, dict):
                        episode_env = {}
                    episode_env.update(summary)
                    episode_info["env"] = episode_env

    def _build_episode_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, value in self._episode_reward_sums.items():
            summary[f"sum/{name}"] = float(value)
        for name, value in self._episode_counts.items():
            summary[f"count/{name}"] = int(value) if isinstance(value, bool) else value
        summary["len"] = self._episode_len
        return summary

    def _reset_episode_stats(self) -> None:
        self._episode_reward_sums = defaultdict(float)
        self._episode_counts = dict(
            stall_steps=0,
            ttc_warning_steps=0,
            collision_happened=0,
        )
        self._episode_len = 0
        self._ttc_warn_horizon = self._infer_ttc_warn_horizon()

    def _infer_ttc_warn_horizon(self) -> float:
        config = getattr(self.env, "config", None)
        if isinstance(config, dict):
            warn = config.get("ttc_warn_horizon", 2.0)
        else:
            warn = getattr(config, "ttc_warn_horizon", 2.0) if config is not None else 2.0
        try:
            return float(warn)
        except (TypeError, ValueError):
            return 2.0

    def _compose_ego_state(self, obs_info: Dict[str, Any]) -> Dict[str, Any]:
        ego_pos = np.asarray(obs_info.get('ego_pos', np.zeros(3, dtype=np.float32)), dtype=np.float32)
        ego_rot = np.asarray(obs_info.get('ego_rot', np.zeros(3, dtype=np.float32)), dtype=np.float32)
        R_world_ego = SCR.from_euler('XYZ', ego_rot).as_matrix().astype(np.float32)
        yaw = float(ego_rot[2]) if ego_rot.shape[0] > 2 else 0.0

        linear_vel = np.asarray(obs_info.get('linear_velocity', np.zeros(3, dtype=np.float32)), dtype=np.float32)
        ego_vel = R_world_ego.T @ linear_vel
        linear_acc = np.asarray(obs_info.get('linear_acceleration', np.zeros(3, dtype=np.float32)), dtype=np.float32)
        ego_acc = R_world_ego.T @ linear_acc
        angular_velocity = np.asarray(obs_info.get('angular_velocity', np.zeros(3, dtype=np.float32)), dtype=np.float32)

        return {
            'position_world': ego_pos[:2].astype(np.float32).tolist(),
            'ego_velocity': ego_vel[:2].astype(np.float32).tolist(),
            'ego_acceleration': ego_acc[:2].astype(np.float32).tolist(),
            'yaw': yaw,
            'angular_velocity': float(angular_velocity[2]) if angular_velocity.shape[0] > 2 else 0.0,
            'timestamp': float(obs_info.get('timestamp', 0.0))
        }

    def _compose_surroundings(self) -> list:
        actor_mgr = getattr(self.env, 'agent_managers', {}).get('actor', None) if hasattr(self.env, 'agent_managers') else None
        observer = getattr(actor_mgr, 'observer', None) if actor_mgr is not None else None
        surrounding_observer = observer._observers.get('surrounding')
        surrounding_obs = surrounding_observer.observe()
        neighbors = []
        
        for obj in surrounding_obs:
            neighbors.append({
                'position': obj['position'],
                'velocity': obj['velocity'],
                'size': obj.get('size', [4.5, 1.8])
            })
        return neighbors

    def _hsl_gradient_colors(
        self,
        base_rgb: Tuple[int, int, int],
        lighten_only: bool = False
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        r, g, b = [c / 255.0 for c in base_rgb]
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        delta = 25.0 / 255.0
        start_l = l if lighten_only else float(np.clip(l - delta, 0.0, 1.0))
        end_l = float(np.clip(l + delta, 0.0, 1.0))

        start_rgb = colorsys.hls_to_rgb(h, start_l, s)
        end_rgb = colorsys.hls_to_rgb(h, end_l, s)

        return self._rgbf_to_bgr(start_rgb), self._rgbf_to_bgr(end_rgb)

    def _rgbf_to_bgr(self, rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
        r, g, b = rgb
        return (int(b * 255), int(g * 255), int(r * 255))
