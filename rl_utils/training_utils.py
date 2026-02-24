import numbers
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize

from rl_framework.rl_configs.config_loader import load_configs
from rl_framework.rl_utils.lr_utils import build_learning_rate_schedule


class TrainingConfig:
    def __init__(
        self,
        config_dir: Optional[str] = None,
        override_dir: Optional[str] = None,
        backend: str = "uniad",
    ):
        """
        Initialize training configuration from YAML files.

        Args:
            config_dir: Directory containing base YAML config files (default: rl_configs/)
            override_dir: Optional directory containing override YAML files
            backend: Config backend, one of {"uniad", "vad"}.
        """
        self.backend = str(backend).strip().lower()
        configs = load_configs(config_dir, override_dir, backend=self.backend)
        ppo_cfg = configs["ppo_training"]
        policy_cfg = configs["policy_network"]
        env_cfg = configs["environment"]
        paths_cfg = configs["paths"]
        self.paths_cfg = paths_cfg

        self.project_root = os.getcwd()
        backend_paths = paths_cfg.get(self.backend)
        if backend_paths is None:
            raise ValueError(f"paths_{self.backend}.yaml must define `{self.backend}` section")
        ad_root_rel = backend_paths.get("ad_root")
        if not ad_root_rel:
            raise ValueError(f"paths_{self.backend}.yaml `{self.backend}` section must define `ad_root`")
        self.AD_root = self.resolve_project_path(ad_root_rel)

        if self.backend == "uniad":
            self.uniad_config_path = self.resolve_ad_path(backend_paths.config_path)
            self.uniad_checkpoint_path = self.resolve_ad_path(backend_paths.checkpoint_path)
            self.pretrained_planning_weights = backend_paths.get("pretrained_planning_weights")
            if self.pretrained_planning_weights:
                self.pretrained_planning_weights = self.resolve_ad_path(self.pretrained_planning_weights)
        else:
            self.uniad_config_path = ""
            self.uniad_checkpoint_path = ""
            self.pretrained_planning_weights = None

        self.output_dir = os.path.join(self.project_root, paths_cfg.output.base_dir)
        self.log_dir = os.path.join(self.project_root, paths_cfg.output.log_dir)
        self.checkpoint_dir = os.path.join(self.project_root, paths_cfg.output.checkpoint_dir)
        self.tensorboard_log = os.path.join(self.project_root, paths_cfg.output.tensorboard_log)
        self.vec_normalize_path = os.path.join(self.project_root, paths_cfg.normalization.vec_normalize_path)
        self.eval_results_path = os.path.join(self.project_root, paths_cfg.output.eval_results)

        self.uniad_device = env_cfg.devices.uniad_device if torch.cuda.is_available() else "cpu"
        self.policy_device = env_cfg.devices.policy_device if torch.cuda.is_available() else "cpu"

        self.embed_dims = policy_cfg.architecture.embed_dims
        self.bev_h = policy_cfg.architecture.bev_h
        self.bev_w = policy_cfg.architecture.bev_w
        self.planning_steps = policy_cfg.architecture.planning_steps

        self.normalize_obs = ppo_cfg.normalization.normalize_obs
        self.normalize_reward = ppo_cfg.normalization.normalize_reward
        self.clip_obs = ppo_cfg.normalization.clip_obs
        self.clip_reward = ppo_cfg.normalization.clip_reward

        self.n_steps = ppo_cfg.ppo_algorithm.n_steps
        self.batch_size = ppo_cfg.ppo_algorithm.batch_size
        self.n_epochs = ppo_cfg.ppo_algorithm.n_epochs
        self.gamma = ppo_cfg.ppo_algorithm.gamma
        self.gae_lambda = ppo_cfg.ppo_algorithm.gae_lambda
        self.clip_range = ppo_cfg.ppo_algorithm.clip_range
        self.clip_range_vf = ppo_cfg.ppo_algorithm.clip_range_vf
        self.ent_coef = ppo_cfg.ppo_algorithm.ent_coef
        self.vf_coef = ppo_cfg.ppo_algorithm.vf_coef
        self.max_grad_norm = ppo_cfg.ppo_algorithm.max_grad_norm
        self.learning_rate = ppo_cfg.learning_rate.base_learning_rate
        self.use_sde = ppo_cfg.ppo_algorithm.use_sde
        self.sde_sample_freq = ppo_cfg.ppo_algorithm.sde_sample_freq
        self.lr_schedule = ppo_cfg.learning_rate.schedule
        self.lr_final_ratio = ppo_cfg.learning_rate.lr_final_ratio
        self.target_kl = ppo_cfg.ppo_algorithm.target_kl
        self.grad_accum_steps = ppo_cfg.ppo_algorithm.grad_accum_steps

        self.total_timesteps = ppo_cfg.training_logistics.total_timesteps
        self.save_freq = ppo_cfg.training_logistics.save_freq
        self.eval_freq = ppo_cfg.training_logistics.eval_freq
        self.eval_episodes = ppo_cfg.training_logistics.eval_episodes
        self.verbose = ppo_cfg.training_logistics.verbose
        self.episode_info_print_every = ppo_cfg.training_logistics.episode_info_print_every
        self.episode_info_log_to_tb = ppo_cfg.training_logistics.episode_info_log_to_tb

        self.with_adapter = policy_cfg.architecture.with_adapter

        self.training_phase = 0
        self.actor_lr = policy_cfg.learning_rates.actor_lr
        self.critic_lr = policy_cfg.learning_rates.critic_lr
        self.actor_lora_lr_ratio = policy_cfg.learning_rates.actor_lora_lr_ratio
        self.actor_adapter_lr_ratio = policy_cfg.learning_rates.actor_adapter_lr_ratio
        self.actor_std_lr_ratio = policy_cfg.learning_rates.actor_std_lr_ratio
        self.lora_warmup_updates = policy_cfg.learning_rates.lora_warmup_updates
        self.adapter_warmup_updates = policy_cfg.learning_rates.adapter_warmup_updates
        self.std_warmup_updates = policy_cfg.learning_rates.std_warmup_updates

        self.log_std_bounds = tuple(policy_cfg.exploration.log_std_bounds)
        self.policy_log_std_init = policy_cfg.exploration.log_std_init
        self.policy_use_expln = policy_cfg.exploration.use_expln
        self.policy_full_std = policy_cfg.exploration.full_std
        self.policy_latent_sde_scale = policy_cfg.exploration.latent_sde_scale

        self.kl_reg_coef = policy_cfg.regularization.kl_reg_coef
        self.bc_reg_coef = policy_cfg.regularization.bc_reg_coef
        self.unfreeze_adapter_phase2 = policy_cfg.architecture.unfreeze_adapter_phase2

        self.lora_rank = policy_cfg.lora.rank
        self.lora_alpha = policy_cfg.lora.alpha
        self.lora_dropout = policy_cfg.lora.dropout
        self.optimizer_weight_decay = policy_cfg.optimizer.weight_decay

        self.phase_behavior: Dict[str, Any] = {}

        self.phase_presets = {}
        for phase_id, preset_data in ppo_cfg.phase_presets.to_dict().items():
            phase_dict = {}
            for key, value in preset_data.items():
                if key == "phase_behavior":
                    phase_dict[key] = dict(value)
                elif key == "log_std_bounds":
                    phase_dict[key] = tuple(value)
                else:
                    phase_dict[key] = value
            self.phase_presets[int(phase_id)] = phase_dict

        self.phase_schedule = [tuple(item) for item in ppo_cfg.phase_schedule]
        self.apply_phase_defaults()

    def resolve_project_path(self, resource_path: str) -> str:
        """Resolve a path relative to workspace root."""
        path_str = str(resource_path)
        if os.path.isabs(path_str):
            return os.path.normpath(path_str)
        return os.path.normpath(os.path.join(self.project_root, path_str))

    def resolve_ad_path(self, resource_path: str) -> str:
        """Resolve a path relative to AD_root."""
        path_str = str(resource_path)
        if os.path.isabs(path_str):
            return os.path.normpath(path_str)
        return os.path.normpath(os.path.join(self.AD_root, path_str))

    def create_dirs(self):
        """创建必要的目录"""
        for dir_path in [self.output_dir, self.log_dir, self.checkpoint_dir, self.tensorboard_log]:
            os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 输出目录已创建: {self.output_dir}")

    def apply_phase_defaults(self, phase: Optional[int] = None):
        if phase is not None:
            self.training_phase = phase
        defaults = self.phase_presets.get(self.training_phase)
        if defaults is None:
            self.training_phase = 1
            defaults = self.phase_presets[self.training_phase]
        self.phase_behavior = {}
        for key, value in defaults.items():
            if key == "phase_behavior":
                self.phase_behavior = value.copy()
            else:
                setattr(self, key, value)


class TensorboardCallback(BaseCallback):
    """自定义回调：记录额外的训练指标到 Tensorboard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_len_mean",
                np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_rew_mean",
                np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
            )
        return True


class EpisodeInfoPrinterCallback(BaseCallback):
    """打印并记录环境在 info['episode']['env'] 中提供的诊断统计"""

    def __init__(
        self,
        print_every: int = 1,
        log_to_tb: bool = True,
        keys_to_display: Optional[List[str]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.print_every = max(1, int(print_every))
        self.log_to_tb = bool(log_to_tb)
        self.keys_to_display = keys_to_display or [
            "sum/progress",
            "sum/ttc",
            "sum/position",
            "sum/heading",
            "sum/collision",
            "count/stall_steps",
            "count/ttc_warning_steps",
        ]
        self.episode_counter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            episode_info = info.get("episode")
            if not isinstance(episode_info, dict):
                continue
            env_stats = episode_info.get("env")
            if env_stats is None and isinstance(info.get("env"), dict):
                env_stats = info["env"]
                episode_info["env"] = env_stats
            if not isinstance(env_stats, dict):
                continue
            self.episode_counter += 1
            self._log_env_stats(env_stats)
            if self.episode_counter % self.print_every == 0:
                reward_str = self._format_value(episode_info.get("r"))
                length_str = self._format_value(episode_info.get("l"))
                selected = []
                for key in self.keys_to_display:
                    if key in env_stats:
                        selected.append(f"{key}={self._format_value(env_stats[key])}")
                detail = " ".join(selected)
                print(f"[Episode {self.episode_counter}] r={reward_str} l={length_str} {detail}".strip())
        return True

    def _log_env_stats(self, env_stats: Dict[str, Any]) -> None:
        if not self.log_to_tb or self.logger is None:
            return
        for key, value in env_stats.items():
            if isinstance(value, numbers.Number):
                tag = key.replace("/", "_")
                self.logger.record(f"episode/env/{tag}", float(value))

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, numbers.Integral):
            return str(int(value))
        if isinstance(value, numbers.Number):
            return f"{float(value):.3f}"
        if value is None:
            return "-"
        return str(value)


def _get_env_action_affine(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract action scale/bias from a (possibly wrapped) SB3 VecEnv/Env.
    Works with VecNormalize, DummyVecEnv, SubprocVecEnv, and nested gym wrappers.
    """
    for scale_key, bias_key in [("action_scale", "action_bias"), ("_action_scale", "_action_bias")]:
        try:
            scale_list = env.get_attr(scale_key)
            bias_list = env.get_attr(bias_key)
            if scale_list and bias_list:
                scale0, bias0 = scale_list[0], bias_list[0]
                for s, b in zip(scale_list[1:], bias_list[1:]):
                    if hasattr(s, "detach"):
                        s = s.detach().cpu().numpy()
                    if hasattr(b, "detach"):
                        b = b.detach().cpu().numpy()
                scale_arr = (
                    scale0.detach().cpu().numpy()
                    if hasattr(scale0, "detach")
                    else np.asarray(scale0, np.float32)
                )
                bias_arr = (
                    bias0.detach().cpu().numpy()
                    if hasattr(bias0, "detach")
                    else np.asarray(bias0, np.float32)
                )
                return scale_arr, bias_arr
        except Exception:
            pass

    base_env = env
    if isinstance(base_env, VecNormalize):
        base_env = base_env.venv

    if hasattr(base_env, "envs") and getattr(base_env, "envs", None):
        base_env = base_env.envs[0]

    while hasattr(base_env, "env"):
        if hasattr(base_env, "action_scale") or hasattr(base_env, "_action_scale"):
            break
        base_env = base_env.env

    scale = getattr(base_env, "action_scale", getattr(base_env, "_action_scale", None))
    bias = getattr(base_env, "action_bias", getattr(base_env, "_action_bias", None))
    if scale is None or bias is None:
        raise AttributeError("Unable to locate env action scale/bias for policy initialization")

    scale_arr = scale.detach().cpu().numpy() if hasattr(scale, "detach") else np.asarray(scale, np.float32)
    bias_arr = bias.detach().cpu().numpy() if hasattr(bias, "detach") else np.asarray(bias, np.float32)
    return scale_arr, bias_arr


def _to_schedule(value: Optional[Union[float, Callable[[float], float]]]):
    if value is None:
        return None
    if callable(value):
        return value
    return get_schedule_fn(value)


def ensure_rollout_buffer_size(model: Any) -> None:
    buffer = getattr(model, "rollout_buffer", None)
    if buffer is None or buffer.buffer_size == model.n_steps:
        return
    buffer_cls = buffer.__class__
    model.rollout_buffer = buffer_cls(
        model.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )


def build_phase_plan(
    total_steps: int,
    schedule: Optional[List[Tuple[int, Union[float, int]]]],
    fallback_phase: int,
) -> List[Tuple[int, int]]:
    total_steps = int(total_steps)
    if total_steps <= 0:
        return []
    if not schedule:
        return [(fallback_phase, total_steps)]
    plan: List[Tuple[int, int]] = []
    remaining = total_steps
    for idx, (phase_id, portion) in enumerate(schedule):
        if remaining <= 0:
            break
        if isinstance(portion, numbers.Integral):
            allocated = int(portion)
        else:
            allocated = int(total_steps * float(portion))
        allocated = max(min(allocated, remaining), 0)
        if allocated <= 0 and idx < len(schedule) - 1:
            continue
        plan.append((phase_id, allocated))
        remaining -= allocated
    if not plan:
        plan.append((fallback_phase, total_steps))
    elif remaining > 0:
        last_phase, last_steps = plan[-1]
        plan[-1] = (last_phase, last_steps + remaining)
    return plan
