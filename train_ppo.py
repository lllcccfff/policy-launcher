"""
PPO 训练脚本
使用 UniADWrapper 包装的环境和 CustomUniADPolicy 进行强化学习训练
"""
import os
import sys
import numbers
import pprint
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime

from gymnasium import spaces

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_schedule_fn, explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from rl_framework.uniad.env_wrapper import UniADWrapper
from rl_framework.uniad.policy_network import CustomUniADPolicy
from rl_framework.uniad.loader import create_uniad
from rl_framework.uniad.dataparser import parse_uniad_obs
from rl_framework.common.metrics import EpisodeMetricTracker
from rl_framework.rl_configs.config_loader import load_configs
from rl_framework.rl_utils.lr_utils import (
    apply_aux_lr_brake,
    apply_aux_lr_warmup,
    assign_actor_critic_lr_schedules,
    build_learning_rate_schedule,
    configure_aux_lr_warmup_steps,
    configure_phase_lr_context,
    step_aux_lr_warmup,
    update_learning_rate,
)


class UniADPPO(PPO):
    """Custom PPO trainer that injects UniAD-specific regularization into the loss."""

    def __init__(self, *args, **kwargs):
        self.grad_accum_steps = max(1, int(kwargs.pop("grad_accum_steps", 1)))
        super().__init__(*args, **kwargs)
        self._actor_lr_schedule: Optional[Union[float, Callable[[float], float]]] = None
        self._critic_lr_schedule: Optional[Union[float, Callable[[float], float]]] = None
        self._phase_total_timesteps: int = 0
        self._phase_start_num_timesteps: int = 0
        self._lora_warmup_steps = 10
        self._adapter_warmup_steps = 15
        self._std_warmup_steps = 15
        self._lora_warmup_progress = 0
        self._adapter_warmup_progress = 0
        self._std_warmup_progress = 0
        self._update_iterations = 0
        self._bc_loss_ma = 0.0
        self._kl_stop_triggers = 0

    def train(self) -> None:
        self.policy.set_training_mode(False)
        self._update_iterations += 1
        self._apply_aux_lr_warmup()
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, reg_losses = [], [], []
        clip_fractions = []
        approx_kl_divs = []

        continue_training = True
        kl_early_stop = False

        grad_accum_steps = max(1, getattr(self, "grad_accum_steps", 1))
        accum_count = 0
        self.policy.optimizer.zero_grad()
        last_loss = None
        pre_step_checked = False
        kl_window: List[float] = []

        adv_mean = None
        adv_std = None
        if self.normalize_advantage:
            buffer_adv = self.rollout_buffer.advantages
            if buffer_adv is not None:
                adv_tensor = buffer_adv
                if not isinstance(buffer_adv, torch.Tensor):
                    adv_tensor = torch.as_tensor(buffer_adv, device=self.policy.device)
                if adv_tensor.numel() > 1:
                    adv_mean = adv_tensor.mean()
                    adv_std = adv_tensor.std().clamp_min(1e-8)

        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                if self.use_sde:
                    # Keep noise reset frequency identical to vanilla PPO for minimal behavior change.
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and advantages.numel() > 1 and adv_mean is not None:
                    advantages = (advantages - adv_mean) / adv_std

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                reg_loss = torch.zeros((), device=values.device)
                if hasattr(self.policy, "pop_regularization_loss"):
                    reg_loss = self.policy.pop_regularization_loss()
                reg_losses.append(reg_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + reg_loss
                )

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    approx_kl_value = float(approx_kl_div.cpu().numpy())
                    approx_kl_divs.append(approx_kl_value)
                    kl_window.append(approx_kl_value)

                last_loss = loss
                (loss / grad_accum_steps).backward()
                accum_count += 1

                if accum_count % grad_accum_steps == 0:
                    if kl_window and self.target_kl is not None:
                        mean_kl = sum(kl_window) / len(kl_window)
                        kl_window.clear()
                        if mean_kl > 1.5 * self.target_kl:
                            continue_training = False
                            self._kl_stop_triggers += 1
                            self.policy.optimizer.zero_grad()
                            if self.verbose >= 1:
                                print(
                                    f"[KLStop] early stop before step at epoch {epoch + 1}, mean_kl={mean_kl:.4f}, target={self.target_kl:.4f}"
                                )
                            break
                    if not pre_step_checked:
                        max_diff = torch.max(torch.abs(log_prob - rollout_data.old_log_prob)).item()
                        print(f"[PPO] pre-step |log_prob-old_log_prob|_max={max_diff:.3e}")
                        pre_step_checked = True
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    self.policy.optimizer.zero_grad()
                    accum_count = 0

            if not continue_training:
                break

        if accum_count > 0 and continue_training:
            if kl_window and self.target_kl is not None:
                mean_kl = sum(kl_window) / len(kl_window)
                kl_window.clear()
                if mean_kl > 1.5 * self.target_kl:
                    self._kl_stop_triggers += 1
                    self.policy.optimizer.zero_grad()
                    if self.verbose >= 1:
                        print(f"[KLStop] skipped final step, mean_kl={mean_kl:.4f}, target={self.target_kl:.4f}")
                    continue_training = False
            if continue_training:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()
                accum_count = 0

        mean_kl_value = float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0
        self._step_aux_lr_warmup()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        mean_reg_loss = float(np.mean(reg_losses)) if reg_losses else 0.0
        if not hasattr(self, "_bc_loss_ma") or not np.isfinite(self._bc_loss_ma):
            self._bc_loss_ma = mean_reg_loss
        else:
            self._bc_loss_ma = 0.9 * self._bc_loss_ma + 0.1 * mean_reg_loss
        self.logger.record("train/reg_loss", mean_reg_loss)
        self.logger.record("train/reg_loss_ma", self._bc_loss_ma)
        self.logger.record("train/approx_kl", mean_kl_value)
        self.logger.record("train/kl_stop_count", float(self._kl_stop_triggers))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        final_loss = last_loss.item() if last_loss is not None else 0.0
        self.logger.record("train/loss", final_loss)
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        lr_stats = getattr(self.policy, "get_actor_group_lrs", lambda: {})()
        for name, lr in lr_stats.items():
            self.logger.record(f"train/lr_{name}", lr)
            
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _get_phase_progress_remaining(self) -> float:
        if self._phase_total_timesteps > 0:
            elapsed = max(self.num_timesteps - self._phase_start_num_timesteps, 0)
            ratio = min(elapsed / max(self._phase_total_timesteps, 1), 1.0)
            return 1.0 - ratio
        return self._current_progress_remaining

    @staticmethod
    def _eval_schedule(schedule: Optional[Union[float, Callable[[float], float]]], progress: float) -> Optional[float]:
        if schedule is None:
            return None
        if callable(schedule):
            return float(schedule(progress))
        return float(schedule)

    def _apply_aux_lr_warmup(self) -> None:
        apply_aux_lr_warmup(self)

    def _step_aux_lr_warmup(self) -> None:
        step_aux_lr_warmup(self)

    def _apply_aux_lr_brake(self, mean_kl: float) -> float:
        return apply_aux_lr_brake(self, mean_kl)

    def set_aux_lr_warmup_steps(
        self,
        lora_steps: Optional[int] = None,
        adapter_steps: Optional[int] = None,
        std_steps: Optional[int] = None,
        reset_progress: bool = False,
    ) -> None:
        configure_aux_lr_warmup_steps(
            self,
            lora_steps=lora_steps,
            adapter_steps=adapter_steps,
            std_steps=std_steps,
            reset_progress=reset_progress,
        )

    def set_actor_critic_lr_schedules(
        self,
        actor_schedule: Optional[Union[float, Callable[[float], float]]],
        critic_schedule: Optional[Union[float, Callable[[float], float]]],
    ) -> None:
        assign_actor_critic_lr_schedules(self, actor_schedule, critic_schedule)

    def set_phase_lr_context(self, total_phase_steps: int) -> None:
        configure_phase_lr_context(self, total_phase_steps)

    def _update_learning_rate(self, optimizer: torch.optim.Optimizer) -> None:
        update_learning_rate(self, optimizer)

class TrainingConfig:
    def __init__(self, config_dir: Optional[str] = None, override_dir: Optional[str] = None):
        """
        Initialize training configuration from YAML files.

        Args:
            config_dir: Directory containing base YAML config files (default: rl_configs/)
            override_dir: Optional directory containing override YAML files
        """
        # Load all configs from YAML
        configs = load_configs(config_dir, override_dir, backend="uniad")
        ppo_cfg = configs['ppo_training']
        policy_cfg = configs['policy_network']
        env_cfg = configs['environment']
        paths_cfg = configs['paths']

        # ============ Path configuration ============
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.uniad_config_path = os.path.join(self.project_root, paths_cfg.uniad.config_path)
        self.uniad_checkpoint_path = os.path.join(self.project_root, paths_cfg.uniad.checkpoint_path)

        # Output directories
        self.output_dir = os.path.join(self.project_root, paths_cfg.output.base_dir)
        self.log_dir = os.path.join(self.project_root, paths_cfg.output.log_dir)
        self.checkpoint_dir = os.path.join(self.project_root, paths_cfg.output.checkpoint_dir)
        self.tensorboard_log = os.path.join(self.project_root, paths_cfg.output.tensorboard_log)
        self.vec_normalize_path = os.path.join(self.project_root, paths_cfg.normalization.vec_normalize_path)
        self.eval_results_path = os.path.join(self.project_root, paths_cfg.output.eval_results)

        # ============ Device configuration ============
        self.uniad_device = env_cfg.devices.uniad_device if torch.cuda.is_available() else 'cpu'
        self.policy_device = env_cfg.devices.policy_device if torch.cuda.is_available() else 'cpu'

        # ============ Architecture configuration ============
        self.embed_dims = policy_cfg.architecture.embed_dims
        self.bev_h = policy_cfg.architecture.bev_h
        self.bev_w = policy_cfg.architecture.bev_w
        self.planning_steps = policy_cfg.architecture.planning_steps

        # ============ Normalization ============
        self.normalize_obs = ppo_cfg.normalization.normalize_obs
        self.normalize_reward = ppo_cfg.normalization.normalize_reward
        self.clip_obs = ppo_cfg.normalization.clip_obs
        self.clip_reward = ppo_cfg.normalization.clip_reward

        # ============ PPO algorithm hyperparameters ============
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

        # ============ Training logistics ============
        self.total_timesteps = ppo_cfg.training_logistics.total_timesteps
        self.save_freq = ppo_cfg.training_logistics.save_freq
        self.eval_freq = ppo_cfg.training_logistics.eval_freq
        self.eval_episodes = ppo_cfg.training_logistics.eval_episodes
        self.verbose = ppo_cfg.training_logistics.verbose
        self.episode_info_print_every = ppo_cfg.training_logistics.episode_info_print_every
        self.episode_info_log_to_tb = ppo_cfg.training_logistics.episode_info_log_to_tb

        # ============ Policy configuration ============
        self.with_adapter = policy_cfg.architecture.with_adapter
        self.pretrained_planning_weights = paths_cfg.uniad.get('pretrained_planning_weights')
        if self.pretrained_planning_weights:
            self.pretrained_planning_weights = os.path.join(self.project_root, self.pretrained_planning_weights)

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

        # ============ Phase presets from YAML ============
        self.phase_presets = {}
        for phase_id, preset_data in ppo_cfg.phase_presets.to_dict().items():
            phase_dict = {}
            for key, value in preset_data.items():
                if key == 'phase_behavior':
                    phase_dict[key] = dict(value)
                elif key == 'log_std_bounds':
                    phase_dict[key] = tuple(value)
                else:
                    phase_dict[key] = value
            self.phase_presets[int(phase_id)] = phase_dict

        # ============ Phase schedule from YAML ============
        self.phase_schedule = [tuple(item) for item in ppo_cfg.phase_schedule]

        self.apply_phase_defaults()
        
    def create_dirs(self):
        """创建必要的目录"""
        for dir_path in [self.output_dir, self.log_dir, 
                        self.checkpoint_dir, self.tensorboard_log]:
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
    """
    自定义回调：记录额外的训练指标到 Tensorboard
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # 记录额外的统计信息
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.logger.record('rollout/ep_len_mean', 
                             np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]))
            self.logger.record('rollout/ep_rew_mean', 
                             np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]))
        return True

class EpisodeInfoPrinterCallback(BaseCallback):
    """
    打印并记录环境在 info['episode']['env'] 中提供的诊断统计
    """

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

def create_env(base_env, uniad_model, dataparser, config: TrainingConfig):
    """
    创建并包装环境

    Args:
        base_env: 基础仿真环境
        uniad_model: UniAD 模型
        dataparser: 数据解析器
        config: 训练配置

    Returns:
        wrapped_env: 包装后的环境
    """
    # Load environment config for action bounds
    env_cfg = load_configs(backend="uniad").get('environment')

    # 使用 UniADWrapper 包装环境
    wrapped_env = UniADWrapper(
        env=base_env,
        uniad_model=uniad_model,
        dataparser=dataparser,
        embed_dims=config.embed_dims,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        planning_steps=config.planning_steps,
        uniad_device=config.uniad_device,
        policy_device=config.policy_device,
        lateral_bounds=(env_cfg.action_space.lateral_min, env_cfg.action_space.lateral_max),
        longitudinal_bounds=(env_cfg.action_space.longitudinal_min, env_cfg.action_space.longitudinal_max)
    )

    print("✓ 环境已包装")
    return wrapped_env

def make_vec_env(
    base_env_fn: Callable[[], Any],
    uniad_model,
    dataparser,
    config: TrainingConfig,
    normalize: bool = True,
) -> Any:
    def _init():
        base_env = base_env_fn()
        return create_env(base_env, uniad_model, dataparser, config)
    env = DummyVecEnv([_init])
    env = VecMonitor(env, info_keywords=("env",))
    if normalize and (config.normalize_obs or config.normalize_reward):
        env = VecNormalize(
            env,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            clip_obs=config.clip_obs,
            clip_reward=config.clip_reward,
        )
    return env

def _get_env_action_affine(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract action scale/bias from a (possibly wrapped) SB3 VecEnv/Env.
    Works with VecNormalize, DummyVecEnv, SubprocVecEnv, and nested gym wrappers.
    """
    # 1) Prefer VecEnv API: get_attr works through VecNormalize and VecEnv wrappers
    for scale_key, bias_key in [("action_scale", "action_bias"), ("_action_scale", "_action_bias")]:
        try:
            scale_list = env.get_attr(scale_key)
            bias_list = env.get_attr(bias_key)
            if scale_list and bias_list:
                scale0, bias0 = scale_list[0], bias_list[0]
                # Optional: consistency check if multiple envs
                for s, b in zip(scale_list[1:], bias_list[1:]):
                    if hasattr(s, "detach"):
                        s = s.detach().cpu().numpy()
                    if hasattr(b, "detach"):
                        b = b.detach().cpu().numpy()
                    # compare with first (converted below) if needed
                scale_arr = scale0.detach().cpu().numpy() if hasattr(scale0, "detach") else np.asarray(scale0, np.float32)
                bias_arr  = bias0.detach().cpu().numpy() if hasattr(bias0, "detach") else np.asarray(bias0, np.float32)
                return scale_arr, bias_arr
        except Exception:
            pass

    # 2) Fallback: manually unwrap common wrappers
    base_env = env
    if isinstance(base_env, VecNormalize):
        base_env = base_env.venv

    # If VecEnv, try to access the first env instance (DummyVecEnv only)
    if hasattr(base_env, "envs") and getattr(base_env, "envs", None):
        base_env = base_env.envs[0]

    # Unwrap gym wrappers
    while hasattr(base_env, "env"):
        if hasattr(base_env, "action_scale") or hasattr(base_env, "_action_scale"):
            break
        base_env = base_env.env

    scale = getattr(base_env, "action_scale", getattr(base_env, "_action_scale", None))
    bias  = getattr(base_env, "action_bias",  getattr(base_env, "_action_bias", None))
    if scale is None or bias is None:
        raise AttributeError("Unable to locate env action scale/bias for policy initialization")

    scale_arr = scale.detach().cpu().numpy() if hasattr(scale, "detach") else np.asarray(scale, np.float32)
    bias_arr  = bias.detach().cpu().numpy()  if hasattr(bias, "detach") else np.asarray(bias, np.float32)
    return scale_arr, bias_arr

def create_ppo_model(env, config: TrainingConfig):
    """
    创建 PPO 模型
    
    Args:
        env: 环境
        config: 训练配置
        
    Returns:
        model: PPO 模型
    """
    env_action_scale, env_action_bias = _get_env_action_affine(env)
    policy_kwargs = dict(
        planning_steps=config.planning_steps,
        embed_dims=config.embed_dims,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        with_adapter=config.with_adapter,

        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,

        phase=config.training_phase,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,

        log_std_bounds=config.log_std_bounds,
        log_std_init=config.policy_log_std_init,
        use_expln=config.policy_use_expln,
        full_std=config.policy_full_std,
        latent_sde_scale=config.policy_latent_sde_scale,

        kl_reg_coef=config.kl_reg_coef,
        bc_reg_coef=config.bc_reg_coef,
        unfreeze_adapter_phase2=config.unfreeze_adapter_phase2,
        actor_lora_lr_ratio=config.actor_lora_lr_ratio,
        actor_adapter_lr_ratio=config.actor_adapter_lr_ratio,
        actor_std_lr_ratio=config.actor_std_lr_ratio,
        
        weight_decay_coef=config.optimizer_weight_decay,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4),
        env_action_scale=env_action_scale.tolist(),
        env_action_bias=env_action_bias.tolist(),
    )

    learning_rate = build_learning_rate_schedule(config)
    
    model = UniADPPO(
        policy=CustomUniADPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.tensorboard_log,
        verbose=config.verbose,
        device=config.policy_device,
        grad_accum_steps=config.grad_accum_steps,
    )
    if hasattr(model, "set_aux_lr_warmup_steps"):
        model.set_aux_lr_warmup_steps(
            config.lora_warmup_updates,
            config.adapter_warmup_updates,
            config.std_warmup_updates,
        )
    
    if callable(learning_rate):
        print("✓ PPO 模型已创建 (cosine LR schedule)")
    else:
        print("✓ PPO 模型已创建 (constant LR)")

    if config.pretrained_planning_weights is not None:
        if os.path.exists(config.pretrained_planning_weights):
            print(f"加载预训练权重: {config.pretrained_planning_weights}")
            model.policy.load_pretrained_weights(
                config.pretrained_planning_weights,
                strict=True
            )
            print("✓ 预训练权重已加载")
        else:
            print(f"⚠ 预训练权重文件不存在: {config.pretrained_planning_weights}")
    
    return model

def setup_callbacks(config: TrainingConfig, eval_env=None):
    """
    设置训练回调函数
    
    Args:
        config: 训练配置
        eval_env: 评估环境（可选）
        
    Returns:
        callback: 回调函数列表
    """
    callbacks = []
    
    # 1. 检查点保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.checkpoint_dir,
        name_prefix='ppo_uniad',
        save_replay_buffer=False,
        save_vecnormalize=bool(config.normalize_obs or config.normalize_reward),
    )
    callbacks.append(checkpoint_callback)
    print(f"✓ 检查点保存频率: 每 {config.save_freq} 步")
    
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.checkpoint_dir, 'best_model'),
            log_path=config.log_dir,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        print(f"✓ 评估频率: 每 {config.eval_freq} 步")
    
    # 3. Tensorboard 回调
    tensorboard_callback = TensorboardCallback()
    callbacks.append(tensorboard_callback)
    print("✓ Tensorboard 日志已启用")

    episode_info_callback = EpisodeInfoPrinterCallback(
        print_every=getattr(config, "episode_info_print_every", 1),
        log_to_tb=getattr(config, "episode_info_log_to_tb", True),
    )
    callbacks.append(episode_info_callback)
    print(f"✓ Episode 诊断打印: 每 {getattr(config, 'episode_info_print_every', 1)} 个 episode")
    
    return CallbackList(callbacks)

def _to_schedule(value: Optional[Union[float, Callable[[float], float]]]):
    if value is None:
        return None
    if callable(value):
        return value
    return get_schedule_fn(value)

def ensure_rollout_buffer_size(model: UniADPPO) -> None:
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

def apply_phase_settings(
    model: UniADPPO,
    config: TrainingConfig,
    phase_id: int,
    phase_steps: int,
) -> None:
    previous_n_steps = getattr(model, "n_steps", None)
    config.apply_phase_defaults(phase_id)

    model.n_steps = config.n_steps
    model.batch_size = config.batch_size
    model.n_epochs = config.n_epochs
    clip_range_schedule = _to_schedule(config.clip_range)
    if clip_range_schedule is None:
        raise ValueError("clip_range must be provided for each phase preset")
    model.clip_range = clip_range_schedule
    model.clip_range_vf = _to_schedule(getattr(config, "clip_range_vf", None))
    model.ent_coef = config.ent_coef
    model.vf_coef = config.vf_coef
    model.target_kl = config.target_kl
    model.grad_accum_steps = max(1, getattr(config, "grad_accum_steps", getattr(model, "grad_accum_steps", 1)))

    policy = model.policy
    policy.configure_phase(
        config.training_phase,
        config.actor_lr,
        config.critic_lr,
        config.log_std_bounds,
        config.kl_reg_coef,
        config.bc_reg_coef,
        config.phase_behavior,
        actor_lora_lr_ratio=config.actor_lora_lr_ratio,
        actor_adapter_lr_ratio=config.actor_adapter_lr_ratio,
        actor_std_lr_ratio=config.actor_std_lr_ratio,
    )

    if previous_n_steps != model.n_steps:
        ensure_rollout_buffer_size(model)


def run_phase_training(
    model: UniADPPO,
    config: TrainingConfig,
    callbacks: CallbackList,
) -> int:
    total_steps = int(config.total_timesteps)
    plan = build_phase_plan(total_steps, config.phase_schedule, config.training_phase)
    if not plan:
        print("未配置可用的 phase 训练计划，跳过训练")
        return 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    global_step = 0
    for phase_idx, (phase_id, phase_steps) in enumerate(plan):
        apply_phase_settings(model, config, phase_id, phase_steps)
        actor_schedule = build_learning_rate_schedule(config, base_lr=config.actor_lr)
        critic_schedule = build_learning_rate_schedule(config, base_lr=config.critic_lr)
        if hasattr(model, "set_actor_critic_lr_schedules"):
            model.set_actor_critic_lr_schedules(actor_schedule, critic_schedule)
        if hasattr(model, "set_phase_lr_context"):
            model.set_phase_lr_context(phase_steps)
        print(f"\n--- Phase {phase_id} | steps={phase_steps:,} (global={global_step:,}/{total_steps:,}) ---")
        tb_name = f"PPO_phase{phase_id}_{timestamp}"
        model.learn(
            total_timesteps=phase_steps,
            callback=callbacks,
            log_interval=1,
            tb_log_name=tb_name,
            reset_num_timesteps=True,
            progress_bar=True,
        )
        global_step += phase_steps
        print(f"Phase {phase_id} 完成，累计训练步数 {global_step:,}/{total_steps:,}")
    return global_step

def train(
    base_env_fn: Callable[[], Any],
    config: TrainingConfig,
    eval_env_fn: Optional[Callable[[], Any]] = None,
    model_path: Optional[str] = None,
):
    """
    Args:
        base_env_fn: 创建基础仿真环境的函数
        config: 训练配置
        eval_env_fn: 可选的评估环境创建函数
    """
    print("\n" + "="*60)
    print("开始 PPO 训练")
    print("="*60 + "\n")

    config.create_dirs()

    print("\n[1/5] 加载 UniAD 模型...")
    uniad_config = {
        'config_path': config.uniad_config_path,
        'checkpoint_path': config.uniad_checkpoint_path,
        'device': config.uniad_device
    }
    uniad_model = create_uniad(uniad_config)
    print("✓ UniAD 模型加载完成")

    print("\n[2/5] 创建训练环境...")
    env = make_vec_env(base_env_fn, uniad_model, parse_uniad_obs, config, normalize=False)

    eval_env = None
    if eval_env_fn is not None:
        print("\n[3/5] 创建评估环境...")
        eval_env = make_vec_env(eval_env_fn, uniad_model, parse_uniad_obs, config, normalize=False)

    print("\n[4/5] 创建 PPO 模型...")
    if model_path is not None:
        _, params, _ = load_from_zip_file(model_path, device=config.policy_device)
        for key in list(params.keys()):
            if "optimizer" in key:
                params.pop(key)
        model = create_ppo_model(env, config)
        model.set_parameters(params, exact_match=False, device=config.policy_device)
    else:
        model = create_ppo_model(env, config)

    callbacks = setup_callbacks(config, eval_env=eval_env)

    print("\n[5/5] 开始训练...")
    print(f"总训练步数: {config.total_timesteps:,}")
    print(f"UniAD device: {next(uniad_model.parameters()).device}")
    print(f"Policy device: {next(model.policy.parameters()).device}")
    print(f"Tensorboard 日志: {config.tensorboard_log}")
    print("\n" + "="*60 + "\n")

    try:
        trained_steps = run_phase_training(model, config, callbacks)
        if trained_steps < config.total_timesteps:
            print(f"⚠ 计划训练步数 {config.total_timesteps:,}, 实际执行 {trained_steps:,}")
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60 + "\n")

        final_model_path = os.path.join(config.checkpoint_dir, 'new reward weight, double position, success bonus, living reward')
        model.save(final_model_path)
        print(f"✓ 最终模型已保存: {final_model_path}")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        save_path = os.path.join(config.checkpoint_dir, 'interrupted_model')
        model.save(save_path)
        print(f"✓ 模型已保存: {save_path}")
    
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        save_path = os.path.join(config.checkpoint_dir, 'error_model')
        model.save(save_path)
        print(f"✓ 模型已保存: {save_path}")

def evaluate(model_path: str, base_env_fn: Callable[[], Any], config: TrainingConfig, n_episodes: int = 10):
    """
    评估训练好的模型

    Args:
        model_path: 模型路径
        base_env: 基础环境
        config: 训练配置
        n_episodes: 评估 episode 数量
    """
    print(f"\n加载模型进行评估: {model_path}")
    uniad_config = {
        'config_path': config.uniad_config_path,
        'checkpoint_path': config.uniad_checkpoint_path,
        'device': config.uniad_device
    }
    uniad_model = create_uniad(uniad_config)

    env = make_vec_env(base_env_fn, uniad_model, parse_uniad_obs, config, normalize=False)
    if (config.normalize_obs or config.normalize_reward) and os.path.exists(config.vec_normalize_path):
        env = VecNormalize.load(config.vec_normalize_path, env)
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_obs = config.normalize_obs
        env.norm_reward = config.normalize_reward
    _, params, _ = load_from_zip_file(model_path, device=config.policy_device)
    for key in list(params.keys()):
        if "optimizer" in key:
            params.pop(key)

    model = create_ppo_model(env, config)
    model.policy.configure_phase(
        config.training_phase,
        config.actor_lr,
        config.critic_lr,
        config.log_std_bounds,
        config.kl_reg_coef,
        config.bc_reg_coef,
        config.phase_behavior,
        actor_lora_lr_ratio=config.actor_lora_lr_ratio,
        actor_adapter_lr_ratio=config.actor_adapter_lr_ratio,
        actor_std_lr_ratio=config.actor_std_lr_ratio,
    )
    model.set_parameters(params, exact_match=False, device=config.policy_device)

    all_eval_metrics = []

    for episode_idx in range(n_episodes):
        n_envs = getattr(env, "num_envs", 1)
        trackers = [EpisodeMetricTracker() for _ in range(n_envs)]
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []

        model.policy.reset_noise(n_envs)
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs, reset_infos = reset_output
            if isinstance(reset_infos, (list, tuple)):
                for idx in range(min(n_envs, len(reset_infos))):
                    trackers[idx].update(reset_infos[idx])
        else:
            obs = reset_output
        per_env_rewards = [0.0 for _ in range(n_envs)]
        per_env_lengths = [0 for _ in range(n_envs)]
        completed = 0
        metric_keys = ['NC', 'DAC', 'TTC', 'COM', 'RC']
        sum_steps = 0

        done = False
        done_flags = [False] * n_envs

        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, done, infos = env.step(action)

            for idx in range(n_envs):
                trackers[idx].update(infos)
                per_env_rewards[idx] += float(rewards)
                per_env_lengths[idx] += 1

                if done_flags[idx]:
                    ep_reward = per_env_rewards[idx]
                    ep_length = per_env_lengths[idx]
                    metrics = trackers[idx].finalize()
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episode_metrics.append(metrics)

                    episode_id = len(episode_rewards)
                    metric_str = ", ".join(f"{k}={metrics[k]:.3f}" for k in metric_keys)
                    print(f"Episode {episode_id}/{n_episodes}: Reward={ep_reward:.2f}, Length={ep_length}, {metric_str}")

                    per_env_rewards[idx] = 0.0
                    per_env_lengths[idx] = 0
                    trackers[idx] = EpisodeMetricTracker()

            sum_steps += 1
            if sum_steps > 200:
                done = True
                break

        valid_indices = [i for i, metrics in enumerate(episode_metrics) if metrics['RC'] >= 0.01]

        if not valid_indices:
            print(f"第 {episode_idx + 1} 次评估: 所有episode的RC都低于0.01, 跳过此轮评估")
            continue

        valid_rewards = [episode_rewards[i] for i in valid_indices]
        valid_lengths = [episode_lengths[i] for i in valid_indices]
        valid_metrics = [episode_metrics[i] for i in valid_indices]

        mean_reward = np.mean(valid_rewards)
        std_reward = np.std(valid_rewards) if len(valid_rewards) > 1 else 0.0
        mean_length = np.mean(valid_lengths)
        std_length = np.std(valid_lengths) if len(valid_lengths) > 1 else 0.0
        mean_metrics = {k: np.mean([m[k] for m in valid_metrics]) for k in metric_keys}

        eval_result = {
            'episode_idx': episode_idx + 1,
            'valid_count': len(valid_indices),
            'total_count': len(episode_metrics),
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'std_length': std_length,
            'mean_metrics': mean_metrics.copy()  # 包含 NC, DAC, TTC, COM, RC
        }
        all_eval_metrics.append(eval_result)

        eval_results = []
        eval_results.append(f"\n第 {episode_idx + 1} 次评估结果 (有效episodes: {len(valid_indices)}/{len(episode_metrics)}):")
        eval_results.append(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        eval_results.append(f"  平均长度: {mean_length:.1f} ± {std_length:.1f}")
        eval_results.append("  平均指标: " + ", ".join(f"{k}={mean_metrics[k]:.3f}" for k in metric_keys))
        eval_results.append("")

        results_str = "\n".join(eval_results)

        print(results_str)

        with open(config.eval_results_path, 'a', encoding='utf-8') as f:
            f.write(results_str + "\n")

    if all_eval_metrics:
        overall_mean_reward = np.mean([r['mean_reward'] for r in all_eval_metrics])
        overall_std_reward = np.std([r['mean_reward'] for r in all_eval_metrics]) if len(all_eval_metrics) > 1 else 0.0
        overall_mean_length = np.mean([r['mean_length'] for r in all_eval_metrics])
        overall_std_length = np.std([r['mean_length'] for r in all_eval_metrics]) if len(all_eval_metrics) > 1 else 0.0

        metric_keys = ['NC', 'DAC', 'TTC', 'COM', 'RC']
        overall_mean_metrics = {}
        for key in metric_keys:
            metric_values = [r['mean_metrics'][key] for r in all_eval_metrics]
            overall_mean_metrics[key] = np.mean(metric_values)

        print(f"\n{'='*60}")
        print(f"{n_episodes}轮评估总体平均结果 (基于 {len(all_eval_metrics)} 次有效评估):")
        print(f"  平均奖励: {overall_mean_reward:.2f} ± {overall_std_reward:.2f}")
        print(f"  平均长度: {overall_mean_length:.1f} ± {overall_std_length:.1f}")
        print(f"  平均指标: NC={overall_mean_metrics['NC']:.3f}, DAC={overall_mean_metrics['DAC']:.3f}, TTC={overall_mean_metrics['TTC']:.3f}, COM={overall_mean_metrics['COM']:.3f}, RC={overall_mean_metrics['RC']:.3f}")
        print(f"{'='*60}")
