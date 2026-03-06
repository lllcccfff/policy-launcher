"""
PPO training script for VAD backend only.
"""
import argparse
import os
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import torch
from metadrive.envs.grpc_client_env import GrpcClientEnv

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from rl_framework.common.metrics import EpisodeMetricTracker
from rl_framework.rl_configs.config_loader import load_configs
from rl_framework.rl_utils.ppo_trainer import ADPPO
from rl_framework.rl_utils.training_utils import (
    TrainingConfig as BaseTrainingConfig,
    TensorboardCallback,
    EpisodeInfoPrinterCallback,
    _to_schedule,
    _get_env_action_affine,
    build_learning_rate_schedule,
    ensure_rollout_buffer_size,
    build_phase_plan,
)
from rl_framework.vad.env_wrapper import VADWrapper
from rl_framework.vad.policy_network import CustomVADPolicy
from rl_framework.vad.loader import create_vad, extract_vad_planning_weights
from rl_framework.vad.dataparser import parse_vad_obs


class TrainingConfig(BaseTrainingConfig):
    def __init__(
        self,
        config_dir: Optional[str] = None,
        override_dir: Optional[str] = None,
    ):
        super().__init__(config_dir=config_dir, override_dir=override_dir, backend="vad")
        vad_paths = self.paths_cfg.get("vad")
        if vad_paths is None:
            raise ValueError("paths_vad.yaml must define `vad` section")
        self.vad_config_path = self.resolve_ad_path(vad_paths.config_path)
        self.vad_checkpoint_path = self.resolve_ad_path(vad_paths.checkpoint_path)
        self.ego_fut_mode = 3
        self.use_sde = True
        # VAD has no adapter branch, keep compatibility placeholders for shared LR toolchain.
        self.actor_adapter_lr_ratio = 0.0
        self.adapter_warmup_updates = 0
        self.grpc_host = os.environ.get("VAD_GRPC_HOST", "127.0.0.1")
        self.grpc_port = int(os.environ.get("VAD_GRPC_PORT", "50052"))
        self.grpc_timeout_sec = float(os.environ.get("VAD_GRPC_TIMEOUT_SEC", "10.0"))
        self.grpc_auto_wait_ready = (
            os.environ.get("VAD_GRPC_AUTO_WAIT_READY", "1").lower() not in {"0", "false", "no"}
        )
        self.apply_phase_defaults(self.training_phase)

    def apply_phase_defaults(self, phase: Optional[int] = None):
        super().apply_phase_defaults(phase=phase)
        self.phase_behavior["train_adapter"] = False
        self.actor_adapter_lr_ratio = 0.0


def create_env(base_env, vad_model, dataparser, config: TrainingConfig):
    env_cfg = load_configs(backend="vad").get("environment")
    wrapped_env = VADWrapper(
        env=base_env,
        vad_model=vad_model,
        dataparser=dataparser,
        embed_dims=config.embed_dims,
        planning_steps=config.planning_steps,
        max_agent_queries=env_cfg.observation_space.max_agent_queries,
        max_map_queries=env_cfg.observation_space.max_map_queries,
        ego_fut_mode=env_cfg.observation_space.ego_fut_mode,
        bev_h=env_cfg.observation_space.bev_h,
        bev_w=env_cfg.observation_space.bev_w,
        uniad_device=config.uniad_device,
        policy_device=config.policy_device,
        lateral_bounds=(env_cfg.action_space.lateral_min, env_cfg.action_space.lateral_max),
        longitudinal_bounds=(
            env_cfg.action_space.longitudinal_min,
            env_cfg.action_space.longitudinal_max,
        ),
    )
    print("✓ 环境已包装 (VAD)")
    return wrapped_env


def make_vec_env(
    vad_model,
    dataparser,
    config: TrainingConfig,
    normalize: bool = True,
) -> Any:
    def _init():
        base_env = GrpcClientEnv(
            host=config.grpc_host,
            port=config.grpc_port,
            timeout_sec=config.grpc_timeout_sec,
            auto_wait_ready=config.grpc_auto_wait_ready,
        )
        return create_env(base_env, vad_model, dataparser, config)

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


def create_ppo_model(env, config: TrainingConfig, planning_weights: Dict[str, Any]):
    env_action_scale, env_action_bias = _get_env_action_affine(env)
    policy_kwargs = dict(
        planning_steps=config.planning_steps,
        embed_dims=config.embed_dims,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        ego_fut_mode=config.ego_fut_mode,
        planning_weights=planning_weights,
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
    model = ADPPO(
        policy=CustomVADPolicy,
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
            lora_steps=config.lora_warmup_updates,
            adapter_steps=config.adapter_warmup_updates,
            std_steps=config.std_warmup_updates,
        )
    lr_mode = "cosine LR schedule" if callable(learning_rate) else "constant LR"
    print(f"✓ PPO 模型已创建 ({lr_mode}, VAD)")
    return model


def setup_callbacks(config: TrainingConfig, eval_env=None):
    callbacks = []
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.checkpoint_dir,
        name_prefix="ppo_vad",
        save_replay_buffer=False,
        save_vecnormalize=bool(config.normalize_obs or config.normalize_reward),
    )
    callbacks.append(checkpoint_callback)
    print(f"✓ 检查点保存频率: 每 {config.save_freq} 步")

    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.checkpoint_dir, "best_model"),
            log_path=config.log_dir,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        print(f"✓ 评估频率: 每 {config.eval_freq} 步")

    callbacks.append(TensorboardCallback())
    print("✓ Tensorboard 日志已启用")

    callbacks.append(
        EpisodeInfoPrinterCallback(
            print_every=getattr(config, "episode_info_print_every", 1),
            log_to_tb=getattr(config, "episode_info_log_to_tb", True),
        )
    )
    print(
        f"✓ Episode 诊断打印: 每 {getattr(config, 'episode_info_print_every', 1)} 个 episode"
    )
    return CallbackList(callbacks)


def apply_phase_settings(
    model: ADPPO,
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
    model.grad_accum_steps = max(
        1, getattr(config, "grad_accum_steps", getattr(model, "grad_accum_steps", 1))
    )

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
    model: ADPPO,
    config: TrainingConfig,
    callbacks: CallbackList,
) -> int:
    total_steps = int(config.total_timesteps)
    plan = build_phase_plan(total_steps, config.phase_schedule, config.training_phase)
    if not plan:
        print("未配置可用的 phase 训练计划，跳过训练")
        return 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_step = 0
    for phase_id, phase_steps in plan:
        apply_phase_settings(model, config, phase_id, phase_steps)
        actor_schedule = build_learning_rate_schedule(config, base_lr=config.actor_lr)
        critic_schedule = build_learning_rate_schedule(config, base_lr=config.critic_lr)
        if hasattr(model, "set_actor_critic_lr_schedules"):
            model.set_actor_critic_lr_schedules(actor_schedule, critic_schedule)
        if hasattr(model, "set_phase_lr_context"):
            model.set_phase_lr_context(phase_steps)
        print(
            f"\n--- Phase {phase_id} | steps={phase_steps:,} "
            f"(global={global_step:,}/{total_steps:,}) ---"
        )
        tb_name = f"PPO_VAD_phase{phase_id}_{timestamp}"
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


def _validate_planning_alignment(
    config: TrainingConfig,
    planning_weights: Dict[str, Any],
) -> None:
    return


def train(
    config: TrainingConfig,
    with_eval_env: bool = True,
    model_path: Optional[str] = None,
):
    print("\n" + "=" * 60)
    print("开始 VAD PPO 训练")
    print("=" * 60 + "\n")

    config.create_dirs()

    print("\n[1/5] 加载 VAD 模型...")
    vad_config = {
        "config_path": config.vad_config_path,
        "checkpoint_path": config.vad_checkpoint_path,
        "device": config.uniad_device,
        "AD_root": config.AD_root,
    }
    vad_model = create_vad(vad_config)
    planning_weights = extract_vad_planning_weights(vad_model)
    _validate_planning_alignment(config, planning_weights)
    print("✓ VAD 模型加载完成")
    print(
        f"✓ gRPC env: {config.grpc_host}:{config.grpc_port} "
        f"(timeout={config.grpc_timeout_sec}s)"
    )

    print("\n[2/5] 创建训练环境...")
    env = make_vec_env(vad_model, parse_vad_obs, config, normalize=False)

    eval_env = None
    if with_eval_env:
        print("\n[3/5] 创建评估环境...")
        eval_env = make_vec_env(vad_model, parse_vad_obs, config, normalize=False)

    print("\n[4/5] 创建 PPO 模型...")
    if model_path is not None:
        _, params, _ = load_from_zip_file(model_path, device=config.policy_device)
        for key in list(params.keys()):
            if "optimizer" in key:
                params.pop(key)
        model = create_ppo_model(env, config, planning_weights=planning_weights)
        model.set_parameters(params, exact_match=False, device=config.policy_device)
    else:
        model = create_ppo_model(env, config, planning_weights=planning_weights)

    callbacks = setup_callbacks(config, eval_env=eval_env)

    print("\n[5/5] 开始训练...")
    print(f"总训练步数: {config.total_timesteps:,}")
    model_impl = vad_model.module if hasattr(vad_model, "module") else vad_model
    print(f"VAD device: {next(model_impl.parameters()).device}")
    print(f"Policy device: {next(model.policy.parameters()).device}")
    print(f"Tensorboard 日志: {config.tensorboard_log}")
    print("\n" + "=" * 60 + "\n")

    try:
        trained_steps = run_phase_training(model, config, callbacks)
        if trained_steps < config.total_timesteps:
            print(f"⚠ 计划训练步数 {config.total_timesteps:,}, 实际执行 {trained_steps:,}")

        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60 + "\n")

        final_model_path = os.path.join(config.checkpoint_dir, "ppo_vad_final")
        model.save(final_model_path)
        print(f"✓ 最终模型已保存: {final_model_path}")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        save_path = os.path.join(config.checkpoint_dir, "ppo_vad_interrupted")
        model.save(save_path)
        print(f"✓ 模型已保存: {save_path}")

    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback

        traceback.print_exc()
        save_path = os.path.join(config.checkpoint_dir, "ppo_vad_error")
        model.save(save_path)
        print(f"✓ 模型已保存: {save_path}")


def evaluate(
    model_path: str,
    config: TrainingConfig,
    n_episodes: int = 10,
):
    print(f"\n加载模型进行评估: {model_path}")
    vad_config = {
        "config_path": config.vad_config_path,
        "checkpoint_path": config.vad_checkpoint_path,
        "device": config.uniad_device,
        "AD_root": config.AD_root,
    }
    vad_model = create_vad(vad_config)
    planning_weights = extract_vad_planning_weights(vad_model)
    _validate_planning_alignment(config, planning_weights)
    print(
        f"✓ gRPC eval env: {config.grpc_host}:{config.grpc_port} "
        f"(timeout={config.grpc_timeout_sec}s)"
    )

    env = make_vec_env(vad_model, parse_vad_obs, config, normalize=False)
    if (config.normalize_obs or config.normalize_reward) and os.path.exists(
        config.vec_normalize_path
    ):
        env = VecNormalize.load(config.vec_normalize_path, env)
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_obs = config.normalize_obs
        env.norm_reward = config.normalize_reward

    _, params, _ = load_from_zip_file(model_path, device=config.policy_device)
    for key in list(params.keys()):
        if "optimizer" in key:
            params.pop(key)
    model = create_ppo_model(env, config, planning_weights=planning_weights)
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
    metric_keys = ["NC", "DAC", "TTC", "COM", "RC"]

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

        done = False
        done_flags = [False] * n_envs
        sum_steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, infos = env.step(action)

            if hasattr(terminated, "__iter__"):
                done_flags = [t or tr for t, tr in zip(terminated, truncated)]
            else:
                done_flags = [terminated or truncated] * n_envs
            done = all(done_flags)

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
                    print(
                        f"Episode {episode_id}/{n_episodes}: "
                        f"Reward={ep_reward:.2f}, Length={ep_length}, {metric_str}"
                    )

                    per_env_rewards[idx] = 0.0
                    per_env_lengths[idx] = 0
                    trackers[idx] = EpisodeMetricTracker()

            sum_steps += 1
            if sum_steps > 200:
                done = True
                break

        valid_indices = [i for i, metrics in enumerate(episode_metrics) if metrics["RC"] >= 0.01]
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
            "episode_idx": episode_idx + 1,
            "valid_count": len(valid_indices),
            "total_count": len(episode_metrics),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "mean_metrics": mean_metrics.copy(),
        }
        all_eval_metrics.append(eval_result)

        result_lines = []
        result_lines.append(
            f"\n第 {episode_idx + 1} 次评估结果 (有效episodes: {len(valid_indices)}/{len(episode_metrics)}):"
        )
        result_lines.append(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        result_lines.append(f"  平均长度: {mean_length:.1f} ± {std_length:.1f}")
        result_lines.append(
            "  平均指标: " + ", ".join(f"{k}={mean_metrics[k]:.3f}" for k in metric_keys)
        )
        result_lines.append("")
        result_text = "\n".join(result_lines)
        print(result_text)
        with open(config.eval_results_path, "a", encoding="utf-8") as f:
            f.write(result_text + "\n")

    if all_eval_metrics:
        overall_mean_reward = np.mean([r["mean_reward"] for r in all_eval_metrics])
        overall_std_reward = (
            np.std([r["mean_reward"] for r in all_eval_metrics])
            if len(all_eval_metrics) > 1
            else 0.0
        )
        overall_mean_length = np.mean([r["mean_length"] for r in all_eval_metrics])
        overall_std_length = (
            np.std([r["mean_length"] for r in all_eval_metrics])
            if len(all_eval_metrics) > 1
            else 0.0
        )
        overall_mean_metrics = {}
        for key in metric_keys:
            metric_values = [r["mean_metrics"][key] for r in all_eval_metrics]
            overall_mean_metrics[key] = np.mean(metric_values)

        print(f"\n{'=' * 60}")
        print(f"{n_episodes}轮评估总体平均结果 (基于 {len(all_eval_metrics)} 次有效评估):")
        print(f"  平均奖励: {overall_mean_reward:.2f} ± {overall_std_reward:.2f}")
        print(f"  平均长度: {overall_mean_length:.1f} ± {overall_std_length:.1f}")
        print(
            "  平均指标: "
            f"NC={overall_mean_metrics['NC']:.3f}, "
            f"DAC={overall_mean_metrics['DAC']:.3f}, "
            f"TTC={overall_mean_metrics['TTC']:.3f}, "
            f"COM={overall_mean_metrics['COM']:.3f}, "
            f"RC={overall_mean_metrics['RC']:.3f}"
        )
        print(f"{'=' * 60}")
    env.close()
