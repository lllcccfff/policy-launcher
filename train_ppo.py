"""
PPO 训练脚本
使用 UniADWrapper 包装的环境和 CustomUniADPolicy 进行强化学习训练
"""
import os
import sys
import pprint
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime

# Stable-Baselines3
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from rl_framework.uniad.env_wrapper import UniADWrapper
from rl_framework.uniad.policy_network import CustomUniADPolicy
from rl_framework.uniad.loader import create_uniad
from rl_framework.uniad.dataparser import parse_uniad_obs
from rl_framework.common.metrics import EpisodeMetricTracker
from rl_framework.rl_configs.config_loader import load_configs
from rl_framework.rl_utils.ppo_trainer import ADPPO
from rl_framework.rl_utils.training_utils import (
    TrainingConfig,
    TensorboardCallback,
    EpisodeInfoPrinterCallback,
    _to_schedule,
    _get_env_action_affine,
    ensure_rollout_buffer_size,
    build_phase_plan,
    build_learning_rate_schedule,
)

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
    
    model = ADPPO(
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
    model: ADPPO,
    config: TrainingConfig,
    callbacks: CallbackList,
) -> int:
    total_steps = int(config.total_timesteps)
    plan = build_phase_plan(total_steps, config.phase_schedule, config.training_phase)
    if not plan:
        print("未配置可用的 phase 训练计划，跳过训练")
        return 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_name = f"PPO_{timestamp}"
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
        model.learn(
            total_timesteps=phase_steps,
            callback=callbacks,
            log_interval=1,
            tb_log_name=tb_name,
            reset_num_timesteps=False,
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
        'device': config.uniad_device,
        'AD_root': config.AD_root,
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
        'device': config.uniad_device,
        'AD_root': config.AD_root,
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
