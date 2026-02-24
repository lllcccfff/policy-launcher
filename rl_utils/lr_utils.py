import math
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


# =========================
# Policy-side LR utilities
# =========================

def resolve_actor_group_lr(policy: Any, group_name: str) -> float:
    action_lr = float(getattr(policy, "actor_lr", 0.0))
    base = float(getattr(policy, "actor_lr_base", action_lr))
    aux_brake = float(getattr(policy, "actor_aux_brake_scale", 1.0))
    if group_name == "lora":
        return base * float(getattr(policy, "actor_lora_lr_ratio", 1.0)) * float(
            getattr(policy, "actor_lora_warmup_scale", 1.0)
        ) * aux_brake
    if group_name == "adapter":
        return base * float(getattr(policy, "actor_adapter_lr_ratio", 1.0)) * float(
            getattr(policy, "actor_adapter_warmup_scale", 1.0)
        ) * aux_brake
    if group_name == "log_std":
        return base * float(getattr(policy, "actor_std_lr_ratio", 1.0)) * float(
            getattr(policy, "actor_std_warmup_scale", 1.0)
        ) * aux_brake
    return action_lr


def update_policy_optimizer_lrs(policy: Any) -> None:
    if not hasattr(policy, "optimizer") or policy.optimizer is None:
        return
    group_names = getattr(policy, "_actor_group_names", ())
    for name in group_names:
        idx = getattr(policy, "_actor_group_indices", {}).get(name)
        if idx is None or idx >= len(policy.optimizer.param_groups):
            continue
        group = policy.optimizer.param_groups[idx]
        lr = resolve_actor_group_lr(policy, name)
        if not any(p.requires_grad for p in group.get("params", [])):
            lr = 0.0
        group["lr"] = lr
    critic_idx = getattr(policy, "_critic_group_idx", None)
    if critic_idx is not None and critic_idx < len(policy.optimizer.param_groups):
        policy.optimizer.param_groups[critic_idx]["lr"] = policy.critic_lr


def set_policy_actor_lr(policy: Any, lr: float) -> None:
    policy.actor_lr = float(lr)
    policy.actor_lr_base = float(lr)
    update_policy_optimizer_lrs(policy)


def set_policy_actor_aux_scales(
    policy: Any,
    lora_scale: float,
    adapter_scale: float,
    std_scale: Optional[float] = None,
) -> None:
    policy.actor_lora_warmup_scale = float(max(0.0, min(lora_scale, 1.0)))
    policy.actor_adapter_warmup_scale = float(max(0.0, min(adapter_scale, 1.0)))
    std_scale = 1.0 if std_scale is None else std_scale
    policy.actor_std_warmup_scale = float(max(0.0, min(std_scale, 1.0)))
    update_policy_optimizer_lrs(policy)


def set_policy_actor_aux_brake(policy: Any, scale: float) -> None:
    scale = float(max(0.0, min(scale, 1.0)))
    if abs(scale - float(getattr(policy, "actor_aux_brake_scale", 1.0))) < 1e-6:
        return
    policy.actor_aux_brake_scale = scale
    update_policy_optimizer_lrs(policy)


def collect_actor_group_lrs(policy: Any) -> Dict[str, float]:
    lrs: Dict[str, float] = {}
    if not hasattr(policy, "optimizer") or policy.optimizer is None:
        return lrs
    group_names = getattr(policy, "_actor_group_names", ())
    for name in group_names:
        idx = getattr(policy, "_actor_group_indices", {}).get(name)
        if idx is None or idx >= len(policy.optimizer.param_groups):
            continue
        lrs[name] = float(policy.optimizer.param_groups[idx].get("lr", 0.0))
    return lrs


# =========================
# Trainer-side LR utilities
# =========================

def aux_lr_scale(progress: int, warmup_steps: int) -> float:
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps <= 0:
        return 1.0
    progress = min(max(progress, 0), warmup_steps)
    return progress / float(warmup_steps)


def apply_aux_lr_warmup(trainer: Any) -> None:
    policy = trainer.policy
    if not hasattr(policy, "set_actor_aux_lr_scales"):
        return
    train_lora = getattr(policy, "train_lora", False)
    train_adapter = getattr(policy, "train_adapter", False)
    train_std = getattr(policy, "train_log_std", False)
    lora_scale = aux_lr_scale(trainer._lora_warmup_progress, trainer._lora_warmup_steps) if train_lora else 0.0
    adapter_scale = (
        aux_lr_scale(trainer._adapter_warmup_progress, trainer._adapter_warmup_steps) if train_adapter else 0.0
    )
    std_scale = aux_lr_scale(trainer._std_warmup_progress, trainer._std_warmup_steps) if train_std else 0.0
    policy.set_actor_aux_lr_scales(lora_scale, adapter_scale, std_scale)


def step_aux_lr_warmup(trainer: Any) -> None:
    updated = False
    policy = trainer.policy
    if getattr(policy, "train_lora", False) and trainer._lora_warmup_progress < trainer._lora_warmup_steps:
        trainer._lora_warmup_progress += 1
        updated = True
    if getattr(policy, "train_adapter", False) and trainer._adapter_warmup_progress < trainer._adapter_warmup_steps:
        trainer._adapter_warmup_progress += 1
        updated = True
    if getattr(policy, "train_log_std", False) and trainer._std_warmup_progress < trainer._std_warmup_steps:
        trainer._std_warmup_progress += 1
        updated = True
    if updated:
        apply_aux_lr_warmup(trainer)


def apply_aux_lr_brake(trainer: Any, mean_kl: float) -> float:
    if trainer.target_kl is None or mean_kl <= trainer.target_kl:
        if hasattr(trainer.policy, "set_actor_aux_brake_scale"):
            trainer.policy.set_actor_aux_brake_scale(1.0)
        return 1.0
    scale = math.sqrt(max(trainer.target_kl, 1e-12) / (mean_kl + 1e-8))
    scale = float(np.clip(scale, 0.0, 1.0))
    if hasattr(trainer.policy, "set_actor_aux_brake_scale"):
        trainer.policy.set_actor_aux_brake_scale(scale)
    return scale


def adjust_aux_warmup_for_actor_lr(trainer: Any, prev_lr: float, new_lr: float) -> None:
    prev_lr = float(prev_lr)
    new_lr = float(new_lr)
    if prev_lr <= 0 or new_lr <= 0 or abs(prev_lr - new_lr) < 1e-12:
        return
    ratio = prev_lr / new_lr
    if ratio <= 0:
        return

    def _rescale(progress: float, steps: int) -> float:
        if steps <= 0:
            return 0.0
        progress = float(progress) * ratio
        progress = max(0.0, min(progress, float(steps)))
        return progress

    if getattr(trainer.policy, "train_lora", False):
        trainer._lora_warmup_progress = _rescale(trainer._lora_warmup_progress, trainer._lora_warmup_steps)
    if getattr(trainer.policy, "train_adapter", False):
        trainer._adapter_warmup_progress = _rescale(trainer._adapter_warmup_progress, trainer._adapter_warmup_steps)
    if getattr(trainer.policy, "train_log_std", False):
        trainer._std_warmup_progress = _rescale(trainer._std_warmup_progress, trainer._std_warmup_steps)
    apply_aux_lr_warmup(trainer)


def configure_aux_lr_warmup_steps(
    trainer: Any,
    lora_steps: Optional[int] = None,
    adapter_steps: Optional[int] = None,
    std_steps: Optional[int] = None,
    reset_progress: bool = False,
) -> None:
    if lora_steps is not None:
        trainer._lora_warmup_steps = max(0, int(lora_steps))
    if reset_progress:
        trainer._lora_warmup_progress = 0
    trainer._lora_warmup_progress = min(trainer._lora_warmup_progress, trainer._lora_warmup_steps)

    if adapter_steps is not None:
        trainer._adapter_warmup_steps = max(0, int(adapter_steps))
    if reset_progress:
        trainer._adapter_warmup_progress = 0
    trainer._adapter_warmup_progress = min(trainer._adapter_warmup_progress, trainer._adapter_warmup_steps)

    if std_steps is not None:
        trainer._std_warmup_steps = max(0, int(std_steps))
    if reset_progress:
        trainer._std_warmup_progress = 0
    trainer._std_warmup_progress = min(trainer._std_warmup_progress, trainer._std_warmup_steps)
    apply_aux_lr_warmup(trainer)


def assign_actor_critic_lr_schedules(
    trainer: Any,
    actor_schedule: Optional[Union[float, Callable[[float], float]]],
    critic_schedule: Optional[Union[float, Callable[[float], float]]],
) -> None:
    trainer._actor_lr_schedule = actor_schedule
    trainer._critic_lr_schedule = critic_schedule


def configure_phase_lr_context(trainer: Any, total_phase_steps: int) -> None:
    trainer._phase_total_timesteps = max(int(total_phase_steps), 0)
    trainer._phase_start_num_timesteps = trainer.num_timesteps


def update_learning_rate(trainer: Any, optimizer: Any) -> None:
    progress = trainer._get_phase_progress_remaining()
    actor_lr = trainer._eval_schedule(trainer._actor_lr_schedule, progress)
    critic_lr = trainer._eval_schedule(trainer._critic_lr_schedule, progress)
    if actor_lr is not None:
        prev_actor_lr = float(getattr(trainer.policy, "actor_lr_base", getattr(trainer.policy, "actor_lr", 0.0)))
        adjust_aux_warmup_for_actor_lr(trainer, prev_actor_lr, actor_lr)
        if hasattr(trainer.policy, "set_actor_learning_rate"):
            trainer.policy.set_actor_learning_rate(actor_lr)
        else:
            trainer.policy.actor_lr = actor_lr
    if critic_lr is not None:
        trainer.policy.critic_lr = critic_lr
    if hasattr(trainer.policy, "_update_optimizer_lrs"):
        trainer.policy._update_optimizer_lrs()


def build_learning_rate_schedule(
    config: Any,
    base_lr: Optional[float] = None,
) -> Union[float, Callable[[float], float]]:
    schedule_type = getattr(config, "lr_schedule", "constant")
    base_value = config.learning_rate if base_lr is None else base_lr
    if schedule_type != "cosine":
        return base_value
    final_ratio = max(0.0, min(getattr(config, "lr_final_ratio", 0.1), 1.0))
    final_lr = base_value * final_ratio

    def cosine_schedule(progress_remaining: float) -> float:
        progress = min(max(progress_remaining, 0.0), 1.0)
        cos_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return final_lr + (base_value - final_lr) * cos_factor

    return cosine_schedule
