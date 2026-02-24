from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from rl_framework.rl_utils.lr_utils import (
    apply_aux_lr_brake,
    apply_aux_lr_warmup,
    assign_actor_critic_lr_schedules,
    configure_aux_lr_warmup_steps,
    configure_phase_lr_context,
    step_aux_lr_warmup,
    update_learning_rate,
)


class ADPPO(PPO):
    """Custom PPO trainer with grouped actor LR controls and regularization logging."""

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
