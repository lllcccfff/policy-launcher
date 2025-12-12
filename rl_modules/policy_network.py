import copy
import math
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import is_parametrized
from einops import rearrange
from gymnasium import spaces

from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class UniADFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embed_dims: int = 256):
        super().__init__(observation_space, features_dim=embed_dims)
        self.embed_dims = embed_dims

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return observations


class LoRAParametrization(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.down = nn.Parameter(torch.empty(rank, fan_in))
        self.up = nn.Parameter(torch.empty(fan_out, rank))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.zeros_(self.up)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(self.up, self.dropout(self.down))
        return base_weight + self.scaling * delta


class CustomUniADNetwork(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        with_adapter: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_adapter = with_adapter
        self.lora_rank = lora_rank

        attn_layer = nn.TransformerDecoderLayer(
            embed_dims, 8, dim_feedforward=embed_dims * 2, dropout=0.1, batch_first=False
        )
        self.attn_module = nn.TransformerDecoder(attn_layer, 3)
        for param in self.attn_module.parameters():
            param.requires_grad = False
        self.lora_modules = nn.ModuleList()
        self._lora_config = dict(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self._lora_injected = False
        self._inject_lora_into_decoder(lora_rank, lora_alpha, lora_dropout)

        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            blocks = [copy.deepcopy(bev_adapter_block) for _ in range(3)]
            self.bev_adapter = nn.Sequential(*blocks)

        value_bev_channels = embed_dims // 2
        self.value_bev_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, value_bev_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, value_bev_channels),
            nn.ReLU(),
            nn.Conv2d(value_bev_channels, value_bev_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, value_bev_channels),
            nn.ReLU(),
            nn.Conv2d(value_bev_channels, embed_dims, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # gate critic access to actor latent; sigmoid(gate)≈0 keeps backbone frozen initially
        self.value_plan_gate = nn.Parameter(torch.tensor(-2.0))

        critic_input_dim = embed_dims * 2
        self.value_adapter = nn.Sequential(
            nn.LayerNorm(critic_input_dim),
            nn.Linear(critic_input_dim, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )

        self.latent_dim_pi = embed_dims
        self.latent_dim_vf = embed_dims

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        plan_query = features["plan_query"]
        bev_feat = features["bev_feat"]

        plan_query = plan_query.permute(1, 0, 2)  # [B, P, C] -> [P, B, C]
        bev_feat = bev_feat.permute(1, 0, 2)  # [B, HW, C] -> [HW, B, C]

        if self.with_adapter:
            bev_feat = rearrange(bev_feat, "(h w) b c -> b c h w", h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)
            bev_feat = rearrange(bev_feat, "b c h w -> (h w) b c")

        def attn_forward(pq: torch.Tensor, bf: torch.Tensor) -> torch.Tensor:
            return self.attn_module(pq, bf)

        if torch.is_grad_enabled():
            plan_query_out = checkpoint(attn_forward, plan_query, bev_feat, use_reentrant=False)
        else:
            plan_query_out = attn_forward(plan_query, bev_feat)
        if plan_query_out.shape[0] != 1:
            plan_query_out = plan_query_out[:1]
        latent_policy = plan_query_out.squeeze(0)

        bev_context = self._encode_bev_for_value(bev_feat.detach())
        gate = torch.sigmoid(self.value_plan_gate)
        latent_policy_mix = latent_policy.detach() + gate * (latent_policy - latent_policy.detach())
        critic_input = torch.cat([latent_policy_mix, bev_context], dim=-1)
        latent_value = self.value_adapter(critic_input)
        return latent_policy, latent_value

    def _encode_bev_for_value(self, bev_feat: torch.Tensor) -> torch.Tensor:
        bev_feat = rearrange(bev_feat, "(h w) b c -> b c h w", h=self.bev_h, w=self.bev_w)
        bev_context = self.value_bev_encoder(bev_feat)
        bev_context = bev_context.view(bev_context.shape[0], -1)
        return bev_context

    def _inject_lora_into_decoder(self, rank: int, alpha: float, dropout: float) -> None:
        if rank <= 0:
            self._lora_injected = False
            return
        self._clear_lora_parametrizations()
        self.lora_modules = nn.ModuleList()
        for layer in self.attn_module.layers:
            self.lora_modules.extend(self._register_lora_on_attention(layer.self_attn, rank, alpha, dropout))
            self.lora_modules.extend(self._register_lora_on_attention(layer.multihead_attn, rank, alpha, dropout))
        self._lora_injected = True

    def _register_lora_on_attention(
        self, attn: Optional[nn.MultiheadAttention], rank: int, alpha: float, dropout: float
    ) -> nn.ModuleList:
        modules = nn.ModuleList()
        if attn is None:
            return modules
        attn.in_proj_weight.requires_grad = False
        attn.out_proj.weight.requires_grad = False
        in_lora = LoRAParametrization(attn.in_proj_weight.shape[1], attn.in_proj_weight.shape[0], rank, alpha, dropout)
        out_lora = LoRAParametrization(attn.out_proj.weight.shape[1], attn.out_proj.weight.shape[0], rank, alpha, dropout)
        target_device = attn.in_proj_weight.device
        in_lora = in_lora.to(target_device)
        out_lora = out_lora.to(attn.out_proj.weight.device)
        parametrize.register_parametrization(attn, "in_proj_weight", in_lora)
        parametrize.register_parametrization(attn.out_proj, "weight", out_lora)
        modules.append(in_lora)
        modules.append(out_lora)
        return modules

    def _clear_lora_parametrizations(self) -> bool:
        removed = False
        for layer in self.attn_module.layers:
            removed = self._remove_lora_from_attention(layer.self_attn) or removed
            removed = self._remove_lora_from_attention(layer.multihead_attn) or removed
        if removed:
            self.lora_modules = nn.ModuleList()
        self._lora_injected = False
        return removed

    def _remove_lora_from_attention(self, attn: Optional[nn.Module]) -> bool:
        if attn is None:
            return False
        flag = False
        if is_parametrized(attn, "in_proj_weight"):
            parametrize.remove_parametrizations(attn, "in_proj_weight", leave_parametrized=False)
            flag = True
        out_proj = getattr(attn, "out_proj", None)
        if out_proj is not None and is_parametrized(out_proj, "weight"):
            parametrize.remove_parametrizations(out_proj, "weight", leave_parametrized=False)
            flag = True
        return flag

    def reinject_lora(self, reference_module: Optional[nn.Module] = None) -> None:
        cfg = getattr(self, "_lora_config", None)
        if cfg is None:
            return
        self._inject_lora_into_decoder(cfg["rank"], cfg["alpha"], cfg["dropout"])
        if reference_module is not None and hasattr(reference_module, "out_proj"):
            self.to(reference_module.out_proj.weight.device)

    def forward_actor(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent_pi, _ = self.forward(features)
        return latent_pi

    def forward_critic(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, latent_vf = self.forward(features)
        return latent_vf


class CustomUniADPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        planning_steps: int = 6,
        embed_dims: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        with_adapter: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        phase: int = 0,
        actor_lr: float = 3e-6,
        critic_lr: float = 1e-5,
        log_std_bounds: Tuple[float, float] = (-5.0, -2.0),
        kl_reg_coef: float = 0.0,
        bc_reg_coef: float = 0.0,
        unfreeze_adapter_phase2: bool = True,
        *args,
        **kwargs,
    ):
        self.embed_dims = embed_dims
        self.planning_steps = planning_steps
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_adapter = with_adapter
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.action_dim = planning_steps * 2
        self.low_noise_sigma = 0.05
        self.high_noise_sigma = 0.5

        self.phase = phase
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.log_std_bounds = log_std_bounds
        self.kl_reg_coef = kl_reg_coef
        self.bc_reg_coef = bc_reg_coef
        self.unfreeze_adapter_phase2 = unfreeze_adapter_phase2
        self.reference_action_net: Optional[nn.Module] = None
        self.reference_log_std: Optional[torch.Tensor]
        self._last_reg_loss: Optional[torch.Tensor] = None
        self._reference_synced_for_high_phase = False

        self._ensure_log_std_bounds()

        kwargs["ortho_init"] = False
        kwargs["features_extractor_class"] = UniADFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(embed_dims=embed_dims)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomUniADNetwork(
            embed_dims=self.embed_dims,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            with_adapter=self.with_adapter,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        self._build_mlp_extractor()

        self.action_dist = make_proba_distribution(self.action_space, use_sde=False)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("CustomUniADPolicy only supports continuous action spaces")

        self.action_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.action_dim),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 1),
        )

        log_std_init = torch.zeros(self.action_dim)
        low_noise = max(1e-3, self.low_noise_sigma)
        high_noise = max(low_noise, self.high_noise_sigma)
        log_std_init[0::2] = math.log(low_noise)
        log_std_init[1::2] = math.log(high_noise)
        self.log_std = nn.Parameter(log_std_init)

        self.reference_action_net = copy.deepcopy(self.action_net)
        for param in self.reference_action_net.parameters():
            param.requires_grad = False
        if "reference_log_std" not in dict(self.named_buffers()):
            self.register_buffer(
                "reference_log_std",
                self.log_std.detach().clone()
            )
        else:
            self.reference_log_std.copy_(self.log_std.detach())

        self._apply_phase_constraints()
        self._rebuild_optimizer()

    def _apply_phase_constraints(self) -> None:
        # Attention backbone stays frozen; LoRA controls adaptability.
        self._set_module_trainability(self.mlp_extractor.attn_module, False)

        action_trainable = self.phase >= 1
        lora_trainable = self.phase >= 1
        self.log_std.requires_grad = self.phase >= 2
        adapter_trainable = self.phase >= 2 and self.unfreeze_adapter_phase2

        self._set_module_trainability(self.action_net, action_trainable)
        self._set_module_trainability(self.mlp_extractor.lora_modules, lora_trainable)

        if self.with_adapter and hasattr(self.mlp_extractor, "bev_adapter"):
            self._set_module_trainability(self.mlp_extractor.bev_adapter, adapter_trainable)

        if self.phase >= 2:
            if not self._reference_synced_for_high_phase:
                self._refresh_reference_policy()
                self._reference_synced_for_high_phase = True
        else:
            self._reference_synced_for_high_phase = False
        self._enforce_eval_mode_for_frozen()

    def _set_module_trainability(self, module: nn.Module, trainable: bool) -> None:
        if module is None:
            return
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            for sub_module in module:
                self._set_module_trainability(sub_module, trainable)
            return
        for param in module.parameters():
            param.requires_grad = trainable
        if not trainable:
            module.eval()

    def _enforce_eval_mode_for_frozen(self) -> None:
        if not any(p.requires_grad for p in self.mlp_extractor.attn_module.parameters()):
            self.mlp_extractor.attn_module.eval()
        lora_params = list(self.mlp_extractor.lora_modules.parameters())
        if lora_params and not any(p.requires_grad for p in lora_params):
            for lora_module in self.mlp_extractor.lora_modules:
                lora_module.eval()
        if self.with_adapter and hasattr(self.mlp_extractor, "bev_adapter"):
            adapter_params = list(self.mlp_extractor.bev_adapter.parameters())
            if adapter_params and not any(p.requires_grad for p in adapter_params):
                self.mlp_extractor.bev_adapter.eval()
        if not any(p.requires_grad for p in self.action_net.parameters()):
            self.action_net.eval()

    def _collect_actor_parameters(self):
        modules = [self.action_net, self.mlp_extractor.lora_modules]
        if self.phase >= 2 and self.with_adapter and hasattr(self.mlp_extractor, "bev_adapter"):
            modules.append(self.mlp_extractor.bev_adapter)
        params = []
        for module in modules:
            params.extend([p for p in module.parameters() if p.requires_grad])
        if self.log_std.requires_grad:
            params.append(self.log_std)
        return params

    def _collect_critic_parameters(self):
        modules = [
            self.mlp_extractor.value_bev_encoder,
            self.mlp_extractor.value_adapter,
            self.value_net,
        ]
        params = []
        for module in modules:
            params.extend([p for p in module.parameters() if p.requires_grad])
        if hasattr(self.mlp_extractor, "value_plan_gate"):
            params.append(self.mlp_extractor.value_plan_gate)
        return params

    def set_phase(self, phase: int) -> None:
        self.phase = phase
        self._apply_phase_constraints()
        self._rebuild_optimizer()

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def extract_features(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.features_extractor(obs)

    def _ensure_log_std_bounds(self) -> None:
        if self.log_std_bounds is None:
            return
        desired_low = math.log(max(1e-6, self.low_noise_sigma))
        desired_high = math.log(max(self.low_noise_sigma, self.high_noise_sigma))
        lo, hi = self.log_std_bounds
        if lo > desired_low:
            lo = desired_low
        if hi < desired_high:
            hi = desired_high
        self.log_std_bounds = (lo, hi)

    def _clamped_log_std(self) -> torch.Tensor:
        if self.log_std_bounds is None:
            return self.log_std
        return torch.clamp(self.log_std, self.log_std_bounds[0], self.log_std_bounds[1])

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        mean_actions = self.action_net(latent_pi)
        log_std = self._clamped_log_std()
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, log_std)
        raise ValueError("Unsupported action distribution")

    def train(self, mode: bool) -> "CustomUniADPolicy":
        super().train(mode)
        self._enforce_eval_mode_for_frozen()
        return self

    def configure_phase(
        self,
        phase: int,
        actor_lr: float,
        critic_lr: float,
        log_std_bounds: Tuple[float, float],
        kl_reg_coef: float,
        bc_reg_coef: float,
    ) -> None:
        self.phase = phase
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.log_std_bounds = log_std_bounds
        self._ensure_log_std_bounds()
        self.kl_reg_coef = kl_reg_coef
        self.bc_reg_coef = bc_reg_coef
        self._apply_phase_constraints()
        self._rebuild_optimizer()

    def _refresh_reference_policy(self) -> None:
        if self.reference_action_net is None:
            self.reference_action_net = copy.deepcopy(self.action_net)
        else:
            self.reference_action_net.load_state_dict(self.action_net.state_dict())
        for param in self.reference_action_net.parameters():
            param.requires_grad = False
        if hasattr(self, "reference_log_std"):
            self.reference_log_std.copy_(self.log_std.detach())
        else:
            self.register_buffer("reference_log_std", self.log_std.detach().clone())

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        self._last_reg_loss = self.regularization_loss(latent_pi)
        return values, log_prob, entropy

    def pop_regularization_loss(self) -> torch.Tensor:
        if self._last_reg_loss is None:
            return torch.zeros((), device=self.device)
        value = self._last_reg_loss
        self._last_reg_loss = None
        return value

    def regularization_loss(self, latent_pi: torch.Tensor) -> torch.Tensor:
        if (self.kl_reg_coef <= 0 and self.bc_reg_coef <= 0) or self.reference_action_net is None:
            return torch.zeros((), device=latent_pi.device)
        curr_mean = self.action_net(latent_pi)
        with torch.no_grad():
            ref_mean = self.reference_action_net(latent_pi.detach())
            ref_log_std = self.reference_log_std.to(latent_pi.device)
        loss = torch.zeros((), device=latent_pi.device)
        if self.bc_reg_coef > 0:
            loss = loss + self.bc_reg_coef * F.mse_loss(curr_mean, ref_mean)
        if self.kl_reg_coef > 0:
            curr_log_std = self._clamped_log_std()
            curr_var = torch.exp(2 * curr_log_std)
            ref_var = torch.exp(2 * ref_log_std)
            kl_per_dim = (
                torch.log(torch.sqrt(ref_var) / torch.sqrt(curr_var))
                + (curr_var + (curr_mean - ref_mean) ** 2) / (2 * ref_var)
                - 0.5
            )
            kl = kl_per_dim.sum(dim=-1).mean()
            loss = loss + self.kl_reg_coef * kl
        return loss

    def _rebuild_optimizer(self) -> None:
        actor_params = self._collect_actor_parameters()
        critic_params = self._collect_critic_parameters()
        optim_groups = []
        if actor_params:
            optim_groups.append(dict(params=actor_params, lr=self.actor_lr))
        if critic_params:
            optim_groups.append(dict(params=critic_params, lr=self.critic_lr))
        if not optim_groups:
            optim_groups.append(dict(params=self.parameters(), lr=self.critic_lr))
        self.optimizer = self.optimizer_class(optim_groups, **self.optimizer_kwargs)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.mlp_extractor.to(self.device)
        self.action_net.to(self.device)

        if "attn_module" in checkpoint:
            lora_disabled = False
            if hasattr(self.mlp_extractor, "_clear_lora_parametrizations"):
                lora_disabled = self.mlp_extractor._clear_lora_parametrizations()
            self.mlp_extractor.attn_module.load_state_dict(
                checkpoint["attn_module"], strict=strict
            )
            if lora_disabled and getattr(self.mlp_extractor, "lora_rank", 0) > 0:
                self.mlp_extractor.reinject_lora(self.mlp_extractor.attn_module.layers[0].self_attn)
            print("✓ attn_module loaded")

        if self.with_adapter and "bev_adapter" in checkpoint:
            self.mlp_extractor.bev_adapter.load_state_dict(
                checkpoint["bev_adapter"], strict=strict
            )
            print("✓ bev_adapter loaded")

        if "reg_branch" in checkpoint:
            self.action_net.load_state_dict(checkpoint["reg_branch"], strict=strict)
            print("✓ reg_branch loaded into action_net")
        self.reference_action_net = copy.deepcopy(self.action_net)
        for param in self.reference_action_net.parameters():
            param.requires_grad = False
        self.reference_log_std = self.log_std.detach().clone()
