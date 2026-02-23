import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict

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
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from rl_framework.common.policy_network import ADFeaturesExtractor, LoRAParametrization


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

        self._cross_attn_lora_targets: Tuple[str, ...] = ("q", "v")
        attn_layer = nn.TransformerDecoderLayer(
            embed_dims, 8, dim_feedforward=embed_dims * 2, dropout=0.1, batch_first=False
        )
        self.attn_module = nn.TransformerDecoder(attn_layer, 3)
        for param in self.attn_module.parameters():
            param.requires_grad = False
        self.lora_modules = nn.ModuleList()
        self._lora_config = dict(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self._lora_injected = False

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
        critic_input = torch.cat([latent_policy.detach(), bev_context], dim=-1)
        latent_value = self.value_adapter(critic_input)
        return latent_policy, latent_value

    def _encode_bev_for_value(self, bev_feat: torch.Tensor) -> torch.Tensor:
        bev_feat = rearrange(bev_feat, "(h w) b c -> b c h w", h=self.bev_h, w=self.bev_w)
        bev_context = self.value_bev_encoder(bev_feat)
        bev_context = bev_context.view(bev_context.shape[0], -1)
        return bev_context

    def ensure_decoder_lora(
        self,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        dropout: Optional[float] = None,
        targets: Optional[Tuple[str, ...]] = None,
    ) -> bool:
        cfg = self._lora_config
        rank = cfg.get("rank", 0) if rank is None else rank
        alpha = cfg.get("alpha", 1.0) if alpha is None else alpha
        dropout = cfg.get("dropout", 0.0) if dropout is None else dropout
        targets = self._cross_attn_lora_targets if targets is None else targets
        if rank <= 0:
            return False
        if self._lora_injected:
            return True
        self.lora_modules = nn.ModuleList()
        for layer_idx, layer in enumerate(self.attn_module.layers):
            self._assert_no_self_attn_parametrizations(layer_idx, layer.self_attn)
            mods = self._register_lora_on_attention(
                layer.multihead_attn,
                rank,
                alpha,
                dropout,
                enabled_targets=targets,
            )
            if mods:
                self.lora_modules.extend(mods)
        self._lora_injected = len(self.lora_modules) > 0
        return self._lora_injected

    def _register_lora_on_attention(
        self,
        attn: Optional[nn.MultiheadAttention],
        rank: int,
        alpha: float,
        dropout: float,
        enabled_targets: Tuple[str, ...] = ("q",),
    ) -> nn.ModuleList:
        modules = nn.ModuleList()
        if attn is None:
            return modules
        embed_dim = attn.embed_dim
        slice_map = {
            "q": (0, embed_dim),
            "k": (embed_dim, 2 * embed_dim),
            "v": (2 * embed_dim, 3 * embed_dim),
        }
        invalid_targets = set(enabled_targets) - set(slice_map.keys())
        if invalid_targets:
            raise ValueError(f"Unknown projection targets for LoRA: {invalid_targets}")
        if "k" in enabled_targets:
            raise ValueError("LoRA on K projection is disabled for stability")
        enabled_slices = tuple(slice_map[name] for name in enabled_targets if name in slice_map)
        if not enabled_slices:
            raise ValueError("At least one projection (q/k/v) must be enabled for LoRA")
        attn.in_proj_weight.requires_grad = False
        attn.out_proj.weight.requires_grad = False
        if hasattr(attn, "in_proj_bias") and attn.in_proj_bias is not None:
            attn.in_proj_bias.requires_grad = False
        if hasattr(attn.out_proj, "bias") and attn.out_proj.bias is not None:
            attn.out_proj.bias.requires_grad = False
        in_lora = LoRAParametrization(
            attn.in_proj_weight.shape[1],
            attn.in_proj_weight.shape[0],
            rank,
            alpha,
            dropout,
            enabled_slices=enabled_slices,
        )
        target_device = attn.in_proj_weight.device
        in_lora = in_lora.to(target_device)
        parametrize.register_parametrization(attn, "in_proj_weight", in_lora)
        mask = in_lora.row_mask.view(-1)
        assert mask[embed_dim : 2 * embed_dim].abs().sum() == 0, "LoRA mask must not touch cross-attn K projection"
        modules.append(in_lora)
        return modules

    def _assert_no_self_attn_parametrizations(self, layer_idx: int, attn: Optional[nn.MultiheadAttention]) -> None:
        if attn is None:
            return
        assert not is_parametrized(attn, "in_proj_weight"), f"Self-attn LoRA found on decoder layer {layer_idx}"
        out_proj = getattr(attn, "out_proj", None)
        if out_proj is not None:
            assert not is_parametrized(out_proj, "weight"), f"Self-attn out_proj LoRA found on layer {layer_idx}"

    def has_lora(self) -> bool:
        return self._lora_injected

    def forward_actor(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent_pi, _ = self.forward(features)
        return latent_pi

    def forward_critic(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, latent_vf = self.forward(features)
        return latent_vf


class FrozenReferenceActor(nn.Module):
    """
    Frozen copy of the actor path (features_extractor -> mlp_extractor -> action_net).
    Keeping this forward frozen prevents attention/adapter updates from drifting the teacher.
    """

    def __init__(
        self,
        features_extractor: nn.Module,
        mlp_extractor: CustomUniADNetwork,
        action_net: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.features_extractor = copy.deepcopy(features_extractor)
        self.mlp_extractor = copy.deepcopy(mlp_extractor)
        self.action_net = copy.deepcopy(action_net)
        self.to(device)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, obs: Any) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


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
        log_std_init: float = -3.0,
        full_std: bool = True,
        use_expln: bool = True,
        kl_reg_coef: float = 0.0,
        bc_reg_coef: float = 0.0,
        unfreeze_adapter_phase2: bool = True,
        weight_decay_coef: float = 0.0,
        env_action_scale: Optional[Sequence[float]] = None,
        env_action_bias: Optional[Sequence[float]] = None,
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
        self.weight_decay_coef = weight_decay_coef
        self.sde_log_std_init = log_std_init
        self.sde_full_std = full_std
        self.sde_use_expln = use_expln
        self.reference_actor: Optional[FrozenReferenceActor] = None
        self.reference_log_std: Optional[torch.Tensor]
        self._last_reg_loss: Optional[torch.Tensor] = None
        self.phase_behavior: Dict[str, Any] = {}

        self._ensure_log_std_bounds()

        kwargs["ortho_init"] = False
        kwargs["features_extractor_class"] = ADFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(embed_dims=embed_dims)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            **kwargs,
        )

        if env_action_scale is None or env_action_bias is None:
            raise ValueError("env_action_scale and env_action_bias must be provided")
        scale_tensor = torch.as_tensor(env_action_scale, dtype=torch.float32)
        bias_tensor = torch.as_tensor(env_action_bias, dtype=torch.float32)
        if scale_tensor.numel() != self.action_dim or bias_tensor.numel() != self.action_dim:
            raise ValueError(
                f"env action affine params must match action_dim={self.action_dim}, "
                f"got scale={scale_tensor.shape}, bias={bias_tensor.shape}"
            )
        self.register_buffer("env_action_scale", scale_tensor)
        self.register_buffer("env_action_bias", bias_tensor)

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

        dist_kwargs = None
        if self.use_sde:
            dist_kwargs = dict(
                full_std=self.sde_full_std,
                use_expln=self.sde_use_expln,
                learn_features=False,
                squash_output=True,
            )
        self.action_dist = make_proba_distribution(
            self.action_space,
            use_sde=self.use_sde,
            dist_kwargs=dist_kwargs,
        )
        if not isinstance(
            self.action_dist,
            (DiagGaussianDistribution, StateDependentNoiseDistribution),
        ):
            raise ValueError("CustomUniADPolicy only supports continuous action spaces")

        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_dim_pi = self.mlp_extractor.latent_dim_pi
            if self.sde_full_std:
                log_std_shape = (latent_dim_pi, self.action_dim)
            else:
                log_std_shape = (latent_dim_pi, 1)
            log_std_init = torch.full(log_std_shape, self.sde_log_std_init)
            self.log_std = nn.Parameter(log_std_init)
            self.action_dist.latent_sde_dim = latent_dim_pi
            self.action_dist.sample_weights(self.log_std)
            self.reference_log_std = None
        else:
            log_std_init = torch.zeros(self.action_dim)
            low_noise = max(1e-3, self.low_noise_sigma)
            high_noise = max(low_noise, self.high_noise_sigma)
            log_std_init[0::2] = math.log(low_noise)
            log_std_init[1::2] = math.log(high_noise)
            self.log_std = nn.Parameter(log_std_init)
            self.reference_log_std = self.log_std.detach().clone()

        self.train_action = False
        self.train_lora = False
        self.train_adapter = False
        self.train_log_std = False
        self._apply_phase_constraints()
        self._setup_optimizer_once()

    def _apply_phase_constraints(self) -> None:
        # Attention backbone stays frozen; LoRA controls adaptability.
        self._set_module_trainability(self.mlp_extractor.attn_module, False)

        action_trainable = self.train_action
        lora_trainable = self.train_lora
        self.log_std.requires_grad = self.train_log_std
        adapter_trainable = self.train_adapter and self.with_adapter and self.unfreeze_adapter_phase2

        self._set_module_trainability(self.action_net, action_trainable)
        self._set_module_trainability(self.mlp_extractor.lora_modules, lora_trainable)
        self._set_module_trainability(self.mlp_extractor.bev_adapter, adapter_trainable)

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

    def _collect_actor_parameters(self, trainable_only: bool = True):
        modules = [self.action_net, self.mlp_extractor.lora_modules, self.mlp_extractor.bev_adapter]
        params = []
        for module in modules:
            for p in module.parameters():
                if trainable_only and not p.requires_grad:
                    continue
                params.append(p)
        if not trainable_only or self.log_std.requires_grad:
            params.append(self.log_std)
        return params

    def _get_optimizer_param_ids(self) -> Set[int]:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return set()
        existing: Set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                existing.add(id(param))
        return existing

    def _maybe_add_params_to_actor_optimizer(self, params: List[nn.Parameter], tag: str = "modules") -> None:
        if not params or not hasattr(self, "optimizer") or self.optimizer is None:
            return
        trainable_params = [p for p in params if p.requires_grad]
        if not trainable_params:
            return
        existing_ids = self._get_optimizer_param_ids()
        new_params = [p for p in trainable_params if id(p) not in existing_ids]
        if not new_params:
            print(f"{tag} params already present in optimizer")
            return
        target_group = None
        if self._actor_group_idx is not None and self._actor_group_idx < len(self.optimizer.param_groups):
            target_group = self.optimizer.param_groups[self._actor_group_idx]
        if target_group is None:
            template_group = self.optimizer.param_groups[0] if self.optimizer.param_groups else None
            if template_group is not None:
                group_config = {k: v for k, v in template_group.items() if k != "params"}
            else:
                group_config = dict(self.optimizer.defaults)
            group_config["params"] = new_params
            group_config["lr"] = self.actor_lr
            self.optimizer.add_param_group(group_config)
            self._actor_group_idx = len(self.optimizer.param_groups) - 1
        else:
            target_group["params"].extend(new_params)
        print(f"Added {len(new_params)} new {tag} params to optimizer")

    def sync_actor_optimizer_with_modules(self, modules: List[nn.Module], tag: str = "modules") -> None:
        if not modules:
            return
        params: List[nn.Parameter] = []
        for module in modules:
            if module is None:
                continue
            params.extend(list(module.parameters()))
        self._maybe_add_params_to_actor_optimizer(params, tag=tag)

    def _collect_critic_parameters(self, trainable_only: bool = True):
        modules = [
            self.mlp_extractor.value_bev_encoder,
            self.mlp_extractor.value_adapter,
            self.value_net,
        ]
        params = []
        for module in modules:
            for p in module.parameters():
                if trainable_only and not p.requires_grad:
                    continue
                params.append(p)
        return params

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
        physical_mean = self.action_net(latent_pi)
        scale = self.env_action_scale.to(physical_mean.device)
        bias = self.env_action_bias.to(physical_mean.device)
        eps = 1e-3
        norm_mean = (physical_mean - bias) / (scale + eps)
        norm_mean = torch.clamp(norm_mean, -1.0 + eps, 1.0 - eps)
        pre_squash_mean = torch.atanh(norm_mean)
        if self.use_sde:
            if not isinstance(self.action_dist, StateDependentNoiseDistribution):
                raise ValueError("SDE enabled but action_dist is not StateDependentNoiseDistribution")
            latent_sde = 0.1 * latent_pi
            return self.action_dist.proba_distribution(pre_squash_mean, self.log_std, latent_sde)
        else:
            log_std = self._clamped_log_std()
            if isinstance(self.action_dist, DiagGaussianDistribution):
                return self.action_dist.proba_distribution(pre_squash_mean, log_std)
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
        phase_behavior: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.phase = phase
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.log_std_bounds = log_std_bounds
        self._ensure_log_std_bounds()
        self.kl_reg_coef = kl_reg_coef
        self.bc_reg_coef = bc_reg_coef
        self.phase_behavior = (phase_behavior or {}).copy()

        behavior = self.phase_behavior
        train_action = behavior.get("train_action", False)
        train_lora = behavior.get("train_lora", False)
        train_adapter = behavior.get("train_adapter", False)
        train_log_std = behavior.get("train_log_std", False)

        injected_lora = False
        if train_lora:
            injected_lora = self.mlp_extractor.ensure_decoder_lora(
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                targets=("q", "v"),
            )
        self.train_action = train_action
        self.train_adapter = train_adapter
        self.train_log_std = train_log_std
        self.train_lora = train_lora and self.mlp_extractor.has_lora()

        self._apply_phase_constraints()
        self._sync_actor_optimizer_params()
        self._update_optimizer_lrs()
        self._log_phase_state(assert_lora=train_lora and injected_lora)

    def _sync_reference_actor(self, force: bool = False) -> None:
        if not force and self.reference_actor is not None:
            return
        self.reference_actor = FrozenReferenceActor(
            self.features_extractor,
            self.mlp_extractor,
            self.action_net,
            device=self.device,
        )
        self.reference_actor.eval()
        if self.use_sde:
            self.reference_log_std = None
        else:
            self.reference_log_std = self.log_std.detach().clone()

    def _infer_device_from_obs(self, obs: Any) -> torch.device:
        if isinstance(obs, torch.Tensor):
            return obs.device
        if isinstance(obs, dict):
            for value in obs.values():
                if isinstance(value, torch.Tensor):
                    return value.device
        if isinstance(obs, (list, tuple)):
            for value in obs:
                if isinstance(value, torch.Tensor):
                    return value.device
        return self.device

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        self._last_reg_loss = self.regularization_loss(obs, latent_pi)
        return values, log_prob, entropy

    def pop_regularization_loss(self) -> torch.Tensor:
        if self._last_reg_loss is None:
            return torch.zeros((), device=self.device)
        value = self._last_reg_loss
        self._last_reg_loss = None
        return value

    def regularization_loss(
        self, obs: Any, latent_pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.bc_reg_coef <= 0 or self.reference_actor is None:
            device = latent_pi.device if latent_pi is not None else self._infer_device_from_obs(obs)
            return torch.zeros((), device=device)
        
        if latent_pi is None:
            features = self.extract_features(obs)
            latent_pi, _ = self.mlp_extractor(features)
        
        device = latent_pi.device
        self.reference_actor.eval()
        curr_mean_phys = self.action_net(latent_pi)
        with torch.no_grad():
            ref_mean_phys = self.reference_actor(obs)

        scale = self.env_action_scale.to(device)
        bias  = self.env_action_bias.to(device)
        eps = 1e-6

        curr_mean_norm = (curr_mean_phys - bias) / (scale + eps)
        ref_mean_norm  = (ref_mean_phys  - bias) / (scale + eps)

        curr_mean_norm = curr_mean_norm.clamp(-1.0, 1.0)
        ref_mean_norm  = ref_mean_norm.clamp(-1.0, 1.0)

        loss = self.bc_reg_coef * F.mse_loss(curr_mean_norm, ref_mean_norm)
        return loss

    def _setup_optimizer_once(self) -> None:
        actor_params = self._collect_actor_parameters(trainable_only=True)
        critic_params = self._collect_critic_parameters(trainable_only=True)
        optim_groups = []
        self._actor_group_idx = None
        self._critic_group_idx = None
        if actor_params:
            self._actor_group_idx = len(optim_groups)
            optim_groups.append(dict(params=actor_params, lr=self.actor_lr))
        if critic_params:
            self._critic_group_idx = len(optim_groups)
            optim_groups.append(dict(params=critic_params, lr=self.critic_lr))
        if not optim_groups:
            self._critic_group_idx = 0
            optim_groups.append(dict(params=self.parameters(), lr=self.critic_lr))
        self.optimizer = self.optimizer_class(optim_groups, **self.optimizer_kwargs)
        self._actor_param_list = actor_params
        self._critic_param_list = critic_params

    def _update_optimizer_lrs(self) -> None:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return
        if self._actor_group_idx is not None:
            self.optimizer.param_groups[self._actor_group_idx]["lr"] = self.actor_lr
        if self._critic_group_idx is not None:
            self.optimizer.param_groups[self._critic_group_idx]["lr"] = self.critic_lr

    def _sync_actor_optimizer_params(self) -> None:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return
        modules: List[nn.Module] = []
        if hasattr(self, "action_net"):
            modules.append(self.action_net)
        lora_modules = getattr(self.mlp_extractor, "lora_modules", None)
        if lora_modules is not None:
            modules.append(lora_modules)
        if self.with_adapter and hasattr(self.mlp_extractor, "bev_adapter"):
            modules.append(self.mlp_extractor.bev_adapter)
        modules = [m for m in modules if m is not None]
        if modules:
            self.sync_actor_optimizer_with_modules(modules, tag=f"phase{self.phase}")
        if getattr(self, "log_std", None) is not None and self.log_std.requires_grad:
            self._maybe_add_params_to_actor_optimizer([self.log_std], tag="log_std")

    def _log_phase_state(
        self,
        assert_lora: bool = False,
        prefix_depth: int = 2,
        max_prefixes: int = 12,
        max_examples_per_prefix: int = 2,
    ) -> None:

        name_map: Dict[int, str] = {}
        for n, p in self.named_parameters():
            name_map[id(p)] = n

        def _prefix_of(name: str) -> str:
            parts = name.split(".")
            d = max(1, int(prefix_depth))
            return ".".join(parts[: min(d, len(parts))])

        def _summarize_param_ids(param_ids: Set[int]) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
            counts = defaultdict(int)
            examples = defaultdict(list)
            for pid in param_ids:
                n = name_map.get(pid, "<unmapped_param>")
                pref = _prefix_of(n)
                counts[pref] += 1
                if len(examples[pref]) < max_examples_per_prefix:
                    examples[pref].append(n)

            ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            return ordered, examples

        actor_params = self._collect_actor_parameters(trainable_only=True)
        critic_params = self._collect_critic_parameters(trainable_only=True)

        actor_ids = {id(p) for p in actor_params}
        critic_ids = {id(p) for p in critic_params}

        print(f"[Phase {getattr(self, 'phase', '?')}] trainable actor_tensors={len(actor_params)} | critic_tensors={len(critic_params)}")

        actor_ordered, actor_examples = _summarize_param_ids(actor_ids)
        critic_ordered, critic_examples = _summarize_param_ids(critic_ids)

        def _print_breakdown(title: str, ordered: List[Tuple[str, int]], examples: Dict[str, List[str]]) -> None:
            if not ordered:
                print(f"  {title}: <none>")
                return
            print(f"  {title}:")
            for pref, cnt in ordered[:max_prefixes]:
                ex = examples.get(pref, [])
                ex_str = ""
                if ex:
                    ex_str = " | e.g. " + "; ".join(ex)
                print(f"    - {pref}: {cnt}{ex_str}")
            if len(ordered) > max_prefixes:
                print(f"    ... ({len(ordered) - max_prefixes} more prefixes)")

        _print_breakdown("actor trainable prefixes", actor_ordered, actor_examples)
        _print_breakdown("critic trainable prefixes", critic_ordered, critic_examples)

        if not hasattr(self, "optimizer") or self.optimizer is None:
            return

        for idx, group in enumerate(self.optimizer.param_groups):
            lr = group.get("lr", 0.0)

            group_trainable_ids = {id(p) for p in group["params"] if getattr(p, "requires_grad", False)}
            total_ids = {id(p) for p in group["params"]}

            tag = ""
            if getattr(self, "_actor_group_idx", None) == idx:
                tag = "actor"
            elif getattr(self, "_critic_group_idx", None) == idx:
                tag = "critic"
            tag_str = f" {tag}" if tag else ""

            print(f"  - opt_group[{idx}{tag_str}] lr={lr:.2e} trainable={len(group_trainable_ids)}/{len(total_ids)}")

            ordered, examples = _summarize_param_ids(group_trainable_ids)
            if ordered:
            # Keep this shorter than the actor/critic full breakdown
                for pref, cnt in ordered[: min(6, max_prefixes)]:
                    ex = examples.get(pref, [])
                    ex_str = ""
                    if ex:
                        ex_str = " | e.g. " + "; ".join(ex)
                    print(f"      * {pref}: {cnt}{ex_str}")

        if assert_lora:
            self._assert_lora_in_optimizer()


    def _assert_lora_in_optimizer(self) -> None:
        lora_modules = getattr(self.mlp_extractor, "lora_modules", None)
        if lora_modules is None or self._actor_group_idx is None:
            return
        lora_params = [p for p in lora_modules.parameters() if p.requires_grad]
        if not lora_params:
            return
        if self.optimizer is None:
            raise AssertionError("LoRA params require an initialized optimizer")
        actor_group = self.optimizer.param_groups[self._actor_group_idx]
        actor_ids = {id(p) for p in actor_group["params"]}
        missing = {id(p) for p in lora_params} - actor_ids
        if missing:
            raise AssertionError(f"LoRA parameters missing from actor optimizer group: {len(missing)}")

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.mlp_extractor.to(self.device)
        self.action_net.to(self.device)

        if "attn_module" in checkpoint:
            self.mlp_extractor.attn_module.load_state_dict(
                checkpoint["attn_module"], strict=strict
            )
            print("✓ attn_module loaded")

        if self.with_adapter and "bev_adapter" in checkpoint:
            self.mlp_extractor.bev_adapter.load_state_dict(
                checkpoint["bev_adapter"], strict=strict
            )
            print("✓ bev_adapter loaded")

        if "reg_branch" in checkpoint:
            self.action_net.load_state_dict(checkpoint["reg_branch"], strict=strict)
            print("✓ reg_branch loaded into action_net")
        self._sync_reference_actor(force=True)
