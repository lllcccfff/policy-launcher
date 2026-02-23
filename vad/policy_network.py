import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize
from gymnasium import spaces

from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy

from rl_framework.common.policy_network import ADFeaturesExtractor, LoRAParametrization


class CustomVADPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        planning_steps: int = 6,
        embed_dims: int = 256,
        ego_fut_mode: int = 3,
        planning_weights: Optional[Dict[str, Any]] = None,
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
        weight_decay_coef: float = 0.0,
        env_action_scale: Optional[Sequence[float]] = None,
        env_action_bias: Optional[Sequence[float]] = None,
        *args,
        **kwargs,
    ):
        self.embed_dims = int(embed_dims)
        self.planning_steps = int(planning_steps)
        self.ego_fut_mode = int(ego_fut_mode)
        self.action_dim = self.planning_steps * 2
        self.low_noise_sigma = 0.05
        self.high_noise_sigma = 0.5
        self.phase = int(phase)

        if planning_weights is None:
            raise ValueError("planning_weights must be provided for CustomVADPolicy")
        self.planning_weights = planning_weights

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.log_std_bounds = log_std_bounds
        self.sde_log_std_init = log_std_init
        self.sde_full_std = bool(full_std)
        self.sde_use_expln = bool(use_expln)
        self.weight_decay_coef = float(weight_decay_coef)

        self.kl_reg_coef = float(kl_reg_coef)
        self.bc_reg_coef = float(bc_reg_coef)
        self.phase_behavior: Dict[str, Any] = {}
        self._last_reg_loss: Optional[torch.Tensor] = None

        self._lora_injected = False
        self.lora_modules = nn.ModuleList()
        self.train_action = False
        self.train_lora = False
        self.train_log_std = False

        self._ensure_log_std_bounds()

        kwargs["ortho_init"] = False
        kwargs["features_extractor_class"] = ADFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(embed_dims=self.embed_dims)

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
        self.mlp_extractor = nn.Identity()
        self.mlp_extractor.latent_dim_pi = self.action_dim
        self.mlp_extractor.latent_dim_vf = self.embed_dims

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        self._build_mlp_extractor()
        self._build_planning_modules()
        self.action_net = nn.Identity()
        self.value_adapter = nn.Sequential(
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
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
            raise ValueError("CustomVADPolicy only supports continuous action spaces")

        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_dim_pi = self.action_dim
            if self.sde_full_std:
                log_std_shape = (latent_dim_pi, self.action_dim)
            else:
                log_std_shape = (latent_dim_pi, 1)
            log_std_init = torch.full(log_std_shape, self.sde_log_std_init)
            self.log_std = nn.Parameter(log_std_init)
            self.action_dist.latent_sde_dim = latent_dim_pi
            self.action_dist.sample_weights(self.log_std)
        else:
            log_std_init = torch.zeros(self.action_dim)
            low_noise = max(1e-3, self.low_noise_sigma)
            high_noise = max(low_noise, self.high_noise_sigma)
            log_std_init[0::2] = math.log(low_noise)
            log_std_init[1::2] = math.log(high_noise)
            self.log_std = nn.Parameter(log_std_init)

        self._apply_phase_constraints()
        self._setup_optimizer_once()

    def _build_planning_modules(self) -> None:
        required = [
            "ego_agent_decoder_template",
            "ego_map_decoder_template",
            "ego_fut_decoder_template",
            "ego_agent_decoder_state_dict",
            "ego_map_decoder_state_dict",
            "ego_fut_decoder_state_dict",
        ]
        for key in required:
            if key not in self.planning_weights:
                raise ValueError(f"planning_weights missing key: {key}")

        self.ego_agent_decoder = copy.deepcopy(
            self.planning_weights["ego_agent_decoder_template"]
        ).to(self.device)
        self.ego_map_decoder = copy.deepcopy(
            self.planning_weights["ego_map_decoder_template"]
        ).to(self.device)
        self.ego_fut_decoder = copy.deepcopy(
            self.planning_weights["ego_fut_decoder_template"]
        ).to(self.device)

        self.ego_agent_decoder.load_state_dict(
            self.planning_weights["ego_agent_decoder_state_dict"], strict=True
        )
        self.ego_map_decoder.load_state_dict(
            self.planning_weights["ego_map_decoder_state_dict"], strict=True
        )
        self.ego_fut_decoder.load_state_dict(
            self.planning_weights["ego_fut_decoder_state_dict"], strict=True
        )

        planning_embed_dims = int(self.planning_weights.get("embed_dims", self.embed_dims))
        if planning_embed_dims != self.embed_dims:
            raise ValueError(
                f"embed_dims mismatch: config={self.embed_dims}, planning={planning_embed_dims}"
            )
        planning_fut_mode = int(self.planning_weights.get("ego_fut_mode", self.ego_fut_mode))
        if planning_fut_mode != self.ego_fut_mode:
            raise ValueError(
                f"ego_fut_mode mismatch: config={self.ego_fut_mode}, planning={planning_fut_mode}"
            )
        planning_ts = int(self.planning_weights.get("fut_ts", self.planning_steps))
        if planning_ts != self.planning_steps:
            raise ValueError(
                f"planning_steps mismatch: config={self.planning_steps}, planning={planning_ts}"
            )

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

    @staticmethod
    def _to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        return mask > 0.5

    def _validate_command(self, ego_fut_cmd: torch.Tensor, batch_size: int) -> torch.Tensor:
        if ego_fut_cmd.ndim != 2:
            raise ValueError(f"ego_fut_cmd must be 2D [B, M], got {ego_fut_cmd.shape}")
        if ego_fut_cmd.shape[0] != batch_size:
            raise ValueError(
                f"ego_fut_cmd batch mismatch: expected {batch_size}, got {ego_fut_cmd.shape[0]}"
            )
        if ego_fut_cmd.shape[1] != self.ego_fut_mode:
            raise ValueError(
                f"ego_fut_cmd mode mismatch: expected {self.ego_fut_mode}, got {ego_fut_cmd.shape[1]}"
            )
        if not torch.isfinite(ego_fut_cmd).all():
            raise ValueError("ego_fut_cmd contains non-finite values")
        rounded = torch.round(ego_fut_cmd)
        if not torch.allclose(ego_fut_cmd, rounded, atol=1e-4):
            raise ValueError("ego_fut_cmd must be one-hot")
        if not torch.all((rounded == 0) | (rounded == 1)):
            raise ValueError("ego_fut_cmd must contain only 0/1")
        if not torch.all(rounded.sum(dim=-1) == 1):
            raise ValueError("ego_fut_cmd must contain exactly one active mode")
        return rounded.to(dtype=torch.float32)

    def _extract_observation_tensors(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        required = [
            "ego_query",
            "ego_pos_emb",
            "agent_query",
            "agent_pos_emb",
            "agent_mask",
            "map_query",
            "map_pos_emb",
            "map_mask",
            "ego_fut_cmd",
        ]
        missing = [name for name in required if name not in features]
        if missing:
            raise ValueError(f"Missing observation keys for CustomVADPolicy: {missing}")

        tensors = {k: features[k] for k in required}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Observation `{name}` must be a torch.Tensor")

        ego_query = tensors["ego_query"]
        ego_pos_emb = tensors["ego_pos_emb"]
        if ego_query.ndim != 3 or ego_pos_emb.ndim != 3:
            raise ValueError("ego_query and ego_pos_emb must be [B, 1, D]")
        if ego_query.shape[1] != 1 or ego_pos_emb.shape[1] != 1:
            raise ValueError("ego_query and ego_pos_emb second dim must be 1")
        if ego_query.shape[-1] != self.embed_dims or ego_pos_emb.shape[-1] != self.embed_dims:
            raise ValueError("ego_query/ego_pos_emb embed dim mismatch")

        agent_query = tensors["agent_query"]
        agent_pos_emb = tensors["agent_pos_emb"]
        map_query = tensors["map_query"]
        map_pos_emb = tensors["map_pos_emb"]
        agent_mask = self._to_bool_mask(tensors["agent_mask"])
        map_mask = self._to_bool_mask(tensors["map_mask"])
        if (
            agent_query.ndim != 3
            or agent_pos_emb.ndim != 3
            or map_query.ndim != 3
            or map_pos_emb.ndim != 3
        ):
            raise ValueError("agent/map query and pos tensors must be [B, N, D]")
        if agent_mask.ndim != 2 or map_mask.ndim != 2:
            raise ValueError("agent_mask/map_mask must be [B, N]")
        if agent_query.shape != agent_pos_emb.shape or map_query.shape != map_pos_emb.shape:
            raise ValueError("query and pos embedding shapes must match")
        if agent_query.shape[:2] != agent_mask.shape or map_query.shape[:2] != map_mask.shape:
            raise ValueError("query and mask token dimensions mismatch")
        if agent_query.shape[-1] != self.embed_dims or map_query.shape[-1] != self.embed_dims:
            raise ValueError("agent/map embed dim mismatch")

        batch_size = ego_query.shape[0]
        for name in ("agent_query", "agent_pos_emb", "agent_mask", "map_query", "map_pos_emb", "map_mask"):
            if tensors[name].shape[0] != batch_size:
                raise ValueError(f"Batch dimension mismatch on `{name}`")
        ego_fut_cmd = self._validate_command(tensors["ego_fut_cmd"], batch_size)

        return {
            "ego_query": ego_query,
            "ego_pos_emb": ego_pos_emb,
            "agent_query": agent_query,
            "agent_pos_emb": agent_pos_emb,
            "agent_mask": agent_mask,
            "map_query": map_query,
            "map_pos_emb": map_pos_emb,
            "map_mask": map_mask,
            "ego_fut_cmd": ego_fut_cmd,
        }

    def _decode_action_mean_and_value(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self._extract_observation_tensors(features)

        ego_agent_query = self.ego_agent_decoder(
            query=obs["ego_query"].permute(1, 0, 2),
            key=obs["agent_query"].permute(1, 0, 2),
            value=obs["agent_query"].permute(1, 0, 2),
            query_pos=obs["ego_pos_emb"].permute(1, 0, 2),
            key_pos=obs["agent_pos_emb"].permute(1, 0, 2),
            key_padding_mask=obs["agent_mask"],
        )
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=obs["map_query"].permute(1, 0, 2),
            value=obs["map_query"].permute(1, 0, 2),
            query_pos=obs["ego_pos_emb"].permute(1, 0, 2),
            key_pos=obs["map_pos_emb"].permute(1, 0, 2),
            key_padding_mask=obs["map_mask"],
        )

        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2), ego_map_query.permute(1, 0, 2)], dim=-1
        )
        ego_fut_preds = self.ego_fut_decoder(ego_feats)
        batch_size = ego_fut_preds.shape[0]
        ego_fut_preds = ego_fut_preds.reshape(batch_size, self.ego_fut_mode, self.planning_steps, 2)
        cmd = obs["ego_fut_cmd"].to(dtype=ego_fut_preds.dtype)
        selected = torch.sum(ego_fut_preds * cmd[..., None, None], dim=1)
        action_mean = selected.reshape(batch_size, self.action_dim)

        ego_map_out = ego_map_query.permute(1, 0, 2)
        value_in = ego_map_out.squeeze(1).detach()
        latent_vf = self.value_adapter(value_in)
        values = self.value_net(latent_vf)
        return action_mean, values

    def extract_features(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.features_extractor(obs)

    def _physical_mean_to_pre_squash(self, physical_mean: torch.Tensor) -> torch.Tensor:
        scale = self.env_action_scale.to(physical_mean.device)
        bias = self.env_action_bias.to(physical_mean.device)
        eps = 1e-3
        norm_mean = (physical_mean - bias) / (scale + eps)
        norm_mean = torch.clamp(norm_mean, -1.0 + eps, 1.0 - eps)
        return torch.atanh(norm_mean)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        pre_squash_mean = self._physical_mean_to_pre_squash(latent_pi)
        if self.use_sde:
            if not isinstance(self.action_dist, StateDependentNoiseDistribution):
                raise ValueError("SDE enabled but action_dist is not StateDependentNoiseDistribution")
            latent_sde = 0.1 * pre_squash_mean
            return self.action_dist.proba_distribution(pre_squash_mean, self.log_std, latent_sde)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std = self._clamped_log_std()
            return self.action_dist.proba_distribution(pre_squash_mean, log_std)
        raise ValueError("Unsupported action distribution")

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        features = self.extract_features(obs)
        action_mean, values = self._decode_action_mean_and_value(features)
        distribution = self._get_action_dist_from_latent(action_mean)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        features = self.extract_features(obs)
        action_mean, _ = self._decode_action_mean_and_value(features)
        return self._get_action_dist_from_latent(action_mean)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        _, values = self._decode_action_mean_and_value(features)
        return values

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        distribution = self.get_distribution(observation)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        action_mean, values = self._decode_action_mean_and_value(features)
        distribution = self._get_action_dist_from_latent(action_mean)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        self._last_reg_loss = torch.zeros((), device=values.device)
        return values, log_prob, entropy

    def pop_regularization_loss(self) -> torch.Tensor:
        if self._last_reg_loss is None:
            return torch.zeros((), device=self.device)
        value = self._last_reg_loss
        self._last_reg_loss = None
        return value

    def _resolve_attention_module(
        self,
        attn: nn.Module,
        decoder_name: str,
        layer_idx: int,
        attn_idx: int,
    ) -> nn.MultiheadAttention:
        if hasattr(attn, "in_proj_weight"):
            return attn  # type: ignore[return-value]
        nested = getattr(attn, "attn", None)
        if nested is not None and hasattr(nested, "in_proj_weight"):
            return nested  # type: ignore[return-value]
        raise ValueError(
            f"Unsupported attention wrapper at {decoder_name}.layers[{layer_idx}].attentions[{attn_idx}]"
        )

    def _register_lora_on_attention(
        self,
        attn: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        dropout: float,
        enabled_targets: Tuple[str, ...] = ("q", "v"),
    ) -> nn.ModuleList:
        modules = nn.ModuleList()
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
            raise ValueError("LoRA on K projection is disabled")
        enabled_slices = tuple(slice_map[name] for name in enabled_targets)
        if not enabled_slices:
            raise ValueError("At least one projection target must be enabled for LoRA")

        attn.in_proj_weight.requires_grad = False
        if hasattr(attn, "in_proj_bias") and attn.in_proj_bias is not None:
            attn.in_proj_bias.requires_grad = False
        attn.out_proj.weight.requires_grad = False
        if attn.out_proj.bias is not None:
            attn.out_proj.bias.requires_grad = False

        in_lora = LoRAParametrization(
            attn.in_proj_weight.shape[1],
            attn.in_proj_weight.shape[0],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enabled_slices=enabled_slices,
        ).to(attn.in_proj_weight.device)
        parametrize.register_parametrization(attn, "in_proj_weight", in_lora)
        mask = in_lora.row_mask.view(-1)
        assert (
            mask[embed_dim : 2 * embed_dim].abs().sum() == 0
        ), "LoRA mask must not modify K projection"
        modules.append(in_lora)
        return modules

    def ensure_decoder_lora(
        self,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        dropout: Optional[float] = None,
    ) -> bool:
        rank = self.lora_rank if rank is None else rank
        alpha = self.lora_alpha if alpha is None else alpha
        dropout = self.lora_dropout if dropout is None else dropout
        if rank <= 0:
            return False
        if self._lora_injected:
            return True

        lora_modules = nn.ModuleList()
        for decoder_name, decoder in (
            ("ego_agent_decoder", self.ego_agent_decoder),
            ("ego_map_decoder", self.ego_map_decoder),
        ):
            layers = getattr(decoder, "layers", None)
            if layers is None:
                raise ValueError(f"{decoder_name} has no `layers` attribute")
            for layer_idx, layer in enumerate(layers):
                attentions = getattr(layer, "attentions", None)
                if attentions is None:
                    raise ValueError(f"{decoder_name}.layers[{layer_idx}] has no `attentions`")
                for attn_idx, attn in enumerate(attentions):
                    core_attn = self._resolve_attention_module(
                        attn, decoder_name, layer_idx, attn_idx
                    )
                    mods = self._register_lora_on_attention(
                        core_attn,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        enabled_targets=("q", "v"),
                    )
                    if mods:
                        lora_modules.extend(mods)
        self.lora_modules = lora_modules
        self._lora_injected = len(self.lora_modules) > 0
        return self._lora_injected

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

    def _apply_phase_constraints(self) -> None:
        self._set_module_trainability(self.ego_agent_decoder, False)
        self._set_module_trainability(self.ego_map_decoder, False)
        self._set_module_trainability(self.ego_fut_decoder, self.train_action)
        self._set_module_trainability(self.lora_modules, self.train_lora)
        self._set_module_trainability(self.value_adapter, True)
        self._set_module_trainability(self.value_net, True)
        self.log_std.requires_grad = self.train_log_std
        self._enforce_eval_mode_for_frozen()

    @staticmethod
    def _has_trainable_params(module: nn.Module) -> bool:
        return any(param.requires_grad for param in module.parameters())

    def _enforce_eval_mode_for_frozen(self) -> None:
        if not self._has_trainable_params(self.ego_agent_decoder):
            self.ego_agent_decoder.eval()
        if not self._has_trainable_params(self.ego_map_decoder):
            self.ego_map_decoder.eval()
        if not self._has_trainable_params(self.ego_fut_decoder):
            self.ego_fut_decoder.eval()
        if len(self.lora_modules) > 0 and not self._has_trainable_params(self.lora_modules):
            for module in self.lora_modules:
                module.eval()

    def _collect_actor_parameters(self, trainable_only: bool = True) -> List[nn.Parameter]:
        modules = [self.ego_fut_decoder, self.lora_modules]
        params: List[nn.Parameter] = []
        for module in modules:
            for param in module.parameters():
                if trainable_only and not param.requires_grad:
                    continue
                params.append(param)
        if not trainable_only or self.log_std.requires_grad:
            params.append(self.log_std)
        return params

    def _collect_critic_parameters(self, trainable_only: bool = True) -> List[nn.Parameter]:
        modules = [self.value_adapter, self.value_net]
        params: List[nn.Parameter] = []
        for module in modules:
            for param in module.parameters():
                if trainable_only and not param.requires_grad:
                    continue
                params.append(param)
        return params

    def _setup_optimizer_once(self) -> None:
        actor_params = self._collect_actor_parameters(trainable_only=True)
        critic_params = self._collect_critic_parameters(trainable_only=True)

        self._actor_group_idx = None
        self._critic_group_idx = None
        groups: List[Dict[str, Any]] = []
        if actor_params:
            self._actor_group_idx = len(groups)
            groups.append(dict(params=actor_params, lr=self.actor_lr))
        if critic_params:
            self._critic_group_idx = len(groups)
            groups.append(dict(params=critic_params, lr=self.critic_lr))
        if not groups:
            self._critic_group_idx = 0
            groups.append(dict(params=self.parameters(), lr=self.critic_lr))
        self.optimizer = self.optimizer_class(groups, **self.optimizer_kwargs)

    def _update_optimizer_lrs(self) -> None:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return
        if self._actor_group_idx is not None:
            self.optimizer.param_groups[self._actor_group_idx]["lr"] = self.actor_lr
        if self._critic_group_idx is not None:
            self.optimizer.param_groups[self._critic_group_idx]["lr"] = self.critic_lr

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
        self.phase = int(phase)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.log_std_bounds = log_std_bounds
        self._ensure_log_std_bounds()
        self.kl_reg_coef = float(kl_reg_coef)
        self.bc_reg_coef = float(bc_reg_coef)
        self.phase_behavior = (phase_behavior or {}).copy()

        behavior = self.phase_behavior
        train_action = bool(behavior.get("train_action", False))
        train_lora = bool(behavior.get("train_lora", False))
        train_log_std = bool(behavior.get("train_log_std", False))

        if train_lora:
            self.ensure_decoder_lora(
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
        self.train_action = train_action
        self.train_lora = train_lora and self._lora_injected
        self.train_log_std = train_log_std

        self._apply_phase_constraints()
        self._setup_optimizer_once()
        self._update_optimizer_lrs()
        self._log_phase_state()

    def _log_phase_state(self) -> None:
        actor_params = self._collect_actor_parameters(trainable_only=True)
        critic_params = self._collect_critic_parameters(trainable_only=True)
        print(
            f"[Phase {self.phase}] actor_tensors={len(actor_params)} "
            f"critic_tensors={len(critic_params)} lora={self.train_lora} action={self.train_action}"
        )
        if self.optimizer is None:
            return
        for idx, group in enumerate(self.optimizer.param_groups):
            trainable_count = sum(1 for p in group["params"] if p.requires_grad)
            print(
                f"  - opt_group[{idx}] lr={group.get('lr', 0.0):.2e} "
                f"trainable={trainable_count}/{len(group['params'])}"
            )

    def _get_optimizer_param_ids(self) -> Set[int]:
        if self.optimizer is None:
            return set()
        param_ids: Set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param_ids.add(id(param))
        return param_ids

    def train(self, mode: bool) -> "CustomVADPolicy":
        super().train(mode)
        self._enforce_eval_mode_for_frozen()
        return self
