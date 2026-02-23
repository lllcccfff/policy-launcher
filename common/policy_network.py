import math
from typing import Dict, Optional, Tuple

import torch
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ADFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embed_dims: int = 256):
        super().__init__(observation_space, features_dim=embed_dims)
        self.embed_dims = embed_dims

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return observations


class LoRAParametrization(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        enabled_slices: Optional[Tuple[Tuple[int, int], ...]] = None,
    ):
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
        mask = torch.ones(fan_out, 1)
        if enabled_slices is not None:
            mask.zero_()
            for start, end in enabled_slices:
                mask[start:end] = 1.0
        self.register_buffer("row_mask", mask, persistent=False)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(self.up, self.dropout(self.down))
        delta = delta * self.row_mask
        return base_weight + self.scaling * delta