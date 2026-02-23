from typing import Any, Dict

import numpy as np
import torch
from gymnasium import spaces

from rl_framework.uniad.env_wrapper import UniADWrapper
from rl_framework.vad.dataparser import command_to_one_hot, get_vad_img_norm_cfg


class VADWrapper(UniADWrapper):
    def __init__(
        self,
        env,
        vad_model: torch.nn.Module,
        dataparser,
        embed_dims: int = 256,
        planning_steps: int = 6,
        uniad_device: str = "cuda:0",
        policy_device: str = "cuda:1",
    ):
        super().__init__(
            env=env,
            uniad_model=vad_model,
            dataparser=dataparser,
            embed_dims=embed_dims,
            bev_h=1,
            bev_w=1,
            planning_steps=planning_steps,
            uniad_device=uniad_device,
            policy_device=policy_device,
        )
        self.vad_model = self.uniad_model
        model_impl = self.vad_model.module if hasattr(self.vad_model, "module") else self.vad_model
        if not hasattr(model_impl, "pts_bbox_head"):
            raise ValueError("VADWrapper expects model.pts_bbox_head")
        head = model_impl.pts_bbox_head
        self.max_agent_queries = int(head.num_query)
        self.max_map_queries = int(head.map_num_vec)
        self.ego_fut_mode = int(head.ego_fut_mode)
        fut_ts = int(head.fut_ts)
        if fut_ts != self.planning_steps:
            raise ValueError(
                f"planning_steps ({self.planning_steps}) must match VAD fut_ts ({fut_ts})"
            )
        if self.ego_fut_mode != 3:
            raise ValueError(f"VADWrapper expects ego_fut_mode=3, got {self.ego_fut_mode}")

        self.observation_space = spaces.Dict(
            {
                "ego_query": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, self.embed_dims), dtype=np.float32
                ),
                "ego_pos_emb": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, self.embed_dims), dtype=np.float32
                ),
                "agent_query": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_agent_queries, self.embed_dims),
                    dtype=np.float32,
                ),
                "agent_pos_emb": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_agent_queries, self.embed_dims),
                    dtype=np.float32,
                ),
                "agent_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(self.max_agent_queries,), dtype=np.float32
                ),
                "map_query": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_map_queries, self.embed_dims),
                    dtype=np.float32,
                ),
                "map_pos_emb": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_map_queries, self.embed_dims),
                    dtype=np.float32,
                ),
                "map_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(self.max_map_queries,), dtype=np.float32
                ),
                "ego_fut_cmd": spaces.Box(
                    low=0.0, high=1.0, shape=(self.ego_fut_mode,), dtype=np.float32
                ),
            }
        )

    @staticmethod
    def _require_tensor(data: Dict[str, Any], key: str) -> torch.Tensor:
        value = data.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"feature_extractor output `{key}` must be a torch.Tensor")
        return value

    @staticmethod
    def _to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        return mask > 0.5

    def _pad_tokens(
        self,
        query: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor,
        max_tokens: int,
    ):
        if query.ndim != 3 or pos_emb.ndim != 3 or mask.ndim != 2:
            raise ValueError("query/pos_emb/mask shapes are invalid")
        if query.shape[:2] != pos_emb.shape[:2] or query.shape[:2] != mask.shape[:2]:
            raise ValueError("query/pos_emb/mask token dimensions mismatch")

        batch, num_tokens, dim = query.shape
        if dim != self.embed_dims:
            raise ValueError(f"token dim mismatch: expected {self.embed_dims}, got {dim}")
        if batch != 1:
            raise ValueError(f"VADWrapper currently supports batch size 1, got {batch}")

        if num_tokens > max_tokens:
            query = query[:, :max_tokens]
            pos_emb = pos_emb[:, :max_tokens]
            mask = mask[:, :max_tokens]
            num_tokens = max_tokens
        elif num_tokens < max_tokens:
            pad = max_tokens - num_tokens
            query = torch.cat(
                [query, torch.zeros((batch, pad, dim), device=query.device, dtype=query.dtype)],
                dim=1,
            )
            pos_emb = torch.cat(
                [
                    pos_emb,
                    torch.zeros((batch, pad, dim), device=pos_emb.device, dtype=pos_emb.dtype),
                ],
                dim=1,
            )
            pad_mask = torch.ones((batch, pad), device=mask.device, dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=1)
        return query, pos_emb, mask

    def _process_observation(
        self,
        obs_img: Dict[str, Any],
        obs_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        img_norm_cfg = get_vad_img_norm_cfg()
        cameras = {"FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"}

        raw_data = self.dataparser(obs_img, obs_info, cameras, img_norm_cfg)
        if isinstance(raw_data, dict):
            raw_data = self._to_device(raw_data, self.uniad_device)
        if "raw_imgs" in raw_data:
            del raw_data["raw_imgs"]

        with torch.no_grad():
            model_output = self.vad_model(
                return_loss=False,
                rescale=True,
                feature_extractor=True,
                **raw_data,
            )
        if not isinstance(model_output, dict):
            raise ValueError("VAD feature_extractor output must be a dict")

        ego_query = self._require_tensor(model_output, "ego_query")
        ego_pos_emb = self._require_tensor(model_output, "ego_pos_emb")
        agent_query = self._require_tensor(model_output, "agent_query")
        agent_pos_emb = self._require_tensor(model_output, "agent_pos_emb")
        agent_mask = self._to_bool_mask(self._require_tensor(model_output, "agent_mask"))
        map_query = self._require_tensor(model_output, "map_query")
        map_pos_emb = self._require_tensor(model_output, "map_pos_emb")
        map_mask = self._to_bool_mask(self._require_tensor(model_output, "map_mask"))

        agent_query, agent_pos_emb, agent_mask = self._pad_tokens(
            agent_query, agent_pos_emb, agent_mask, self.max_agent_queries
        )
        map_query, map_pos_emb, map_mask = self._pad_tokens(
            map_query, map_pos_emb, map_mask, self.max_map_queries
        )

        cmd_one_hot = command_to_one_hot(int(obs_info["command"]), ego_query.device)

        ego_query = ego_query.to(self.policy_device)
        ego_pos_emb = ego_pos_emb.to(self.policy_device)
        agent_query = agent_query.to(self.policy_device)
        agent_pos_emb = agent_pos_emb.to(self.policy_device)
        agent_mask = agent_mask.to(self.policy_device)
        map_query = map_query.to(self.policy_device)
        map_pos_emb = map_pos_emb.to(self.policy_device)
        map_mask = map_mask.to(self.policy_device)
        cmd_one_hot = cmd_one_hot.to(self.policy_device)

        processed_obs = {
            "ego_query": ego_query[0].cpu().numpy().astype(np.float32),
            "ego_pos_emb": ego_pos_emb[0].cpu().numpy().astype(np.float32),
            "agent_query": agent_query[0].cpu().numpy().astype(np.float32),
            "agent_pos_emb": agent_pos_emb[0].cpu().numpy().astype(np.float32),
            "agent_mask": agent_mask[0].float().cpu().numpy().astype(np.float32),
            "map_query": map_query[0].cpu().numpy().astype(np.float32),
            "map_pos_emb": map_pos_emb[0].cpu().numpy().astype(np.float32),
            "map_mask": map_mask[0].float().cpu().numpy().astype(np.float32),
            "ego_fut_cmd": cmd_one_hot.cpu().numpy().astype(np.float32),
        }
        return processed_obs
