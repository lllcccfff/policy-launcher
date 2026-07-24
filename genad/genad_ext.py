from __future__ import annotations

import copy
from typing import Any

import torch
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.GenAD.GenAD import GenAD


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one test-time augmentation, got {len(value)}")
        return value[0]
    return value


@DETECTORS.register_module()
class GenADNative(GenAD):
    """GenAD inference path that does not require offline evaluation labels."""

    def forward_test(
        self,
        img_metas,
        img=None,
        ego_his_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        **kwargs,
    ):
        del kwargs
        if not isinstance(img_metas, list) or len(img_metas) != 1:
            raise TypeError("img_metas must contain exactly one test-time augmentation")
        if not isinstance(img, torch.Tensor) or img.ndim != 5:
            raise TypeError("img must be a tensor with shape [1, 6, 3, H, W]")
        if tuple(int(value) for value in img.shape[:3]) != (1, 6, 3):
            raise ValueError(f"Expected image shape prefix [1, 6, 3], got {tuple(img.shape[:3])}")

        frame_metas = img_metas[0]
        if len(frame_metas) != 1:
            raise ValueError(f"GenAD native inference requires batch size 1, got {len(frame_metas)}")
        frame_meta = frame_metas[0]
        scene_token = frame_meta["scene_token"]

        if scene_token != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = scene_token
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        current_position = copy.deepcopy(frame_meta["can_bus"][:3])
        current_angle = copy.deepcopy(frame_meta["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is None:
            frame_meta["can_bus"][:3] = 0
            frame_meta["can_bus"][-1] = 0
        else:
            frame_meta["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            frame_meta["can_bus"][-1] -= self.prev_frame_info["prev_angle"]

        img_feats = self.extract_feat(img=img, img_metas=frame_metas)
        outputs = self.pts_bbox_head(
            img_feats,
            frame_metas,
            prev_bev=self.prev_frame_info["prev_bev"],
            ego_his_trajs=_unwrap_singleton(ego_his_trajs),
            ego_lcf_feat=_unwrap_singleton(ego_lcf_feat),
        )

        self.prev_frame_info["prev_pos"] = current_position
        self.prev_frame_info["prev_angle"] = current_angle
        self.prev_frame_info["prev_bev"] = outputs["bev_embed"]

        command = _unwrap_singleton(ego_fut_cmd)
        if not isinstance(command, torch.Tensor) or command.numel() != 3:
            raise ValueError("ego_fut_cmd must be a three-element tensor")

        return [
            {
                "pts_bbox": {
                    "ego_fut_preds": outputs["ego_fut_preds"][0],
                    "ego_fut_cmd": command,
                }
            }
        ]
