from __future__ import annotations

import copy

import torch
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.VAD.VAD import VAD


@DETECTORS.register_module()
class VADNative(VAD):
    """VAD planning inference on simulator observations."""

    def forward_test(
        self,
        img_metas,
        img=None,
        ego_fut_cmd=None,
        **kwargs,
    ):
        del kwargs
        if not isinstance(img_metas, list) or len(img_metas) != 1:
            raise TypeError("img_metas must contain exactly one test-time augmentation")
        if not isinstance(img, torch.Tensor) or img.ndim != 5:
            raise TypeError("img must be a tensor with shape [1, 6, 3, H, W]")
        if tuple(int(value) for value in img.shape[:3]) != (1, 6, 3):
            raise ValueError(f"Expected image shape prefix [1, 6, 3], got {tuple(img.shape[:3])}")
        if not isinstance(ego_fut_cmd, torch.Tensor) or ego_fut_cmd.numel() != 3:
            raise ValueError("ego_fut_cmd must be a three-element tensor")

        frame_metas = img_metas[0]
        if len(frame_metas) != 1:
            raise ValueError(f"VAD native inference requires batch size 1, got {len(frame_metas)}")
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
        )
        self.prev_frame_info["prev_pos"] = current_position
        self.prev_frame_info["prev_angle"] = current_angle
        self.prev_frame_info["prev_bev"] = outputs["bev_embed"]

        return [
            {
                "pts_bbox": {
                    "ego_fut_preds": outputs["ego_fut_preds"][0].cpu(),
                    "ego_fut_cmd": ego_fut_cmd.detach().cpu(),
                }
            }
        ]
