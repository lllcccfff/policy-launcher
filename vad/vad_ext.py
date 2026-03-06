import copy
from typing import Any, Dict, Optional

import torch
from mmdet.models import DETECTORS, HEADS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32

from projects.mmdet3d_plugin.VAD.VAD import VAD
from projects.mmdet3d_plugin.VAD.VAD_head import VADHead

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return value[0]
    return value


@HEADS.register_module()
class VADHeadFeature(VADHead):
    def _parse_padding_mask(
        self, key_tensor: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # key_tensor: [num_tokens, batch, dim]
        batch = key_tensor.shape[1]
        num_tokens = key_tensor.shape[0]
        if mask is None:
            return torch.zeros((batch, num_tokens), dtype=torch.bool, device=key_tensor.device)
        return mask.to(dtype=torch.bool)

    @force_fp32(apply_to=("mlvl_feats", "prev_bev"))
    def forward(  # type: ignore[override]
        self,
        mlvl_feats,
        img_metas,
        prev_bev=None,
        only_bev=False,
        feature_extractor: bool = False,
    ):
        if not feature_extractor:
            return super().forward(
                mlvl_feats,
                img_metas,
                prev_bev=prev_bev,
                only_bev=only_bev,
            )

        captured: Dict[str, torch.Tensor] = {}

        def _agent_pre_hook(module, args, kwargs):
            query = kwargs.get("query")
            query_pos = kwargs.get("query_pos")
            key = kwargs.get("key")
            key_pos = kwargs.get("key_pos")
            key_padding_mask = kwargs.get("key_padding_mask")
            if query is None or query_pos is None or key is None or key_pos is None:
                raise ValueError("ego_agent_decoder received incomplete kwargs")
            captured["ego_query"] = query.permute(1, 0, 2)
            captured["ego_agent_pos_emb"] = query_pos.permute(1, 0, 2)
            captured["agent_query"] = key.permute(1, 0, 2)
            captured["agent_pos_emb"] = key_pos.permute(1, 0, 2)
            captured["agent_mask"] = self._parse_padding_mask(key, key_padding_mask)

        def _map_pre_hook(module, args, kwargs):
            query_pos = kwargs.get("query_pos")
            key = kwargs.get("key")
            key_pos = kwargs.get("key_pos")
            key_padding_mask = kwargs.get("key_padding_mask")
            if query_pos is None or key is None or key_pos is None:
                raise ValueError("ego_map_decoder received incomplete kwargs")
            captured["ego_map_pos_emb"] = query_pos.permute(1, 0, 2)
            captured["map_query"] = key.permute(1, 0, 2)
            captured["map_pos_emb"] = key_pos.permute(1, 0, 2)
            captured["map_mask"] = self._parse_padding_mask(key, key_padding_mask)

        handles = []
        using_forward_wrapper = False
        agent_forward_backup = None
        map_forward_backup = None
        try:
            handles = [
                self.ego_agent_decoder.register_forward_pre_hook(
                    _agent_pre_hook, with_kwargs=True
                ),
                self.ego_map_decoder.register_forward_pre_hook(
                    _map_pre_hook, with_kwargs=True
                ),
            ]
        except TypeError:
            # Compatibility for older torch versions where register_forward_pre_hook
            # does not support with_kwargs.
            using_forward_wrapper = True
            agent_forward_backup = self.ego_agent_decoder.forward
            map_forward_backup = self.ego_map_decoder.forward

            def _agent_forward_wrapper(*args, **kwargs):
                _agent_pre_hook(self.ego_agent_decoder, args, kwargs)
                return agent_forward_backup(*args, **kwargs)

            def _map_forward_wrapper(*args, **kwargs):
                _map_pre_hook(self.ego_map_decoder, args, kwargs)
                return map_forward_backup(*args, **kwargs)

            self.ego_agent_decoder.forward = _agent_forward_wrapper
            self.ego_map_decoder.forward = _map_forward_wrapper
        try:
            outs = super().forward(
                mlvl_feats,
                img_metas,
                prev_bev=prev_bev,
                only_bev=only_bev,
            )
        finally:
            for handle in handles:
                handle.remove()
            if using_forward_wrapper:
                self.ego_agent_decoder.forward = agent_forward_backup
                self.ego_map_decoder.forward = map_forward_backup

        if not isinstance(outs, dict):
            raise ValueError("VADHead forward output must be a dict")

        captured["bev_embed"] = outs["bev_embed"]
        captured["ego_fut_preds"] = outs["ego_fut_preds"]

        required = [
            "ego_query",
            "ego_agent_pos_emb",
            "ego_map_pos_emb",
            "agent_query",
            "agent_pos_emb",
            "agent_mask",
            "map_query",
            "map_pos_emb",
            "map_mask",
            "bev_embed",
        ]
        missing = [key for key in required if key not in captured]
        if missing:
            raise ValueError(f"feature_extractor missing keys: {missing}")
        return captured

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            bboxes = LiDARInstance3DBoxes(bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list


@DETECTORS.register_module()
class VADFeature(VAD):
    def forward_test(  # type: ignore[override]
        self,
        img_metas,
        img=None,
        ego_fut_cmd=None,
        feature_extractor: bool = False,
        native_inference: bool = False,
        **kwargs,
    ):
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be a list, but got {type(img_metas)}")
        if not isinstance(img, torch.Tensor):
            raise TypeError(
                f"img must be a torch.Tensor with shape [1, 6, 3, H, W], got {type(img)}"
            )
        if img.ndim != 5:
            raise ValueError(
                f"img must be 5D [1, 6, 3, H, W], got shape {tuple(img.shape)}"
            )
        if tuple(int(v) for v in img.shape[:3]) != (1, 6, 3):
            raise ValueError(
                f"img shape prefix must be [1, 6, 3], got {tuple(int(v) for v in img.shape[:3])}"
            )
        if not img_metas or not img_metas[0] or "scene_token" not in img_metas[0][0]:
            raise ValueError("img_metas is malformed for VADFeature forward_test")

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, feature_dict = self.simple_test(
            img_metas=img_metas[0],
            img=img,
            prev_bev=self.prev_frame_info["prev_bev"],
            ego_fut_cmd=_unwrap_singleton(ego_fut_cmd),
            feature_extractor=feature_extractor,
            native_inference=native_inference,
            **kwargs,
        )
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return feature_dict

    def simple_test(  # type: ignore[override]
        self,
        img_metas,
        img=None,
        prev_bev=None,
        rescale=False,
        ego_fut_cmd=None,
        feature_extractor: bool = False,
        native_inference: bool = False,
        **kwargs,
    ):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        new_prev_bev, feature_dict, _ = self.simple_test_pts(
            x=img_feats,
            img_metas=img_metas,
            prev_bev=prev_bev,
            rescale=rescale,
            ego_fut_cmd=ego_fut_cmd,
            feature_extractor=feature_extractor,
            native_inference=native_inference,
        )
        if feature_extractor:
            return new_prev_bev, feature_dict

        bbox_list = [dict() for _ in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, feature_dict):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = {}
        return new_prev_bev, bbox_list

    def simple_test_pts(  # type: ignore[override]
        self,
        x,
        img_metas,
        prev_bev=None,
        rescale=False,
        ego_fut_cmd=None,
        feature_extractor: bool = False,
        native_inference: bool = False,
    ):
        outs = self.pts_bbox_head(
            x,
            img_metas,
            prev_bev=prev_bev,
            feature_extractor=feature_extractor,
        )
        if feature_extractor:
            return outs["bev_embed"], outs, {}

        bbox_list = self.pts_bbox_head.get_bboxes(outs)

        bbox_results = []
        cmd_tensor = ego_fut_cmd
        batch_size = int(outs["ego_fut_preds"].shape[0])
        if cmd_tensor is None:
            # Fallback to straight command when command is missing.
            cmd_tensor = torch.zeros(
                (batch_size, 1, 1, 3),
                dtype=torch.float32,
                device=outs["ego_fut_preds"].device,
            )
            cmd_tensor[..., 2] = 1.0
        if not isinstance(cmd_tensor, torch.Tensor):
            cmd_tensor = torch.as_tensor(cmd_tensor, dtype=torch.float32, device=outs["ego_fut_preds"].device)

        for i, (
            bboxes,
            scores,
            labels,
            trajs,
            map_bboxes,
            map_scores,
            map_labels,
            map_pts,
        ) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result["trajs_3d"] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            if cmd_tensor.ndim >= 1 and int(cmd_tensor.shape[0]) == batch_size:
                cmd_for_sample = cmd_tensor[i : i + 1]
            else:
                cmd_for_sample = cmd_tensor
            bbox_result["ego_fut_cmd"] = cmd_for_sample.detach().cpu()
            bbox_results.append(bbox_result)

        return outs["bev_embed"], bbox_results, {}
