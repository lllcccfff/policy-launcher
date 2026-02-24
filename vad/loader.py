import copy
import importlib
import os
import sys
import warnings
from typing import Any, Dict

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model

from rl_framework.common.mmcv_utils import cleanup_mmcv_registries


def _import_plugin_from_cfg(cfg: Config, config_rel_path: str) -> None:
    if not getattr(cfg, "plugin", False):
        return
    if hasattr(cfg, "plugin_dir"):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_rel_path)
    module_path = module_dir.replace("/", ".").replace("\\", ".").strip(".")
    if not module_path:
        raise ValueError("Cannot resolve plugin module path from cfg")
    importlib.import_module(module_path)


def _to_cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def create_vad(vad_config: Dict[str, Any]):
    config_path = vad_config["config_path"]
    checkpoint_path = vad_config["checkpoint_path"]
    device = vad_config["device"]
    runtime_mode = str(vad_config.get("runtime_mode", "feature_rl")).strip().lower()
    supported_modes = {"feature_rl", "native_infer"}
    if runtime_mode not in supported_modes:
        raise ValueError(
            f"Unsupported runtime_mode `{runtime_mode}`. "
            f"Expected one of {sorted(supported_modes)}."
        )
    warnings.filterwarnings("ignore")

    cleanup_mmcv_registries()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ad_root = str(vad_config.get("AD_root", "")).strip()
    if not ad_root:
        raise ValueError("vad_config must define `AD_root`")

    config_path_abs = os.path.abspath(config_path)
    checkpoint_path_abs = os.path.abspath(checkpoint_path)
    vad_root = os.path.abspath(ad_root)

    if vad_root not in sys.path:
        sys.path.insert(0, vad_root)

    # Ensure extension classes are registered before build_model().
    import rl_framework.vad.vad_ext  # noqa: F401

    original_cwd = os.getcwd()

    try:
        os.chdir(vad_root)
        config_rel_path = os.path.relpath(config_path_abs, vad_root)
        cfg = Config.fromfile(config_rel_path)

        if cfg.get("custom_imports", None):
            from mmcv.utils import import_modules_from_strings

            import_modules_from_strings(**cfg["custom_imports"])

        _import_plugin_from_cfg(cfg, config_rel_path)

        if cfg.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.type = "VADFeature"
        if not hasattr(cfg.model, "pts_bbox_head"):
            raise ValueError("cfg.model.pts_bbox_head is required for VAD")
        cfg.model.pts_bbox_head.type = "VADHeadFeature"

        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, checkpoint_path_abs, map_location="cpu")
        model = fuse_conv_bn(model)

        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        if "PALETTE" in checkpoint.get("meta", {}):
            model.PALETTE = checkpoint["meta"]["PALETTE"]

        if isinstance(device, str) and device.startswith("cuda"):
            device_id = 0
            if ":" in device:
                device_id = int(device.split(":")[1])
            model = MMDataParallel(model, device_ids=[device_id]).eval()
        else:
            model = model.to(device).eval()

        model_impl = model.module if hasattr(model, "module") else model
        model_impl.runtime_mode = runtime_mode
        if hasattr(model, "__dict__"):
            model.runtime_mode = runtime_mode

        for param in model.parameters():
            param.requires_grad = False
        return model
    finally:
        os.chdir(original_cwd)


def extract_vad_planning_weights(model) -> Dict[str, Any]:
    model_impl = model.module if hasattr(model, "module") else model
    runtime_mode = getattr(model_impl, "runtime_mode", getattr(model, "runtime_mode", "feature_rl"))
    if runtime_mode != "feature_rl":
        raise ValueError(
            "extract_vad_planning_weights only supports runtime_mode='feature_rl'. "
            f"Got runtime_mode='{runtime_mode}'."
        )
    if not hasattr(model_impl, "pts_bbox_head"):
        raise ValueError("VAD model must have pts_bbox_head")
    head = model_impl.pts_bbox_head
    required = ["ego_agent_decoder", "ego_map_decoder", "ego_fut_decoder"]
    for name in required:
        if not hasattr(head, name):
            raise ValueError(f"pts_bbox_head missing required planning module: {name}")

    return {
        "ego_agent_decoder_state_dict": _to_cpu_state_dict(head.ego_agent_decoder),
        "ego_map_decoder_state_dict": _to_cpu_state_dict(head.ego_map_decoder),
        "ego_fut_decoder_state_dict": _to_cpu_state_dict(head.ego_fut_decoder),
        "ego_agent_decoder_template": copy.deepcopy(head.ego_agent_decoder).cpu(),
        "ego_map_decoder_template": copy.deepcopy(head.ego_map_decoder).cpu(),
        "ego_fut_decoder_template": copy.deepcopy(head.ego_fut_decoder).cpu(),
        "ego_fut_mode": int(head.ego_fut_mode),
        "fut_ts": int(head.fut_ts),
        "embed_dims": int(head.embed_dims),
        "num_agent_queries": int(head.num_query),
        "num_map_queries": int(head.map_num_vec),
    }
