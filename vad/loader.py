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

from policy_launcher.common.mmcv_utils import cleanup_mmcv_registries


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


def create_vad(vad_config: Dict[str, Any]):
    config_path = vad_config["config_path"]
    checkpoint_path = vad_config["checkpoint_path"]
    device = vad_config["device"]
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
    import policy_launcher.vad.vad_ext  # noqa: F401

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
        cfg.model.type = "VADNative"

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

        for param in model.parameters():
            param.requires_grad = False
        return model
    finally:
        os.chdir(original_cwd)
