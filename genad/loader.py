from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model

from rl_framework.common.mmcv_utils import cleanup_mmcv_registries


def _import_plugin(cfg: Config, config_rel_path: str) -> None:
    if not cfg.get("plugin", False):
        return
    module_dir = os.path.dirname(cfg.plugin_dir) if cfg.get("plugin_dir") else os.path.dirname(config_rel_path)
    module_path = module_dir.replace("/", ".").replace("\\", ".").strip(".")
    importlib.import_module(module_path)


def create_genad(genad_config: Dict[str, Any]):
    config_path = os.path.abspath(genad_config["config_path"])
    checkpoint_path = os.path.abspath(genad_config["checkpoint_path"])
    genad_root = os.path.abspath(genad_config["AD_root"])
    device = torch.device(genad_config["device"])

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.isdir(genad_root):
        raise FileNotFoundError(f"GenAD root not found: {genad_root}")

    cleanup_mmcv_registries()
    if genad_root not in sys.path:
        sys.path.insert(0, genad_root)

    original_cwd = os.getcwd()
    try:
        os.chdir(genad_root)
        config_rel_path = os.path.relpath(config_path, genad_root)
        cfg = Config.fromfile(config_rel_path)
        _import_plugin(cfg, config_rel_path)
        import rl_framework.genad.genad_ext  # noqa: F401

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.type = "GenADNative"
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

        if cfg.get("fp16") is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        if "PALETTE" in checkpoint.get("meta", {}):
            model.PALETTE = checkpoint["meta"]["PALETTE"]

        model = model.to(device).eval()
        if device.type == "cuda":
            device_id = device.index if device.index is not None else torch.cuda.current_device()
            model = MMDataParallel(model, device_ids=[device_id]).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model
    finally:
        os.chdir(original_cwd)
