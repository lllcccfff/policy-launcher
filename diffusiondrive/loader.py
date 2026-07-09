from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _add_diffusiondrive_root(diffusiondrive_root: str) -> Path:
    root = Path(diffusiondrive_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DiffusionDrive root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def create_diffusiondrive(config: Dict[str, Any]):
    root = _add_diffusiondrive_root(str(config["diffusiondrive_root"]))
    checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DiffusionDrive checkpoint not found: {checkpoint_path}")

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    cwd = Path.cwd()
    os.chdir(root)
    try:
        import projects.mmdet3d_plugin  # noqa: F401
        import projects.mmdet3d_plugin.models.motion.motion_blocks_v11  # noqa: F401

        cfg_path = root / "projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
        cfg = Config.fromfile(str(cfg_path))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    finally:
        os.chdir(cwd)

    device = torch.device(str(config["device"]))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
