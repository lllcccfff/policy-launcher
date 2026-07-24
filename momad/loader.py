from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def create_momad(config: Dict[str, Any]):
    root = Path(config["momad_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MomAD root not found: {root}")

    checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MomAD checkpoint not found: {checkpoint_path}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    cwd = Path.cwd()
    os.chdir(root)
    try:
        import projects.mmdet3d_plugin  # noqa: F401

        cfg_path = root / "projects/configs/MomAD_small_stage2_roboAD.py"
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
