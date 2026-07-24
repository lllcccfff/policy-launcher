from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _add_sparsedrive_root(sparsedrive_root: str) -> Path:
    root = Path(sparsedrive_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SparseDrive root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def create_sparsedrive(config: Dict[str, Any]):
    root = _add_sparsedrive_root(str(config["sparsedrive_root"]))
    checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SparseDrive checkpoint not found: {checkpoint_path}")

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    cwd = Path.cwd()
    os.chdir(root)
    try:
        import projects.mmdet3d_plugin  # noqa: F401

        cfg_path = root / "projects/configs/sparsedrive_small_stage2.py"
        cfg = Config.fromfile(str(cfg_path))
        cfg.model.img_backbone.pretrained = None
        cfg.model.train_cfg = None
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
        load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    finally:
        os.chdir(cwd)

    device = torch.device(str(config["device"]))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
