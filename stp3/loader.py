from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from rl_framework.stp3.dataparser import DEFAULT_STP3_ROOT, ensure_stp3_root


def create_stp3(stp3_config: Dict[str, Any]) -> torch.nn.Module:
    stp3_root = ensure_stp3_root(stp3_config.get("stp3_root", DEFAULT_STP3_ROOT))
    checkpoint_path = Path(stp3_config["checkpoint_path"]).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ST-P3 checkpoint not found: {checkpoint_path}")

    device = torch.device(str(stp3_config.get("device", "cuda")))

    from stp3.trainer import TrainingModule

    trainer = TrainingModule.load_from_checkpoint(str(checkpoint_path), strict=False, map_location=device)
    trainer.eval()
    trainer.to(device)
    model = trainer.model.eval()

    cfg = model.cfg
    if str(cfg.DATASET.NAME) != "carla":
        raise ValueError(f"ST-P3 gRPC rollout expects a CARLA-config checkpoint, got DATASET.NAME={cfg.DATASET.NAME!r}")
    if not bool(cfg.PLANNING.ENABLED):
        raise ValueError("ST-P3 gRPC rollout requires cfg.PLANNING.ENABLED=True")

    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded ST-P3 checkpoint: {checkpoint_path}")
    print(f"Using ST-P3 root: {stp3_root}")
    return model
