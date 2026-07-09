from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _add_opendrivevla_root(opendrivevla_root: str) -> Path:
    root = Path(opendrivevla_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OpenDriveVLA root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def create_opendrivevla(config: Dict[str, Any]):
    root = _add_opendrivevla_root(str(config["opendrivevla_root"]))
    checkpoint = Path(str(config["checkpoint_path"])).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OpenDriveVLA checkpoint directory not found: {checkpoint}")

    device = torch.device(str(config["device"]))
    attn_implementation = str(config.get("attn_implementation", "flash_attention_2"))

    cwd = Path.cwd()
    os.chdir(root)
    try:
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        tokenizer, model, _, context_len = load_pretrained_model(
            str(checkpoint),
            model_base=None,
            model_name="llava_qwen",
            device_map=device,
            multimodal=True,
            attn_implementation=attn_implementation,
            overwrite_config={
                "image_aspect_ratio": "pad",
                "vision_tower_test_mode": True,
            },
        )
    finally:
        os.chdir(cwd)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model, context_len
