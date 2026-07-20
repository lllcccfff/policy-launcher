from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class AlpamayoR1Bundle:
    model: torch.nn.Module
    processor: Any
    device: torch.device


def _add_alpamayo_src(alpamayo_root: str) -> Path:
    root = Path(alpamayo_root).expanduser().resolve()
    source = root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Alpamayo source directory not found: {source}")
    source_str = str(source)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    return root


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Alpamayo model directory not found: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Alpamayo model config not found: {path / 'config.json'}")
    return path


def _select_cuda_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        raise ValueError(f"Alpamayo R1 inference requires a CUDA device, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for Alpamayo R1 inference.")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def create_alpamayo_r1(config: Dict[str, Any]) -> AlpamayoR1Bundle:
    _add_alpamayo_src(str(config["alpamayo_root"]))
    model_path = _resolve_model_path(str(config["model_path"]))
    device = _select_cuda_device(torch.device(str(config.get("device", "cuda"))))

    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    model = AlpamayoR1.from_pretrained(str(model_path), dtype=torch.bfloat16).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    processor = helper.get_processor(model.tokenizer)
    return AlpamayoR1Bundle(model=model, processor=processor, device=device)
