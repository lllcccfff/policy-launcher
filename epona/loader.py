from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class EponaBundle:
    model: torch.nn.Module
    tokenizer: torch.nn.Module
    args: Any
    device: torch.device


def _add_epona_root(epona_root: str) -> Path:
    root = Path(epona_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Epona root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _resolve_under_root(root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _select_cuda_device(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Epona inference requires a CUDA device, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for Epona inference.")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    torch.cuda.set_device(index)
    return index


def create_epona(config: Dict[str, Any]) -> EponaBundle:
    root = _add_epona_root(str(config["epona_root"]))
    config_path = _resolve_under_root(root, str(config["config_path"]))
    checkpoint_path = _resolve_under_root(root, str(config["checkpoint_path"]))
    vae_checkpoint_path = _resolve_under_root(root, str(config["vae_checkpoint_path"]))

    if not config_path.is_file():
        raise FileNotFoundError(f"Epona config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Epona checkpoint not found: {checkpoint_path}")
    if not vae_checkpoint_path.is_file():
        raise FileNotFoundError(f"Epona VAE checkpoint not found: {vae_checkpoint_path}")

    device = torch.device(str(config.get("device", "cuda")))
    local_rank = _select_cuda_device(device)

    from utils.config_utils import Config
    from models.model import TrainTransformersDiT
    from models.modules.tokenizer import VAETokenizer

    cwd = Path.cwd()
    os.chdir(root)
    try:
        args = Config.fromfile(str(config_path))
        args.batch_size = int(config.get("batch_size", 1))
        args.vae_ckpt = str(vae_checkpoint_path)
        if "num_sampling_steps" in config and config["num_sampling_steps"] is not None:
            args.num_sampling_steps = int(config["num_sampling_steps"])

        model = TrainTransformersDiT(
            args,
            load_path=str(checkpoint_path),
            local_rank=local_rank,
            condition_frames=args.condition_frames,
        )
        tokenizer = VAETokenizer(args, local_rank)
    finally:
        os.chdir(cwd)

    model.eval()
    tokenizer.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in tokenizer.parameters():
        param.requires_grad = False

    return EponaBundle(model=model, tokenizer=tokenizer, args=args, device=torch.device(f"cuda:{local_rank}"))
