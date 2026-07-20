from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


@dataclass(frozen=True)
class AutoVLABundle:
    model: torch.nn.Module
    config: Dict[str, Any]
    device: torch.device
    root: Path


def _add_autovla_root(autovla_root: str) -> Path:
    root = Path(autovla_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AutoVLA root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    navsim_str = str(root / "navsim")
    if navsim_str not in sys.path:
        sys.path.insert(0, navsim_str)
    return root


def _resolve_under_root(root: Path, path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_device(device_value: str) -> torch.device:
    device = torch.device(device_value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for AutoVLA inference.")
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return device


def _load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"AutoVLA checkpoint must be a dict, got {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"AutoVLA checkpoint state_dict must be a dict, got {type(state_dict)!r}")
    return state_dict


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("autovla.", "drivevla."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def create_autovla(config: Dict[str, Any]) -> AutoVLABundle:
    root = _add_autovla_root(str(config["autovla_root"]))
    config_path = _resolve_under_root(root, str(config["config_path"]))
    checkpoint_path = _resolve_under_root(root, config.get("checkpoint_path"))
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"AutoVLA config not found: {config_path}")
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"AutoVLA checkpoint not found: {checkpoint_path}")

    model_config = _load_yaml(config_path)
    model_config["model"]["pretrained_model_path"] = str(
        _resolve_under_root(root, str(model_config["model"]["pretrained_model_path"]))
    )
    model_config["model"]["codebook_cache_path"] = str(
        _resolve_under_root(root, str(model_config["model"]["codebook_cache_path"]))
    )

    device = _normalize_device(str(config.get("device", "cuda:0")))

    cwd = Path.cwd()
    os.chdir(root)
    try:
        from models.autovla import AutoVLA

        model = AutoVLA(model_config, device=str(device))
        if checkpoint_path is not None:
            state_dict = _load_checkpoint(checkpoint_path)
            load_result = model.load_state_dict(_strip_known_prefixes(state_dict), strict=False)
            print(
                "Loaded AutoVLA checkpoint "
                f"{checkpoint_path} missing={len(load_result.missing_keys)} "
                f"unexpected={len(load_result.unexpected_keys)}"
            )
            del state_dict
            gc.collect()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    finally:
        os.chdir(cwd)

    return AutoVLABundle(model=model, config=model_config, device=device, root=root)
