from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch


@dataclass
class LatentTransFuserBundle:
    nets: List[torch.nn.Module]
    config: Any
    args: Dict[str, Any]
    device: torch.device
    model_dir: Path


def add_transfuser_to_path(transfuser_root: str | Path) -> Path:
    root = Path(transfuser_root).resolve()
    team_code_dir = root / "team_code_transfuser"
    if not team_code_dir.is_dir():
        raise FileNotFoundError(f"TransFuser team_code_transfuser directory not found: {team_code_dir}")
    team_code_str = str(team_code_dir)
    if team_code_str not in sys.path:
        sys.path.insert(0, team_code_str)
    return root


def _load_args(model_dir: Path) -> Dict[str, Any]:
    args_path = model_dir / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"Latent TransFuser args.txt not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _apply_official_config(config: Any, args: Dict[str, Any]) -> tuple[str, str, str, bool]:
    if "sync_batch_norm" in args:
        config.sync_batch_norm = bool(args["sync_batch_norm"])
    if "use_point_pillars" in args:
        config.use_point_pillars = args["use_point_pillars"]
    if "n_layer" in args:
        config.n_layer = args["n_layer"]
    if "use_target_point_image" in args:
        config.use_target_point_image = bool(args["use_target_point_image"])

    use_velocity = bool(args["use_velocity"]) if "use_velocity" in args else True
    image_architecture = args["image_architecture"] if "image_architecture" in args else "resnet34"
    lidar_architecture = args["lidar_architecture"] if "lidar_architecture" in args else "resnet18"
    backbone = args["backbone"] if "backbone" in args else "transFuser"
    if backbone != "latentTF":
        raise ValueError(f"Expected backbone='latentTF', got {backbone!r}")
    return backbone, image_architecture, lidar_architecture, use_velocity


def _checkpoint_paths(model_dir: Path, latent_config: Dict[str, Any]) -> list[Path]:
    checkpoint_path = latent_config.get("checkpoint_path")
    if checkpoint_path:
        paths = [Path(checkpoint_path).resolve()]
    else:
        paths = sorted(model_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Latent TransFuser checkpoint not found: {path}")
    return paths


def create_latent_transfuser(latent_config: Dict[str, Any]) -> LatentTransFuserBundle:
    transfuser_root = add_transfuser_to_path(latent_config["transfuser_root"])
    model_dir = Path(latent_config["model_dir"]).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Latent TransFuser model directory not found: {model_dir}")

    from config import GlobalConfig
    from model import LidarCenterNet

    device = torch.device(latent_config["device"])
    args = _load_args(model_dir)
    config = GlobalConfig(setting="eval")
    backbone, image_architecture, lidar_architecture, use_velocity = _apply_official_config(config, args)

    nets: list[torch.nn.Module] = []
    for checkpoint_path in _checkpoint_paths(model_dir, latent_config):
        print(checkpoint_path)
        net = LidarCenterNet(config, str(device), backbone, image_architecture, lidar_architecture, use_velocity)
        if config.sync_batch_norm:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"{checkpoint_path} missing state_dict keys: {missing}")
        if unexpected:
            print(f"[LatentTransFuser] ignored unexpected keys from {checkpoint_path}: {unexpected}")
        net.to(device).eval()
        for param in net.parameters():
            param.requires_grad = False
        nets.append(net)

    return LatentTransFuserBundle(
        nets=nets,
        config=config,
        args=args,
        device=device,
        model_dir=model_dir,
    )
