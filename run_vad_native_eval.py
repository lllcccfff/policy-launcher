#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_VAD_CONFIG = "VAD/projects/configs/VAD/VAD_base_stage_2.py"
DEFAULT_VAD_CKPT = "VAD/ckpts/VAD_base.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate native VAD inference on gRPC env")
    parser.add_argument("--vad-config", type=str, default=DEFAULT_VAD_CONFIG, help="VAD config path")
    parser.add_argument("--vad-checkpoint", type=str, default=DEFAULT_VAD_CKPT, help="VAD checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="VAD inference device")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--timeout", type=float, default=30.0, help="gRPC timeout seconds")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--save-video", action="store_true", help="Save episode videos")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="Output directory")
    parser.add_argument("--save-json", action="store_true", help="Save eval results to json/txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv
    from rl_framework.vad.loader import create_vad
    from rl_framework.vad.native_agent import VADNativeAgent
    from rl_framework.vad.native_eval import evaluate_vad_native

    output_dir = Path(args.output_dir)
    if args.save_video or args.save_json:
        output_dir.mkdir(parents=True, exist_ok=True)

    vad_cfg = {
        "config_path": args.vad_config,
        "checkpoint_path": args.vad_checkpoint,
        "device": args.device,
        "runtime_mode": "native_infer",
    }
    vad_model = create_vad(vad_cfg)
    agent = VADNativeAgent(vad_model)

    env = GrpcClientEnv(
        host=args.host,
        port=args.port,
        timeout_sec=args.timeout,
        auto_wait_ready=True,
    )
    try:
        evaluate_vad_native(
            env=env,
            agent=agent,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            save_video=bool(args.save_video),
            output_dir=output_dir,
            save_json=bool(args.save_json),
        )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

