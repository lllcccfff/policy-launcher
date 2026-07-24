#!/usr/bin/env python3
"""Run a native planning model on a gRPC environment."""

from __future__ import annotations

import argparse
from importlib import import_module

from rl_framework.common.grpc_obs_adapter import unpack_ad_observation

def _is_ordered_scenario_exhausted(exc: BaseException) -> bool:
    message = str(exc)
    return "No more scenarios to evaluate" in message or "No completed scene metrics" in message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native planning model on a gRPC environment")
    parser.add_argument(
        "--model",
        choices=("uniad", "vad", "genad", "momad", "stp3", "opendrivevla", "latent_transfuser", "diffusiondrive", "sparsedrive", "alpamayo1", "alpamayo1.5", "autovla", "epona", "openemma_gpt", "openemma_qwen", "openemma_llava", "openemma_llama"),
        required=True,
        help="Native planning model to run",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50052, help="gRPC server port")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--timeout", type=float, default=360.0, help="gRPC timeout seconds")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--scene-name", type=str, help="Run only this scene")
    return parser.parse_args()


def route_agent(model_name: str, device: str):
    agent_map = {
        "uniad": ("rl_framework.uniad.native_agent", "UniADNativeAgent", True),
        "vad": ("rl_framework.vad.native_agent", "VADNativeAgent", True),
        "genad": ("rl_framework.genad.native_agent", "GenADNativeAgent", True),
        "momad": ("rl_framework.momad.native_agent", "MomADNativeAgent", True),
        "stp3": ("rl_framework.stp3.native_agent", "STP3NativeAgent", True),
        "opendrivevla": ("rl_framework.opendrivevla.native_agent", "OpenDriveVLANativeAgent", True),
        "latent_transfuser": ("rl_framework.latent_transfuser.native_agent", "LatentTransFuserNativeAgent", True),
        "diffusiondrive": ("rl_framework.diffusiondrive.native_agent", "DiffusionDriveNativeAgent", True),
        "sparsedrive": ("rl_framework.sparsedrive.native_agent", "SparseDriveNativeAgent", True),
        "alpamayo1": ("rl_framework.alpamayo_r1.native_agent", "AlpamayoR1NativeAgent", True),
        "alpamayo1.5": ("rl_framework.alpamayo1_5.native_agent", "Alpamayo15NativeAgent", True),
        "autovla": ("rl_framework.autovla.native_agent", "AutoVLANativeAgent", True),
        "epona": ("rl_framework.epona.native_agent", "EponaNativeAgent", True),
        "openemma_gpt": ("rl_framework.openemma.native_agent", "OpenEMMANativeAgentGPT", False),
        "openemma_qwen": ("rl_framework.openemma.native_agent", "OpenEMMANativeAgentQwen", False),
        "openemma_llava": ("rl_framework.openemma.native_agent", "OpenEMMANativeAgentLLaVA", False),
        "openemma_llama": ("rl_framework.openemma.native_agent", "OpenEMMANativeAgentLlama", False),
    }
    module_name, class_name, needs_device = agent_map[model_name]
    agent_class = getattr(import_module(module_name), class_name)
    return agent_class(device) if needs_device else agent_class()


def main() -> int:
    args = parse_args()

    from metadrive.envs.grpc_client_env import GrpcClientEnv

    agent = route_agent(args.model, args.device)
    env = GrpcClientEnv(host=args.host, port=args.port, timeout_sec=args.timeout, auto_wait_ready=True)
    try:
        while True:
            try:
                obs, reset_info = env.reset()
            except Exception as exc:
                if _is_ordered_scenario_exhausted(exc):
                    break
                raise

            obs_img, obs_info, _, _ = unpack_ad_observation(obs, reset_info)
            scene_name = str(reset_info["scene_name"])

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            for _ in range(args.max_steps):
                action = agent.predict_action(obs_img, obs_info)

                obs, reward, terminated, truncated, step_info = env.step(action)
                obs_img, obs_info, _, _ = unpack_ad_observation(obs, step_info)

                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break

            print(
                f"[Scene {scene_name}] steps={steps} reward={total_reward:.3f} "
                f"model={args.model} terminated={terminated} truncated={truncated}"
            )

            if args.scene_name is not None:
                break

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
