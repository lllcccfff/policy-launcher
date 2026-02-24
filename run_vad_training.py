#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import tempfile
import atexit


from rl_framework.train_vad_ppo import evaluate, train, TrainingConfig

_TEMP_SCENE_DIRS = []


def _cleanup_temp_scene_dirs():
    for d in _TEMP_SCENE_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_SCENE_DIRS.clear()


atexit.register(_cleanup_temp_scene_dirs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAD PPO train/eval launcher")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    parser.add_argument("--with-eval", action="store_true", help="Enable eval env/callback during training")
    parser.add_argument("--model-path", type=str, default=None, help="Model path for eval or resume training")
    parser.add_argument("--phase", type=int, default=int(os.environ.get("VAD_TRAIN_PHASE", 0)), help="Training phase id")
    parser.add_argument("--total-timesteps", type=int, default=60_000, help="Total timesteps for training")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Eval callback frequency")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Eval episodes for callback")
    parser.add_argument("--n-eval-episodes", type=int, default=1, help="Eval episodes for --eval mode")
    parser.add_argument("--device", type=str, default=os.environ.get("VAD_DEVICE", "cuda"), help="Unified device for policy and VAD")
    parser.add_argument("--grpc-host", type=str, default=None, help="gRPC server host")
    parser.add_argument("--grpc-port", type=int, default=None, help="gRPC server port")
    parser.add_argument("--grpc-timeout-sec", type=float, default=None, help="gRPC timeout in seconds")
    parser.add_argument("--no-grpc-auto-wait-ready", action="store_true", help="Disable auto wait for gRPC channel readiness")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()
    config.training_phase = int(args.phase)
    config.apply_phase_defaults(config.training_phase)
    config.total_timesteps = int(args.total_timesteps)
    config.save_freq = int(args.save_freq)
    config.eval_freq = int(args.eval_freq)
    config.eval_episodes = int(args.eval_episodes)
    config.learning_rate = 1e-5

    config.device = str(args.device)
    config.policy_device = config.device
    config.uniad_device = config.device

    config.normalize_obs = False
    config.normalize_reward = False

    if args.grpc_host is not None:
        config.grpc_host = str(args.grpc_host)
    if args.grpc_port is not None:
        config.grpc_port = int(args.grpc_port)
    if args.grpc_timeout_sec is not None:
        config.grpc_timeout_sec = float(args.grpc_timeout_sec)
    if args.no_grpc_auto_wait_ready:
        config.grpc_auto_wait_ready = False
    return config


def _print_training_banner(config: TrainingConfig, with_eval: bool) -> None:
    print("=" * 70)
    print(" VAD PPO training launch script")
    print("=" * 70)
    print("\nConfiguration information:")
    print(f"  Device: {config.device}")
    print(f"  Policy Device: {config.policy_device} | VAD Device: {config.uniad_device}")
    print(f"  Training Phase: {config.training_phase}")
    print(f"  Total steps: {config.total_timesteps:,}")
    schedule_str = ", ".join(f"phase{p}:{ratio:.2f}" for p, ratio in config.phase_schedule)
    print(f"  Phase schedule: {schedule_str}")
    print(f"  Rollout steps: {config.n_steps}, Batch size: {config.batch_size}, Epochs: {config.n_epochs}")
    print(f"  Clip range: {config.clip_range}, Target KL: {config.target_kl}")
    print(f"  Actor LR: {config.actor_lr}, Critic LR: {config.critic_lr}")
    print(f"  With eval env: {with_eval}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Tensorboard: {config.tensorboard_log}")
    print(
        f"  gRPC endpoint: {config.grpc_host}:{config.grpc_port} "
        f"(timeout={config.grpc_timeout_sec}s)"
    )
    print("\n" + "=" * 70)
    print(" Start training")
    print("=" * 70)
    print("\nTips:")
    print("  - Use Ctrl+C to safely interrupt training")
    print("  - The training process will automatically save checkpoints")
    print("  - You can use Tensorboard to monitor the training progress:")
    print(f"    tensorboard --logdir={config.tensorboard_log}")
    print("")


def main() -> int:
    args = _parse_args()
    config = _build_config(args)

    if args.eval:
        model_path = args.model_path or os.path.join(config.checkpoint_dir, "ppo_vad_final.zip")
        if not os.path.exists(model_path):
            print(f"\nModel file not found: {model_path}")
            print("\nAvailable models:")
            if os.path.exists(config.checkpoint_dir):
                for file in os.listdir(config.checkpoint_dir):
                    if file.endswith(".zip"):
                        print(f"  - {file}")
            return 1
        print("=" * 70)
        print(" VAD model evaluation")
        print("=" * 70)
        print(f"\nModel path: {model_path}")
        print("\nStart evaluation...")
        evaluate(model_path=model_path, config=config, n_episodes=int(args.n_eval_episodes))
        return 0

    _print_training_banner(config, with_eval=bool(args.with_eval))

    try:
        train(
            config=config,
            with_eval_env=bool(args.with_eval),
            model_path=args.model_path,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print(" Training completed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
