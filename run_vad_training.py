#!/usr/bin/env python3
import sys
import os
import shutil
import tempfile
import atexit

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gaussiandrive_path = os.path.join(project_root, "simulator/GaussianDrive")
easydrive_path = os.path.join(project_root, "simulator/EasyDrive")
streetgs_path = os.path.join(project_root, "simulator/street_gaussians")

sys.path.insert(0, project_root)
sys.path.insert(0, gaussiandrive_path)
sys.path.insert(0, easydrive_path)
sys.path.insert(0, streetgs_path)

os.chdir(gaussiandrive_path)

from rl_framework.train_vad_ppo import train, evaluate, TrainingConfig

_TEMP_SCENE_DIRS = []


def _cleanup_temp_scene_dirs():
    for d in _TEMP_SCENE_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_SCENE_DIRS.clear()


atexit.register(_cleanup_temp_scene_dirs)


def main():
    print("=" * 70)
    print(" VAD PPO training launch script")
    print("=" * 70)

    config = TrainingConfig()
    config.total_timesteps = 75_000
    config.save_freq = 15_000
    config.eval_freq = 20_000
    config.eval_episodes = 3
    config.learning_rate = 1e-5
    config.policy_device = "cuda:0"
    config.uniad_device = "cuda:0"
    config.normalize_obs = False
    config.normalize_reward = False
    config.training_phase = int(os.environ.get("VAD_TRAIN_PHASE", 0))
    config.apply_phase_defaults()

    print("\nConfiguration information:")
    print(f"  Policy Device: {config.policy_device} | VAD Device: {config.uniad_device}")
    print(f"  Training Phase: {config.training_phase}")
    print(f"  Total steps: {config.total_timesteps:,}")
    schedule_str = ", ".join(f"phase{p}:{ratio:.2f}" for p, ratio in config.phase_schedule)
    print(f"  Phase schedule: {schedule_str}")
    print(f"  Rollout steps: {config.n_steps}, Batch size: {config.batch_size}, Epochs: {config.n_epochs}")
    print(f"  Clip range: {config.clip_range}, Target KL: {config.target_kl}")
    print(f"  Actor LR: {config.actor_lr}, Critic LR: {config.critic_lr}")
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

    model_path = None
    try:
        train(config=config, with_eval_env=True, model_path=model_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print(" Training completed")
    print("=" * 70)


def run_evaluation():
    print("=" * 70)
    print(" VAD 模型评估")
    print("=" * 70)

    config = TrainingConfig()
    model_path = os.path.join(config.checkpoint_dir, "ppo_vad_final.zip")
    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("\n可用的模型:")
        if os.path.exists(config.checkpoint_dir):
            for file in os.listdir(config.checkpoint_dir):
                if file.endswith(".zip"):
                    print(f"  - {file}")
        return

    print(f"\n模型路径: {model_path}")
    print("\n开始评估...")
    evaluate(model_path=model_path, config=config, n_episodes=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAD PPO 训练/评估脚本")
    parser.add_argument("--eval", action="store_true", help="评估模式（而不是训练）")
    args = parser.parse_args()

    if args.eval:
        run_evaluation()
    else:
        main()
