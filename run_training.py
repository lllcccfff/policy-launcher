#!/usr/bin/env python3
from posix import environ
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

from rl_framework.train_ppo import train, evaluate, TrainingConfig
from metadrive.envs.scenario_env import ScenarioEnv
from easydrive.models.scenes.simulator_interface import SimulatorInterface
# from lib.integration.simulator_interface import SimulatorInterface
import os

_TEMP_SCENE_DIRS = []

def _cleanup_temp_scene_dirs():
    for d in _TEMP_SCENE_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_SCENE_DIRS.clear()

atexit.register(_cleanup_temp_scene_dirs)

def create_base_env():
    scene_path = os.environ.get("SCENE_CONFIG_DIR", "./configs/exps/sim/scene_train")
    scene_dir = scene_path
    if os.path.isfile(scene_path):
        tmp_dir = tempfile.mkdtemp(prefix="scene_cfg_")
        shutil.copy(scene_path, os.path.join(tmp_dir, os.path.basename(scene_path)))
        _TEMP_SCENE_DIRS.append(tmp_dir)
        scene_dir = tmp_dir

    env_cfg = {
        "scene_config_directory": scene_dir,
    }
    additional_cfg = {
        'image_observation': True,
    }
    env_cfg.update(additional_cfg)

    model = SimulatorInterface()
    env = ScenarioEnv(model, env_cfg)
    
    return env

def main():
    print("="*70)
    print(" PPO training launch script")
    print("="*70)

    config = TrainingConfig()
    
    # PPO setting
    config.total_timesteps = 100_000      
    config.save_freq = 25_000             
    config.eval_freq = 20_000             
    config.eval_episodes = 3             

    config.learning_rate = 1e-5      

    config.device = 'cuda:0'            

    config.with_adapter = True 
    config.pretrained_planning_weights = os.path.join(project_root, "rl_framework/actor_ckpts/planning_modules.pth")
    config.normalize_obs = False
    config.normalize_reward = True
    config.training_phase = int(os.environ.get("UNIAD_TRAIN_PHASE", 0))
    config.apply_phase_defaults()

    print("\nConfiguration information:")
    print(f"  Device: {config.device}")
    print(f"  Training Phase: {config.training_phase}")
    print(f"  Total steps: {config.total_timesteps:,}")
    schedule_str = ", ".join(f"phase{p}:{ratio:.2f}" for p, ratio in config.phase_schedule)
    print(f"  Phase schedule: {schedule_str}")
    print(f"  Rollout steps: {config.n_steps}, Batch size: {config.batch_size}, Epochs: {config.n_epochs}")
    print(f"  Clip range: {config.clip_range}, Target KL: {config.target_kl}")
    print(f"  Actor LR: {config.actor_lr}, Critic LR: {config.critic_lr}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Tensorboard: {config.tensorboard_log}")

    base_env_fn = create_base_env

    print("\nCreating simulation environment...")

    eval_env_fn = base_env_fn

    print("\n" + "="*70)
    print(" Start training")
    print("="*70)
    print("\nTips:")
    print("  - Use Ctrl+C to safely interrupt training")
    print("  - The training process will automatically save checkpoints")
    print("  - You can use Tensorboard to monitor the training progress:")
    print(f"    tensorboard --logdir={config.tensorboard_log}")
    print("")

    # model_path = os.path.join(config.checkpoint_dir, "cirr4_train_model.zip")
    # print(f"\n模型路径: {model_path}")
    model_path = None
    
    try:
        train(base_env_fn, config, eval_env_fn=eval_env_fn, model_path=model_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print(" Training completed")
    print("="*70)

def run_evaluation():
    print("="*70)
    print(" 模型评估")
    print("="*70)

    config = TrainingConfig()

    model_path = os.path.join(
        config.checkpoint_dir, 
        "best_model/best_model.zip"
    )

    model_path = os.path.join(config.checkpoint_dir, "cirr4_train_model.zip")
    # model_path = os.path.join(config.checkpoint_dir, "ppo_uniad_100000_steps.zip")
    
    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("\n可用的模型:")
        if os.path.exists(config.checkpoint_dir):
            for file in os.listdir(config.checkpoint_dir):
                if file.endswith('.zip'):
                    print(f"  - {file}")
        return
    
    print(f"\n模型路径: {model_path}")

    print("\n开始评估...")
    evaluate(model_path, create_base_env, config, n_episodes=1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO 训练/评估脚本')
    parser.add_argument(
        '--eval', 
        action='store_true', 
        help='评估模式（而不是训练）'
    )
    
    args = parser.parse_args()
    
    if args.eval:
        run_evaluation()
    else:
        main()
