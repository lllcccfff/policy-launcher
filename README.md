# RLframework_AD

This directory contains the reinforcement-learning stack that fine-tunes UniAD inside the GaussianDrive/EasyDrive simulator. It covers:

- launch scripts (`run_training.py`, `train_ppo.py`) that orchestrate env creation, phased PPO training, checkpointing, and evaluation
- integration glue so UniAD can ingest simulator observations (`dataparser`, `env_wrapper`, `closeLoop`) and feed RL actions back through an iLQR controller
- custom policy definitions with LoRA adapters and auxiliary regularization (KL/BC) plus metric tracking for NC/DAC/TTC/COM/RC

The code assumes the repository root still has `UniAD_SIM/` and `simulator/` so UniAD configs/checkpoints and MetaDrive scenes remain accessible.

---

## Layout

```
rl_framework/
├── run_training.py       # CLI: builds ScenarioEnv factory, prints config, starts train/eval
├── train_ppo.py          # UniADPPO trainer, TrainingConfig, env wrappers, callbacks, evaluation
├── ilqr/
│   ├── lqr.py            # plan2control: converts planned trajectories to throttle/steer via iLQR
│   ├── lqr_solver.py     # solver implementation + parameters
│   └── utils.py
└── rl_modules/
    ├── closeLoop.py      # create_uniad: load UniAD config/ckpt, clean MMCV registries, wrap in MMDP
    ├── dataparser.py     # parse_raw: resize/normalize camera stacks, build img_metas + can-bus
    ├── env_wrapper.py    # UniADWrapper: pipeline obs→UniAD→plan_query/bev, action→control
    ├── policy_network.py # CustomUniADPolicy: LoRA adapters + actor/critic heads for plan_query/bev
    ├── metrics.py        # EpisodeMetricTracker: NC/DAC/TTC/COM/RC aggregation
    └── ...
```

### Module responsibilities

| Component | File | Summary |
| --- | --- | --- |
| **Launcher** | `run_training.py` | Provides `create_base_env()` (reads `SCENE_CONFIG_DIR`, enables `image_observation`), logs the chosen `TrainingConfig`, and calls `train()` or `run_evaluation()` based on CLI flags. |
| **Trainer** | `train_ppo.py` | Holds `TrainingConfig` (paths, hyper-parameters, phase presets), `UniADPPO` (SB3 PPO subclass with UniAD-specific regularization), and helpers (`create_env`, `create_ppo_model`, `setup_callbacks`, `train`, `evaluate`). |
| **UniAD loader** | `rl_modules/closeLoop.py` | `create_uniad()` resolves config/checkpoint paths within `UniAD_SIM`, loads plugins, fuses Conv+BN, cleans Dice loss registries, and returns an eval-mode `MMDataParallel` UniAD model on the requested GPU. |
| **Dataparser** | `rl_modules/dataparser.py` | `parse_raw()` resizes camera stacks, applies mmcv normalization, computes intrinsics/extrinsics/can-bus, and packages tensors that UniAD expects (`img`, `img_metas`, `command`, pose info). |
| **Env wrapper** | `rl_modules/env_wrapper.py` | `UniADWrapper` intercepts MetaDrive observations, calls `parse_raw` + UniAD forward to produce `plan_query`/`bev_feat` for PPO, converts PPO deltas to trajectories, and feeds them through iLQR to obtain simulator throttle/steer commands; also enriches `info` for metrics. |
| **Policy** | `rl_modules/policy_network.py` | `CustomUniADPolicy` wires `plan_query`/`bev_feat` into attention adapters, LoRA-augmented transformer decoders, and actor/critic MLPs, exposing knobs like adapter toggles, log-std bounds, and KL/BC regularization per training phase. |
| **Controller** | `ilqr/lqr.py` | `plan2control()` takes a smoothed trajectory from PPO + UniAD and computes throttle/brake + steering outputs respecting physical limits (wheelbase, steering rate, acceleration bounds). |
| **Metrics** | `rl_modules/metrics.py` | `EpisodeMetricTracker` collects NC (no collision), DAC, TTC, COM, RC from env infos, letting evaluation mirror UniAD KPI reporting. |

---

## Training

> **Prerequisites**: install torch, mmcv, mmdet3d, stable-baselines3, gymnasium, MetaDrive, etc.; ensure `UniAD_SIM/projects/configs/...` and `UniAD_SIM/ckpts/...` contain the intended config/checkpoint; simulator scenes should be under `simulator/GaussianDrive/configs/`. Export `PYTHONPATH` or run from repo root so `rl_framework` and simulator packages resolve.

1. (Optional) pick a scene directory or single JSON:
   ```bash
   export SCENE_CONFIG_DIR=simulator/GaussianDrive/configs/exps/sim/scene_train
   ```
2. Launch training (defaults to phase 0 schedule):
   ```bash
   python -m rl_framework.run_training
   ```
   `run_training.main()` creates a `TrainingConfig`, prints its settings (devices, timesteps, clip ranges, phase schedule), builds `base_env_fn = create_base_env`, and calls `train(base_env_fn, config, eval_env_fn)`.
3. `train()` (see `train_ppo.py`) performs:
   - load UniAD via `create_uniad()` with `config.uniad_config_path`/`uniad_checkpoint_path`
   - wrap the simulator with `UniADWrapper(parse_raw, traj2control)` so PPO observes `plan_query`/`bev_feat`
   - instantiate `UniADPPO(CustomUniADPolicy, env, policy_kwargs)`; optionally load `config.pretrained_planning_weights`
   - register callbacks (`CheckpointCallback`, `EvalCallback`, `TensorboardCallback`)
   - iterate over `config.phase_schedule`, applying preset hyper-parameters via `_apply_model_phase_settings()` before each `model.learn()`
   - save final model under `config.checkpoint_dir` and VecNormalize stats if used

Artifacts land in `rl_outputs/` by default (`logs/`, `checkpoints_critic_warmup/`, `tensorboard/`, `vec_normalize.pkl`). Adjustable knobs inside `TrainingConfig` include devices, total timesteps, save/eval frequency, adapter toggles, LoRA ranks, and phase-specific LR/clip/regularization settings.

Monitor the run with:
```bash
tensorboard --logdir rl_outputs/tensorboard
```

Interruptions (`Ctrl+C`) trigger a checkpoint save (`interrupted_model.zip`) so training can resume later (pass the zip path to `train(..., model_path=...)`).

---

## Evaluation

```bash
python -m rl_framework.run_training --eval
```

`run_training.run_evaluation()` rebuilds `TrainingConfig`, locates a checkpoint (defaults to `rl_outputs/checkpoints_critic_warmup/cirr4_train_model.zip`—edit if needed), loads UniAD + the wrapped env, restores PPO weights, and calls `evaluate(model_path, create_base_env, config, n_episodes)`. During evaluation:

- `EpisodeMetricTracker` aggregates NC/DAC/TTC/COM/RC for each episode and prints per-episode + mean summaries.
- Results can be redirected to `config.eval_results_path` or parsed from stdout for further analysis.

To evaluate a different model, set `config.checkpoint_dir`/`model_path` before running or provide your own CLI wrapper that points to the desired `.zip`.

---

## Tips & troubleshooting

- **Custom scenes**: `SCENE_CONFIG_DIR` can point to a directory of scenario configs or a single file; when it’s a file, `create_base_env` copies it into a temp folder so MetaDrive accepts it.
- **Devices**: change `config.uniad_device` / `config.policy_device` if you only have one GPU or need CPU mode.
- **Phase tweaks**: edit `TrainingConfig.phase_presets` or `phase_schedule` to run multi-stage fine-tuning (e.g., warm-up critic-only, then unfreeze adapters).
- **Dependency conflicts**: `create_uniad` removes stale `DiceCost/DiceLoss` entries from MMCV registries; if loading still fails, check mmcv/mmdet3d versions.
- **OOM**: reduce `config.n_steps`, `batch_size`, or BEV dimensions (`config.bev_h/w`) and rerun.
