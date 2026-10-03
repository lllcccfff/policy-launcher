# RLframework_AD Evaluation

`eval.py` 通过 gRPC 连接 StreetWorld 仿真服务，运行原生规划模型评测。

## 运行

从 StreetWorld 仓库根目录执行，先启动仿真服务并选定场景：

```bash
python -m metadrive.examples.env_server_easydrive --host 127.0.0.1 --port 50052 --max-workers 1
```

在对应模型的 Python 环境中运行：

```bash
python -m rl_framework.eval --model vad --host 127.0.0.1 --port 50052 --device cuda --timeout 360
```

评测按服务端提供的场景顺序执行；`--max-steps` 限制单场景步数。模型输出交给服务端执行，场景结束后打印步数、累计奖励和终止状态。

支持的 `--model`：

- `uniad`、`vad`、`genad`、`momad`、`stp3`
- `opendrivevla`、`latent_transfuser`、`diffusiondrive`、`sparsedrive`
- `alpamayo1`、`alpamayo1.5`、`autovla`、`epona`
- `openemma_gpt`、`openemma_qwen`、`openemma_llava`、`openemma_llama`

完整参数见 `python -m rl_framework.eval --help`。

## 依赖与目录

各模型需要对应的上游源码、配置、checkpoint 和推理环境；路径由各自的 `native_agent.py` 指定。`requirements.txt` 提供共享依赖，PyTorch、MMCV 等模型依赖按对应模型环境安装。VAD 的已有环境归档可通过 `dockerfile` 构建镜像。OpenDriveVLA 的独立环境与运行说明见 [opendrivevla/README.md](opendrivevla/README.md)。

- `eval.py`：命令行、模型选择和 gRPC 评测循环。
- `common/`：观测解包、坐标与轨迹转换、张量设备迁移及模型加载辅助。
- 各模型目录：原生 agent、模型加载器、观测适配及推理扩展。

轨迹控制与场景指标由 StreetWorld 仿真端负责。
