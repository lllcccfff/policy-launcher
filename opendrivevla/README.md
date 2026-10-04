# OpenDriveVLA eval

模型运行在 `opendrivevla` Conda 环境，仿真服务运行在已有的 `3dv` 环境，通过 gRPC 连接。以下命令均从 StreetWorld 仓库根目录执行。

默认权重：`OpenDriveVLA/checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct`。

## 创建模型环境

本机使用 Python 3.10、PyTorch 2.1.2 / CUDA 12.1、系统 CUDA Toolkit 12.1 和 GCC 11。MMCV 与 MMDetection3D 使用 OpenDriveVLA 自带的源码。

```bash
conda create -n opendrivevla python=3.10 pip -y
conda activate opendrivevla
python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-compile -r rl_framework/opendrivevla/requirements.txt

export CUDA_HOME=/usr/local/cuda-12.1
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps OpenDriveVLA/third_party/mmcv_1_7_2
python -m pip install --no-build-isolation --no-deps OpenDriveVLA/third_party/mmdetection3d_1_0_0rc6
python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7%2Bcu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip check
```

`TORCH_CUDA_ARCH_LIST=8.9` 对应本机 RTX 4090。FlashAttention wheel 是官方提供的 Python 3.10 / PyTorch 2.1 / CUDA 12 构建。

## 启动仿真服务

在一个终端中选择需要评测的场景：

```bash
conda activate 3dv
CUDA_VISIBLE_DEVICES=0 python -m streetworld.examples.env_server_easydrive \
  --ad-policy-config opendrivevla --host 127.0.0.1 --port 50053 \
  --web-port 18083 --max-workers 1 \
  --video-output-dir outputs_benchmark/opendrivevla/videos
```

也可以通过 `streetworld.examples.env_server_scene_config` 的 `--scene-config` 参数指定每行一个场景编号的文本文件，并传入 `--dataset nuscenes`。

## 运行 eval

在另一个终端中：

```bash
conda activate opendrivevla
CUDA_VISIBLE_DEVICES=0 python -m rl_framework.eval \
  --model opendrivevla --host 127.0.0.1 --port 50053 --device cuda --timeout 360
```

eval 读取服务端的场景顺序。模型输出 6 个、间隔 0.5 秒的轨迹点，由仿真端的 iLQR 控制器执行。

## 本机验证

2026-10-03 已完成：

- `pip check` 无依赖冲突；CUDA 张量计算和 FlashAttention 前向计算通过。
- 默认 0.5B checkpoint 与 UniAD 视觉模块成功加载。
- 上述 eval 入口连接 `3dv` 仿真服务，在 nuScenes 场景 `0001` 完成 7 步闭环执行，收到 `terminated=True` 后以退出码 0 结束。
- 联调日志位于 `outputs_benchmark/opendrivevla_smoke/eval.log`，视频位于该目录的 `videos/`。此验证确认运行链路可用，不代表驾驶指标达标。

本机场景首次 Reset 需要读取 NPZ 并解码 PLY，可能耗时数分钟；`--timeout 360` 对 Reset 提供 720 秒等待时间。
