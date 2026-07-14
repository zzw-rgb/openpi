> 本文为 examples/libero/README.md 的中文翻译，仅供阅读参考，以英文原文为准。

# LIBERO 基准（LIBERO Benchmark）

本示例运行 LIBERO 基准：https://github.com/Lifelong-Robot-Learning/LIBERO

注意：在更新本目录下的 requirements.txt 时，需要在 `uv pip compile` 命令中添加一个额外的标志 `--extra-index-url https://download.pytorch.org/whl/cu113`。

本示例需要初始化 git 子模块（submodules）。别忘了运行：

```bash
git submodule update --init --recursive
```

## 使用 Docker（推荐）（With Docker (recommended)）

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

你可以通过提供额外的 `SERVER_ARGS`（参见 `scripts/serve_policy.py`）来自定义加载的检查点，通过提供额外的 `CLIENT_ARGS`（参见 `examples/libero/main.py`）来自定义 LIBERO 任务套件（task suite）。例如：

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## 不使用 Docker（不推荐）（Without Docker (not recommended)）

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

终端窗口 2：

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## 结果（Results）

如果你想复现以下数字，可以评测位于 `gs://openpi-assets/checkpoints/pi05_libero/` 的检查点。该检查点是在 openpi 中用 `pi05_libero` 配置训练的。

| 模型 | Libero Spatial | Libero Object | Libero Goal | Libero 10 | 平均 |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
