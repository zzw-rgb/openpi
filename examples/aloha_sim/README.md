> 本文为 examples/aloha_sim/README.md 的中文翻译，仅供阅读参考，以英文原文为准。

# 运行 Aloha Sim（仿真）（Run Aloha Sim）

## 使用 Docker（With Docker）

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## 不使用 Docker（Without Docker）

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

注意：如果你看到 EGL 错误，可能需要安装以下依赖：

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

终端窗口 2：

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```
