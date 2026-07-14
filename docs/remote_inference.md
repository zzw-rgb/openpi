> 本文为 docs/remote_inference.md 的中文翻译，仅供阅读参考，以英文原文为准。

# 远程运行 openpi 模型（Running openpi models remotely）

我们提供了远程运行 openpi 模型的工具。这对于在机器人之外、更强大的 GPU 上运行推理（inference）很有用，同时也有助于让机器人环境与策略（policy）环境保持分离（从而例如避免与机器人软件之间的依赖地狱 dependency hell）。

## 启动远程策略服务器（Starting a remote policy server）

要启动一个远程策略服务器，你只需运行以下命令：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

`env` 参数指定应加载哪个 $\pi_0$ 检查点（checkpoint）。在底层，该脚本会执行类似下面的命令，你可以用它来启动一个策略服务器，例如用于你自己训练的检查点（这里以 DROID 环境为例）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

这会启动一个策略服务器，提供由 `config` 和 `dir` 参数指定的策略。该策略将在指定端口（默认：8000）上提供服务。

## 从你的机器人代码查询远程策略服务器（Querying the remote policy server from your robot code）

我们提供了一个依赖极少的客户端工具，你可以轻松地将其嵌入任何机器人代码库。

首先，在你的机器人环境中安装 `openpi-client` 包：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

然后，你就可以使用该客户端从机器人代码中查询远程策略服务器。下面是一个如何操作的示例：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

这里，`host` 和 `port` 参数指定了远程策略服务器的 IP 地址和端口。你也可以把它们指定为机器人代码的命令行参数，或在机器人代码库中硬编码。`observation` 是一个包含观测和提示（prompt）的字典，遵循你所提供服务的策略的策略输入规范。我们在[simple client 示例](../examples/simple_client/main.py)中提供了针对不同环境如何构造这个字典的具体示例。
