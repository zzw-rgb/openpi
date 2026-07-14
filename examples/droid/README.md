> 本文为 examples/droid/README.md 的中文翻译，仅供阅读参考，以英文原文为准。

# openpi 中的 DROID 策略（DROID Policies in openpi）

我们提供以下方面的说明：
- [运行我们最好的 $\pi_{0.5}$-DROID 策略的推理](./README.md#running-droid-inference)
- [运行其他预训练 DROID 策略（$\pi_0$、$\pi_0$-FAST 等）的推理](./README.md#running-roboarena-baseline-policies)
- [在*完整* DROID 数据集上预训练*通用（generalist）*策略](./README_train.md#training-on-droid)
- [在你自定义的 DROID 数据集上微调专家（expert）$\pi_{0.5}$](./README_train.md#fine-tuning-on-custom-droid-datasets)

## 运行 DROID 推理（Running DROID Inference）

本示例展示如何在 [DROID 机器人平台](https://github.com/droid-dataset/droid)上运行微调后的 $\pi_{0.5}$-DROID 模型。基于[公开的 RoboArena 基准](https://robo-arena.github.io/leaderboard)，这是我们目前最强的通用 DROID 策略。


### 第 1 步：启动策略服务器（Start a policy server）

由于 DROID 控制笔记本没有强大的 GPU，我们将在另一台配备更强 GPU 的机器上启动一个远程策略服务器，然后在推理期间从 DROID 控制笔记本查询它。

1. 在一台配备强大 GPU（约 NVIDIA 4090）的机器上，按照 [README](https://github.com/Physical-Intelligence/openpi) 中的说明克隆并安装 `openpi` 仓库。
2. 通过以下命令启动 OpenPI 服务器：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

你也可以运行下面等效的命令：

```bash
uv run scripts/serve_policy.py --env=DROID
```

### 第 2 步：运行 DROID 机器人（Run the DROID robot）

1. 确保你在 DROID 控制笔记本和 NUC 上都安装了最新版本的 DROID 包。
2. 在控制笔记本上，激活你的 DROID conda 环境。
3. 克隆 openpi 仓库并安装 openpi 客户端（我们用它来连接策略服务器；它依赖极少，安装应当很快）：在激活 DROID conda 环境的状态下，运行 `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`。
4. 安装 `tyro`，我们用它来做命令行解析：`pip install tyro`。
5. 将本目录中的 `main.py` 文件复制到 `$DROID_ROOT/scripts` 目录。
6. 将 `main.py` 文件中的相机 ID 替换为你自己相机的 ID（你可以通过在命令行运行 `ZED_Explorer` 来查找相机 ID，它会打开一个工具，显示所有已连接的相机及其 ID——你也可以用它来确认相机位置摆放得当，能够看到你希望机器人交互的场景）。
7. 运行 `main.py` 文件。确保将 IP 和主机地址指向策略服务器。（要确认从 DROID 笔记本可以访问到服务器机器，你可以在 DROID 笔记本上运行 `ping <server_ip>`。）另外，确保指定用于策略的外部相机（我们只输入一个外部相机），从 ["left", "right"] 中选择。

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

该脚本会要求你输入一条自由形式的语言指令，供机器人遵循。确保将相机对准你希望机器人交互的场景。你*不需要*精细地控制相机角度、物体位置等。根据我们的经验，该策略相当鲁棒。祝你玩得愉快！

## 疑难排查（Troubleshooting）

| 问题 | 解决方案 |
|-------|----------|
| 无法访问策略服务器 | 确保服务器正在运行，且 IP 和端口正确。你可以在 DROID 笔记本上运行 `ping <server_ip>` 来检查服务器机器是否可达。 |
| 找不到相机 | 确保相机 ID 正确，且相机已连接到 DROID 笔记本。有时重新插拔相机会有帮助。你可以在命令行运行 `ZED_Explore` 来检查所有已连接的相机。 |
| 策略推理慢/不稳定 | 尝试为 DROID 笔记本使用有线网络连接以降低延迟（每个分块 0.5 - 1 秒的延迟是正常的）。 |
| 策略未能很好地完成任务 | 在我们的实验中，该策略能够在广泛的环境、相机位置和光照条件下执行简单的桌面操作任务（抓取-放置 pick-and-place）。如果策略未能很好地完成任务，你可以尝试修改场景或物体摆放以降低任务难度。同时确保你传给策略的相机视角能看到场景中所有相关物体（该策略仅以单个外部相机 + 腕部相机为条件，确保你把想要的相机馈送给了策略）。使用 `ZED_Explore` 来检查你传给策略的相机视角能否看到场景中所有相关物体。最后，该策略远非完美，在更复杂的操作任务上会失败，但通常会做出不错的尝试。:) |


## 运行其他策略（Running Other Policies）

我们提供了运行 [RoboArena](https://robo-arena.github.io/) 论文中基线（baseline）DROID 策略的配置。只需运行下面的命令即可为相应策略启动推理服务器。然后按照上面的说明在 DROID 机器人上运行评测。

```
# Train from pi0-FAST, using FAST tokenizer
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# Train from pi0, using flow matching
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid

# Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid

# Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid

# Trained from PaliGemma, using FSQ tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid

# pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

你可以在 [roboarena_config.py](../../src/openpi/training/misc/roboarena_config.py) 中找到这些推理配置。
