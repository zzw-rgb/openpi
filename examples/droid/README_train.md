> 本文为 examples/droid/README_train.md 的中文翻译，仅供阅读参考，以英文原文为准。

# 在 DROID 上训练（Training on DROID）

这里我们介绍如何在*完整* DROID 数据集上微调（fine-tune）pi0.5 模型。这是 pi05-DROID 训练流程的一个近似开源复现（在数据加载和所用动作空间上有细微差异）——关于如何用在 DROID 平台上采集的较小自定义数据集来微调你的模型的教程，见下文。

与 openpi 其余部分使用 LeRobot 进行数据加载不同，完整 DROID 训练需要使用 RLDS 作为数据格式（因为目前 LeRobot 对于 DROID 这样更大的数据集尚不够可扩展——不过他们正在改进）。下面，我们提供了为 RLDS 数据加载更新 openpi 环境的说明，以及在哪里下载 DROID 数据集。

## 安装（Install）

RLDS 数据加载需要一些额外的依赖。RLDS 依赖组依赖于 `tensorflow-cpu==2.15.0`，它只为 **Python 3.11** 提供 wheel 包。在 sync 之前，请确保用 Python 3.11 创建你的虚拟环境：

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --group rlds
```

## 下载 DROID 数据集（Download DROID dataset）

你可以用以下命令下载 DROID 数据集（在安装 `gsutil` google cloud CLI 之后）：
```
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1
```

注意，下载 1.0.1 版本很重要（而不是 v1.0.0）：它包含完整的语言标注集（约 75k 个 episode），而 v1.0.0 只有 30k 个 episode 的标注。如果出于某种原因你想使用其他版本，可在[此处](src/openpi/training/droid_rlds_dataset.py)的 `DroidRldsDataset` 对象中修改 `version="1.0.1"` 这一行。

下载 DROID RLDS 数据集需要 1.8TB 的磁盘存储空间。

## 运行（Run）

首先，将你 `TrainConfig` 中的 `rlds_data_dir` 路径改为你下载 `droid` 数据集的目录（参见 [src/openpi/training/config.py](src/openpi/training/config.py)）。

然后，计算归一化统计量（normalization statistics，这将花费约 10 分钟）：
```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi05_full_droid_finetune --max-frames 10_000_000
```

运行训练：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_full_droid_finetune --exp-name=my_experiment --overwrite
```

**注意**：原始的 pi0.5-DROID 模型是用关节速度（joint velocity）动作训练的。关节速度动作与仿真评测环境不兼容（更难仿真）。因此，我们不建议用关节速度动作训练，而是在这里使用关节位置（joint position）动作。


## 计算资源需求（Compute Requirements）

我们的 DROID 训练配置在 8x H100 GPU 上收敛大约需要 2 天（100k 次迭代，bs256，约 1 个 epoch）。如果你从 PaliGemma 初始化而非 pi0 初始化开始，请预留在 8x H100 上约 5 天的时间（240k 次迭代，即 3 个 epoch）。

我们尝试过用 LoRA 进行更廉价的微调，但到目前为止尚未发现策略表现良好。


## 数据过滤（Data Filtering）

与任何多样化的真实机器人数据集一样，DROID 数据集并非完美「干净」，我们发现数据过滤能显著提升策略性能。具体来说，DROID 数据集包含许多机器人不动的*空闲（idle）*时间步（部分原因是数据采集期间使用的 VR 遥操作 teleoperation 接口，这里我们不展开太多细节）。恰当地过滤这些空闲转移（transitions）可以提升策略性能。

默认情况下，我们的 openpi 训练配方实现了与训练所有 pi-DROID 模型时相同的空闲过滤器（idle filter）。我们通过预先计算训练期间要采样的数据集索引来实现它。你可以查看 [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) 了解我们如何计算这些索引。粗略地说，我们过滤掉那些接下来的动作分块（chunk of actions）大部分为空闲的时间步。在训练期间，我们的代码会自动从云存储拉取我们预先计算好的索引列表并应用它们。如果你想修改空闲过滤器/创建自定义采样逻辑，可以修改我们的脚本来生成新的索引列表，并通过 [src/openpi/training/config.py](src/openpi/training/config.py) 中的 `filter_dict_path="<path_to_filter_dict>"` 参数提供它。

**注意**：我们的过滤索引列表仅对上面下载部分提到的 `droid/1.0.1` 数据集有效，对任何其他版本的 DROID 数据集都不会提供有效的过滤，所以请确保你下载了上面的数据集！如果你有自定义的 DROID 版本，可以重新运行 [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) 脚本来生成新的采样索引列表。

## RoboArena

欢迎考虑将你的 DROID 策略提交到 [RoboArena 基准](https://robo-arena.github.io/)，它允许你在多样化的任务和场景上、**在真实世界中**评测你的策略！:)

如果你对 RoboArena 有疑问，请发邮件至 [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com)。


# 在自定义 DROID 数据集上微调（Fine-Tuning on Custom DROID Datasets）

这里我们介绍如何在一个自定义（较小）的、于 DROID 平台上采集的数据集上微调模型。与其他数据集一样，我们将首先把自定义 DROID 数据集转换为 LeRobot，然后在其上微调一个模型（pi05-droid）。

注意：我们在这里使用 LeRobot，因为我们假设自定义 DROID 微调数据集相对较小（小于数十小时）。对于更大的数据集（如完整的 DROID 数据集），我们建议使用 RLDS，因为它效率更好（参见上面的示例）。


## 第 1 步：将你的自定义 DROID 数据集转换为 LeRobot（Converting your custom DROID dataset to LeRobot）

本示例中，我们将使用真实 DROID 数据集的一个小子集。这是一个仅有 30 个演示（demonstrations）的子集——我们假设你会用自己的数据集代替，但这里给出下载我们子集（1.6GB）的命令：
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 <your_target_path>
```

我们还会下载 DROID 数据集的语言标注，以便将演示与语言指令配对。同样，对于你自己的数据，你可以手动输入语言指令，无需下载我们的标注。要下载 DROID 语言标注（12MB），运行：
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json <your_target_dir>
```

对于你自己的数据集，确保每个 episode 的目录包含一个名为 `recordings/MP4` 的文件夹——如果没有，你需要先使用[此处](https://github.com/droid-dataset/droid/blob/main/scripts/convert/svo_to_mp4.py)的脚本运行 MP4 视频提取（从 SVO 文件）。

现在，我们将使用 `convert_droid_to_lerobot.py` 脚本来创建该数据集的 LeRobot 版本（30 个演示耗时小于 5 分钟）：
```
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir <your_target_path>
```

## 第 2 步：用你的自定义数据集运行微调（Run fine-tuning with your custom dataset）

现在我们可以用转换后的自定义数据集运行微调。我们提供了一个在我们创建的自定义数据集上微调 `pi05_droid` 的示例配置。你可以轻松修改该配置以适配其他基座模型，或在 `config.py` 中使用你的自定义 DROID 数据集（搜索 `pi05_droid_finetune`）。

启动训练：
```
uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite
```

训练完成后，你可以按照 [`examples/droid/README.md`](examples/droid/README.md) 中的说明来提供策略服务并在机器人上运行它。
