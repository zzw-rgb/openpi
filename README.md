> 本文为 README.md 的中文翻译，仅供阅读参考，以英文原文为准。

# openpi

openpi 收录了由 [Physical Intelligence 团队](https://www.physicalintelligence.company/)发布的面向机器人的开源模型与工具包。

目前，本仓库包含三类模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)，一个基于流（flow-based）的视觉-语言-动作模型（vision-language-action model，VLA）。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)，一个基于 FAST 动作 tokenizer 的自回归（autoregressive）VLA。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)，π₀ 的升级版本，具备更强的开放世界泛化能力，采用[知识隔离（knowledge insulation）](https://www.physicalintelligence.company/research/knowledge_insulation)进行训练。注意：在本仓库中，$\pi_{0.5}$ 的训练与推理目前仅支持 flow matching 头（head）。

对于所有模型，我们都提供了在 10k+ 小时机器人数据上预训练（pre-trained）的 _base model_（基座模型）检查点（checkpoint），以及开箱即用或在你自己的数据集上进行微调（fine-tuning）的示例。

这是一个实验性项目：$\pi_0$ 是为我们自己的机器人开发的，这些机器人与广泛使用的平台（如 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/)）有所不同。尽管我们乐观地相信研究者和实践者能够开展富有创造性的新实验，将 $\pi_0$ 适配到自己的平台上，但我们并不指望每一次这样的尝试都会成功。总而言之：$\pi_0$ 可能对你有用，也可能没用，但欢迎你亲自尝试、一探究竟！

## 更新（Updates）

- [Sept 2025] 我们在 openpi 中发布了 PyTorch 支持。
- [Sept 2025] 我们发布了 pi05，π₀ 的升级版本，具备更强的开放世界泛化能力。
- [Sept 2025]：我们为 DROID 训练添加了[改进的空闲过滤器（idle filter）](examples/droid/README_train.md#data-filtering)。
- [Jun 2025]：我们添加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/)上训练 VLA 的[说明](examples/droid/README_train.md)。这是训练 pi0-FAST-DROID 所用训练流程的一个近似开源实现。


## 环境要求（Requirements）

要运行本仓库中的模型，你需要一块 NVIDIA GPU，且至少满足以下规格。以下估计假设使用单块 GPU，但你也可以通过在训练配置中设置 `fsdp_devices` 来使用多块 GPU 进行模型并行，以降低每块 GPU 的显存需求。另请注意，当前的训练脚本尚不支持多节点（multi-node）训练。

| 模式               | 所需显存        | 示例 GPU           |
| ------------------ | --------------- | ------------------ |
| 推理（Inference）  | > 8 GB          | RTX 4090           |
| 微调（LoRA）       | > 22.5 GB       | RTX 4090           |
| 微调（全量 Full）  | > 70 GB         | A100 (80GB) / H100 |

本仓库已在 Ubuntu 22.04 上测试，目前不支持其他操作系统。

## 安装（Installation）

克隆本仓库时，请确保更新子模块（submodules）：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 来管理 Python 依赖。参见 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)进行配置。安装好 uv 后，运行以下命令来配置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：拉取作为依赖的 LeRobot 需要 `GIT_LFS_SKIP_SMUDGE=1`。

**Docker**：作为 uv 安装方式的替代方案，我们也提供了使用 Docker 安装 openpi 的说明。如果你在系统配置上遇到问题，可以考虑使用 Docker 来简化安装。详见 [Docker Setup](docs/docker.md)。




## 模型检查点（Model Checkpoints）

### 基座模型（Base Models）
我们提供了多个基座 VLA 模型检查点。这些检查点已在 10k+ 小时机器人数据上完成预训练，可用于微调。

| 模型         | 用途        | 描述                                                                                                        | 检查点路径                                     |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | 微调        | 用于微调的基座 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)                                  | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | 微调        | 用于微调的基座自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)                  | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | 微调        | 用于微调的基座 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)                               | `gs://openpi-assets/checkpoints/pi05_base`     |

### 微调模型（Fine-Tuned Models）
我们还为各种机器人平台和任务提供了「专家（expert）」检查点。这些模型是从上述基座模型微调而来，旨在直接运行在目标机器人上。它们在你的特定机器人上可能有效，也可能无效。由于这些检查点是在相对较小的数据集上微调的（数据采集自 ALOHA、DROID Franka 等更常见的机器人），它们可能无法泛化到你的特定配置，不过我们发现其中一些（尤其是 DROID 检查点）在实践中泛化得相当广泛。

| 模型                     | 用途          | 描述                                                                                                                                                                                                     | 检查点路径                                            |
| ------------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | 推理          | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 $\pi_0$-FAST 模型：能够在 DROID 机器人平台上、在全新场景中零样本（0-shot）执行多种简单的桌面操作任务                                          | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | 微调          | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 $\pi_0$ 模型：推理速度快于 $\pi_0$-FAST-DROID，但对语言指令的遵循可能不如后者                                                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | 推理          | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 $\pi_0$ 模型：能够在 ALOHA 机器人平台上零样本折叠多种毛巾                                                                               | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | 推理          | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 $\pi_0$ 模型：能够从保鲜盒（tupperware）中取出食物                                                                                      | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | 推理          | 在公开 [ALOHA](https://dit-policy.github.io/) 数据上微调的 $\pi_0$ 模型：能够给笔拔帽                                                                                                                    | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO       | 推理          | 为 [LIBERO](https://libero-project.github.io/datasets) 基准微调的 $\pi_{0.5}$ 模型：取得最先进（state-of-the-art）的性能（参见 [LIBERO README](examples/libero/README.md)）                              | `gs://openpi-assets/checkpoints/pi05_libero`          |
| $\pi_{0.5}$-DROID        | 推理 / 微调   | 在 [DROID 数据集](https://droid-dataset.github.io/)上、采用[知识隔离（knowledge insulation）](https://www.physicalintelligence.company/research/knowledge_insulation)微调的 $\pi_{0.5}$ 模型：推理快、语言遵循好 | `gs://openpi-assets/checkpoints/pi05_droid`           |


默认情况下，检查点会在需要时自动从 `gs://openpi-assets` 下载，并缓存到 `~/.cache/openpi`。你可以通过设置 `OPENPI_DATA_HOME` 环境变量来覆盖下载路径。




## 运行预训练模型的推理（Running Inference for a Pre-Trained Model）

我们的预训练模型检查点只需几行代码即可运行（这里以 $\pi_0$-FAST-DROID 模型为例）：
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
你也可以在[示例 notebook](examples/inference.ipynb) 中试用。

我们为在 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人上运行预训练检查点推理提供了详细的分步示例。

**远程推理（Remote Inference）**：我们提供了**远程**运行模型推理的[示例和代码](docs/remote_inference.md)：模型可以运行在另一台服务器上，并通过 websocket 连接将动作流式传输给机器人。这使得在机器人之外使用更强大的 GPU 变得容易，并让机器人环境与策略（policy）环境保持分离。

**无需机器人测试推理**：我们提供了一个[脚本](examples/simple_client/README.md)，可在没有机器人的情况下测试推理。该脚本会生成一个随机观测（observation）并用模型运行推理。详见[此处](examples/simple_client/README.md)。





## 在你自己的数据上微调基座模型（Fine-Tuning Base Models on Your Own Data）

我们将以在 [LIBERO 数据集](https://libero-project.github.io/datasets)上微调 $\pi_{0.5}$ 模型为贯穿全程的示例，说明如何在你自己的数据上微调基座模型。我们将讲解三个步骤：
1. 将你的数据转换为 LeRobot 数据集（我们用它来训练）
2. 定义训练配置并运行训练
3. 启动一个策略服务器（policy server）并运行推理

### 1. 将你的数据转换为 LeRobot 数据集

我们在 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py) 中提供了一个将 LIBERO 数据转换为 LeRobot 数据集的最小示例脚本。你可以轻松修改它来转换自己的数据！你可以从[此处](https://huggingface.co/datasets/openvla/modified_libero_rlds)下载原始的 LIBERO 数据集，并用以下命令运行脚本：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**注意：** 如果你只是想在 LIBERO 上微调，可以跳过这一步，因为我们的 LIBERO 微调配置已指向一个预先转换好的 LIBERO 数据集。这一步仅仅是一个示例，你可以将其改造用于自己的数据。

### 2. 定义训练配置并运行训练

要在你自己的数据上微调基座模型，你需要定义用于数据处理和训练的配置。我们在下面为 LIBERO 提供了带有详细注释的示例配置，你可以将其修改用于自己的数据集：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义从 LIBERO 环境到模型、以及反向的数据映射。训练和推理都会用到。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义如何处理来自 LeRobot 数据集的原始 LIBERO 数据以用于训练。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参数、数据配置和权重加载器（weight loader）。

我们为在 LIBERO 数据上微调 [π₀](src/openpi/training/config.py)、[π₀-FAST](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 提供了示例微调配置。

在运行训练之前，我们需要计算训练数据的归一化统计量（normalization statistics）。用你的训练配置名称运行下面的脚本：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

现在我们可以用以下命令启动训练（`--overwrite` 标志用于在你以相同配置重新运行微调时覆盖已有的检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

该命令会将训练进度记录到控制台，并将检查点保存到 `checkpoints` 目录。你也可以在 Weights & Biases 仪表盘上监控训练进度。为了最大化利用 GPU 显存，请在运行训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`——这会让 JAX 使用高达 90% 的 GPU 显存（默认值为 75%）。

**注意：** 我们提供了从预训练中*重新加载（reloading）*用于状态/动作归一化的归一化统计量的功能。如果你正在一个曾经属于我们预训练混合数据（pre-training mixture）的机器人上微调一个新任务，这会很有帮助。关于如何重新加载归一化统计量的更多细节，参见 [norm_stats.md](docs/norm_stats.md) 文件。

### 3. 启动策略服务器并运行推理

训练完成后，我们可以通过启动一个策略服务器、然后从 LIBERO 评测脚本中查询它来运行推理。启动一个模型服务器很简单（本例使用第 20,000 次迭代的检查点，请按需修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这会启动一个监听 8000 端口、等待接收观测的服务器。随后我们可以运行一个评测脚本（或机器人运行时 runtime）来查询该服务器。

特别地，对于运行 LIBERO 评测，我们提供（并推荐使用）一套 Docker 化的工作流，它同时处理策略服务器和评测脚本。详见 [LIBERO README](examples/libero/README.md)。

如果你想在自己的机器人运行时中嵌入策略服务器调用，我们在[远程推理文档](docs/remote_inference.md)中提供了一个最小示例。



### 更多示例（More Examples）

我们在以下 README 中提供了更多关于如何在 ALOHA 平台上微调和运行模型推理的示例：
- [ALOHA 模拟器（ALOHA Simulator）](examples/aloha_sim)
- [ALOHA 真机（ALOHA Real）](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持（PyTorch Support）

openpi 现在在原有 JAX 版本之外，提供了 π₀ 和 π₀.₅ 模型的 PyTorch 实现！该 PyTorch 实现已在 LIBERO 基准上验证（推理和微调均已验证）。目前有若干特性尚不支持（未来可能会改变）：

- π₀-FAST 模型
- 混合精度（mixed precision）训练
- FSDP（fully-sharded data parallelism，全分片数据并行）训练
- LoRA（low-rank adaptation，低秩适配）训练
- 训练期间的 EMA（exponential moving average，指数移动平均）权重

### 配置（Setup）
1. 确保你安装了所有依赖的最新版本：`uv sync`

2. 再次确认你安装了 transformers 4.53.2：`uv pip show transformers`

3. 应用 transformers 库的补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

这会用必要的模型改动覆盖 transformers 库中的若干文件：1）支持 AdaRMS，2）正确控制激活值（activations）的精度，3）允许使用 KV cache 而不对其进行更新。

**警告**：在默认的 uv 链接模式（hardlink，硬链接）下，这会永久影响你 uv 缓存中的 transformers 库，意味着这些改动会在 transformers 重新安装后依然存在，甚至可能传播到其他使用 transformers 的项目。要完全撤销此操作，你必须运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch（Converting JAX Models to PyTorch）

要将 JAX 模型检查点转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理（Running Inference with PyTorch）

PyTorch 实现使用与 JAX 版本相同的 API——你只需将检查点路径改为指向转换后的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 的策略服务器（Policy Server with PyTorch）

策略服务器与 PyTorch 模型的工作方式完全相同——只需指向转换后的检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 微调（Finetuning with PyTorch）

要在 PyTorch 中微调模型：

1. 将 JAX 基座模型转换为 PyTorch 格式：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 在你的配置中通过 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径

3. 用以下模式之一启动训练：

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置（Precision Settings）

JAX 和 PyTorch 实现按如下方式处理精度：

**JAX：**
1. 推理：大多数权重和计算使用 bfloat16，少数计算为了稳定性使用 float32
2. 训练：默认使用混合精度：权重和梯度为 float32，（大多数）激活值和计算为 bfloat16。你可以通过在配置中将 `dtype` 设为 float32 来改为全 float32 训练。

**PyTorch：**
1. 推理：与 JAX 一致——大多数权重和计算使用 bfloat16，少数权重为了稳定性转换为 float32
2. 训练：支持全 bfloat16（默认）或全 float32。你可以通过在配置中设置 `pytorch_training_precision` 来更改。bfloat16 占用更少显存，但相比 float32 会表现出更高的 loss。混合精度尚不支持。

在使用 torch.compile 时，JAX 与 PyTorch 之间的推理速度相当。

## 疑难排查（Troubleshooting）

我们会在此收集常见问题及其解决方案。如果你遇到问题，请先查阅此处。如果找不到解决方案，请在仓库中提交 issue（指南参见[此处](CONTRIBUTING.md)）。

| 问题                                        | 解决方案                                                                                                                                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv sync` 因依赖冲突失败                     | 尝试删除虚拟环境目录（`rm -rf .venv`）并再次运行 `uv sync`。如果问题依旧，检查你是否安装了最新版本的 `uv`（`uv self update`）。                                                              |
| 训练时 GPU 显存不足                          | 确保在运行训练前设置了 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高），以允许 JAX 使用更多 GPU 显存。你也可以使用 `--fsdp-devices <n>`（`<n>` 为你的 GPU 数量）来启用[全分片数据并行（fully-sharded data parallelism）](https://engineering.fb.com/2021/07/15/open-source/fsdp/)，它以更慢的训练速度换取更低的显存占用（减速幅度取决于你的具体配置）。如果仍然显存不足，你可能需要考虑禁用 EMA。 |
| 策略服务器连接错误                           | 检查服务器是否正在运行并监听预期端口。验证客户端与服务器之间的网络连通性和防火墙设置。                                                                                                       |
| 训练时缺少 norm stats 错误                   | 在开始训练前，用你的配置名称运行 `scripts/compute_norm_stats.py`。                                                                                                                          |
| 数据集下载失败                               | 检查你的网络连接。对于 HuggingFace 数据集，确保你已登录（`huggingface-cli login`）。                                                                                                        |
| CUDA/GPU 错误                                | 验证 NVIDIA 驱动是否正确安装。对于 Docker，确保已安装 nvidia-container-toolkit。检查 GPU 兼容性。你**不需要**在系统级别安装 CUDA 库——它们会通过 uv 安装。如果你遇到 CUDA 问题，甚至可以尝试*卸载*系统的 CUDA 库，因为系统库有时会导致冲突。 |
| 运行示例时出现 Import 错误                    | 确保你已用 `uv sync` 安装了所有依赖。某些示例可能有额外要求，列在它们各自的 README 中。                                                                                                      |
| 动作维度不匹配                               | 验证你的数据处理变换（transforms）是否与你机器人期望的输入/输出维度匹配。检查你策略类中的动作空间（action space）定义。                                                                      |
| 训练 loss 发散                               | 检查你数据集 `norm_stats.json` 中的 `q01`、`q99` 和 `std` 值。某些很少被使用的维度可能出现非常小的 `q01`、`q99` 或 `std` 值，导致归一化后出现巨大的状态和动作。作为一种变通方案，你可以手动调整 norm stats。 |
