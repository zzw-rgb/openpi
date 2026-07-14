> 本文为 examples/aloha_real/README.md 的中文翻译，仅供阅读参考，以英文原文为准。

# 运行 Aloha（真机）（Run Aloha (Real Robot)）

本示例演示如何使用 [ALOHA 配置](https://github.com/tonyzhaozh/aloha)的真实机器人运行。关于如何加载检查点（checkpoint）并运行推理（inference）的说明，参见[此处](../../docs/remote_inference.md)。我们在下面列出了每个所提供微调模型的相关检查点路径。

## 前置条件（Prerequisites）

本仓库使用了 ALOHA 仓库的一个 fork，仅做了极小的修改以使用 Realsense 相机。

1. 遵循 ALOHA 仓库中的[硬件安装说明](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation)。
1. 修改 `third_party/aloha/aloha_scripts/realsense_publisher.py` 文件，使用你自己相机的序列号（serial numbers）。

## 使用 Docker（With Docker）

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## 不使用 Docker（Without Docker）

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.aloha_real.main
```

终端窗口 2：

```bash
roslaunch aloha ros_nodes.launch
```

终端窗口 3：

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## **ALOHA 检查点指南（ALOHA Checkpoint Guide）**


`pi0_base` 模型可以在 ALOHA 平台上零样本（zero shot）用于一个简单任务，此外我们还提供了两个示例微调检查点，「fold the towel（折叠毛巾）」和「open the tupperware and put the food on the plate（打开保鲜盒并把食物放到盘子上）」，它们可以在 ALOHA 上执行更高级的任务。

虽然我们发现这些策略在多个 ALOHA 工作站的未见（unseen）条件下都能工作，但我们在这里提供一些关于如何最好地布置场景以最大化策略成功机会的指引。我们涵盖了这些策略应使用的提示词（prompts）、我们观察到它效果良好的物体，以及具有良好代表性的初始状态分布。零样本运行这些策略仍是一个非常实验性的功能，无法保证它们能在你的机器人上工作。使用 `pi0_base` 的推荐方式是用来自目标机器人的数据进行微调。


---

### **吐司任务（Toast Task）**

该任务涉及机器人从烤面包机中取出两片吐司并放到盘子上。

- **检查点路径**：`gs://openpi-assets/checkpoints/pi0_base`
- **提示词**："take the toast out of the toaster"
- **所需物体**：两片吐司、一个盘子和一台标准烤面包机。
- **物体分布**：
  - 对真吐司和橡胶假吐司都有效
  - 兼容标准的双槽烤面包机
  - 对不同颜色的盘子都有效

### **场景布置指南（Scene Setup Guidelines）**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- 烤面包机应放置在工作区的左上象限。
- 两片吐司都应从烤面包机内部开始，且至少有 1 cm 的面包从顶部露出。
- 盘子应大致放置在工作区的下方中央。
- 对自然光和人造光都有效，但避免让场景太暗（例如，不要把设置放在封闭空间内或窗帘下）。


### **毛巾任务（Towel Task）**

该任务涉及将一条小毛巾（例如，大致为手巾大小）折叠成八分之一。

- **检查点路径**：`gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **提示词**："fold the towel"
- **物体分布**：
  - 对不同纯色的毛巾都有效
  - 对纹理浓重或带条纹的毛巾表现较差

### **场景布置指南（Scene Setup Guidelines）**
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- 毛巾应铺平并大致放在桌子中央。
- 选择一条不会与桌面融为一体的毛巾。


### **保鲜盒任务（Tupperware Task）**

该任务涉及打开一个装有食物的保鲜盒（tupperware）并将其内容倒到盘子上。

- **检查点路径**：`gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **提示词**："open the tupperware and put the food on the plate"
- **所需物体**：保鲜盒、食物（或类食物物品）和一个盘子。
- **物体分布**：
  - 对各种假食物有效（例如假鸡块、薯条和炸鸡）。
  - 兼容不同盖子颜色和形状的保鲜盒，在带有角翻盖（corner flap）的方形保鲜盒上表现最佳（见下图）。
  - 该策略见过不同纯色的盘子。

### **场景布置指南（Scene Setup Guidelines）**
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- 当保鲜盒和盘子都大致居中放在工作区时观察到最佳性能。
- 位置摆放：
  - 保鲜盒应在左侧。
  - 盘子应在右侧或底部。
  - 保鲜盒的翻盖应朝向盘子。

## 在你自己的 Aloha 数据集上训练（Training on your own Aloha dataset）

1. 将数据集转换为 LeRobot 数据集 v2.0 格式。

    我们提供了一个脚本 [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py)，可将数据集转换为 LeRobot 数据集 v2.0 格式。作为示例，我们已将来自 [BiPlay 仓库](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw)的 `aloha_pen_uncap_diverse_raw` 数据集转换，并上传到 HuggingFace Hub，即 [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse)。


2. 定义一个使用自定义数据集的训练配置。

    我们提供了 [pi0_aloha_pen_uncap 配置](../../src/openpi/training/config.py)作为示例。关于如何用新配置运行训练，你应参考根目录的 [README](../../README.md)。

重要：我们的基座检查点包含来自各种常见机器人配置的归一化统计量（normalization stats）。当你用来自这些配置之一的自定义数据集微调基座检查点时，我们建议使用基座检查点中提供的相应归一化统计量。在本示例中，这是通过在 `AssetsConfig` 中指定 trossen 的 asset_id 以及指向预训练检查点资产目录（asset directory）的路径来实现的。
