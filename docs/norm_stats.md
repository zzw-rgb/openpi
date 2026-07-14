> 本文为 docs/norm_stats.md 的中文翻译，仅供阅读参考，以英文原文为准。

# 归一化统计量（Normalization statistics）

遵循常见做法，我们的模型在策略训练和推理期间会对本体感受状态（proprioceptive state）输入和动作目标（action targets）进行归一化（normalization）。用于归一化的统计量是在训练数据上计算得到，并与模型检查点（checkpoint）一同存储的。

## 重新加载归一化统计量（Reloading normalization statistics）

当你在一个新数据集上微调（fine-tune）我们的某个模型时，你需要决定是（A）复用现有的归一化统计量，还是（B）在你的新训练数据上计算新的统计量。哪种方式更适合你，取决于你的机器人与任务同预训练数据集中机器人与任务分布的相似程度。下面，我们列出了每个模型所有可用的预训练归一化统计量。

**如果你的目标机器人与这些预训练统计量之一相匹配，可以考虑重新加载相同的归一化统计量。** 通过重新加载归一化统计量，你数据集中的动作对模型来说会更加「熟悉」，从而可能带来更好的性能。你可以通过在训练配置中添加一个 `AssetsConfig` 来重新加载归一化统计量，让它指向对应的检查点目录和归一化统计量 ID，如下所示，这里以 `pi0_base` 检查点的 `Trossen`（即 ALOHA）机器人统计量为例：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

关于重新加载归一化统计量的完整训练配置示例，参见[训练配置文件](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)中的 `pi0_aloha_pen_uncap` 配置。

**注意：** 要成功重新加载归一化统计量，重要的一点是你的机器人 + 数据集需遵循预训练中使用的动作空间（action space）定义。我们在下面提供了对动作空间定义的详细描述。

**注意 #2：** 重新加载归一化统计量是否有益，取决于你的机器人与任务同预训练数据集中机器人与任务分布的相似程度。我们建议始终两种方式都尝试——重新加载，以及用在你新数据集上重新计算的一套统计量进行训练（关于如何计算新统计量的说明，参见[主 README](../README.md)）——然后选出对你的任务更有效的一种。


## 提供的预训练归一化统计量（Provided Pre-training Normalization Statistics）

下面列出了我们提供的所有预训练归一化统计量。我们为 `pi0_base` 和 `pi0_fast_base` 两个模型都提供了这些统计量。对于 `pi0_base`，将 `assets_dir` 设为 `gs://openpi-assets/checkpoints/pi0_base/assets`；对于 `pi0_fast_base`，将 `assets_dir` 设为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。

| 机器人 | 描述 | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF 双臂机器人，配平行夹爪（parallel grippers） | trossen |
| Mobile ALOHA | 安装在 Slate 底盘上的移动版 ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于 DROID 配置的 7-DoF 机械臂，配平行夹爪 | droid |
| Franka Emika (non-DROID) | Franka FR3 机械臂，配 Robotiq 2F-85 夹爪 | franka |
| UR5e | 6-DoF UR5e 机械臂，配 Robotiq 2F-85 夹爪 | ur5e |
| UR5e bi-manual | 双臂 UR5e 配置，配 Robotiq 2F-85 夹爪 | ur5e_dual |
| ARX | 双臂 ARX-5 机械臂配置，配平行夹爪 | arx |
| ARX mobile | 安装在 Slate 底盘上的移动版双臂 ARX-5 机械臂配置 | arx_mobile |
| Fibocom mobile | Fibocom 移动机器人，配 2x ARX-5 机械臂 | fibocom_mobile |


## Pi0 模型动作空间定义（Pi0 Model Action Space Definitions）

开箱即用时，`pi0_base` 和 `pi0_fast_base` 都使用以下动作空间定义（左、右是从机器人背后朝向工作区看时定义的）：
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

本体感受状态使用与动作空间相同的定义，唯一的例外是移动机器人的底盘 x-y 位置（最后两个维度），我们不将其包含在本体感受状态中。

对于 7-DoF 机器人（例如 Franka），我们使用动作空间的前 7 个维度作为关节动作，第 8 个维度作为夹爪动作。

Pi 机器人的通用信息：
- 关节角以弧度（radians）表示，零位对应各机器人接口库所报告的零位置，唯一的例外是 ALOHA——标准的 ALOHA 代码使用了略有不同的约定（详见 [ALOHA 示例代码](../examples/aloha_real/README.md)）。
- 夹爪位置在 [0.0, 1.0] 范围内，0.0 对应完全张开，1.0 对应完全闭合。
- 控制频率：UR5e 和 Franka 为 20 Hz，ARX 和 Trossen（ALOHA）机械臂为 50 Hz。

对于 DROID，我们使用原始的 DROID 动作配置：前 7 个维度为关节速度（joint velocity）动作，第 8 个维度为夹爪动作，控制频率为 15 Hz。
