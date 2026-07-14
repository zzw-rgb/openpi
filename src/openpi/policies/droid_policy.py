"""DROID / Franka 平台的输入输出变换（transform）：机器人原始观测 ⇄ 模型统一格式。

DROID 是采自真实 Franka Panda 机械臂的大规模数据集/平台。本文件提供该平台专属的两个
transform，作为 policy_config 里“平台 data_transforms”环节被插入推理数据流：
  - DroidInputs：把机器人端发来的原始观测对齐到模型统一输入。将 7 维关节位置与 1 维夹爪拼成
    8 维 state；把外部第三人称相机与手腕相机图像统一成 uint8 的 HWC 排布；再按模型类型填三路
    图像槽位。要点：DROID 实机只有一路外部相机 + 一路（左）手腕相机，右手腕槽位用零图占位——
    对 π0/π0.5 会把该槽位的 image_mask 置 False（告诉模型这一路是补零、无效），而 π0-FAST 不做
    mask。prompt 若以 bytes 传来会解码成 str；actions 仅训练时存在，推理时透传。
  - DroidOutputs：把模型输出裁回平台真实维度。模型内部动作维统一为 32（其余是 pad），DROID
    只取前 8 维（7 关节速度 + 1 夹爪），即 [ah, 32] → [ah, 8]。

另有 make_droid_example 造随机观测样例、_parse_image 统一图像 dtype 与排布，供测试/调试。
输入：DROID 观测字典（observation/* 键）。输出：含 state、image、image_mask 的字典。
真实观测由 examples/droid/main.py 采集并经 websocket 送来；归一化/反归一化在本变换前后由
policy_config 拼入的 Normalize/Unnormalize 完成。
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# 造一份随机的 DROID 观测样例（供测试/调试用），字段与真实机器人端发来的键名一致：
# 两路 224x224 图像、7 维关节位置、1 维夹爪、一条语言 prompt。
def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


# 把任意来源的图像统一成模型期望的 uint8、HWC 排布：
# 浮点图（0~1）先乘 255 转 uint8；若是 CHW（首维=3）则转置成 HWC。
def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# DroidInputs：把 DROID/Franka 平台的原始观测对齐到模型统一输入格式。
@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    # 决定相机命名与图像掩码方案（π0/π0.5 与 π0-FAST 不同）。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 夹爪开合量：若是标量先升成 1D，才能和 7 维关节位置拼接。
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        # state = 7 维关节位置 + 1 维夹爪 = 8 维本体状态。state: [s=8]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # 外部第三人称相机 + 手腕相机，统一成 uint8 HWC。image: [H, W, C]
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        # 按模型类型确定三个图像槽位的命名、内容与是否有效（mask）。
        # π0 模型固定三路输入：一路第三人称 + 左右两路手腕；缺的槽位用零图占位并 mask 掉。
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # DROID 只有外部相机和一路手腕相机：右手腕槽位填零图，并把它的 mask 置 False。
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                # FAST 模型不对占位图做 mask，因此三路 mask 都为 True（第二路仍是零图）。
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # 统一输出：state + 三路图像(dict) + 三路图像掩码(dict)。掩码告诉模型哪些相机是真数据。
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # 训练时才有 actions（监督标签）；推理时通常没有，这里透传给后续变换。
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # prompt（语言指令）：若以 bytes 传来则解码成 str，供后续 tokenize 使用。
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


# DroidOutputs：把模型输出的动作裁回 DROID 平台真正需要的维度。
@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        # 模型内部动作维统一为 32（不足部分是 pad），DROID 只取前 8 维（7 关节 + 1 夹爪）。
        # actions: [ah, 32] -> [ah, 8]
        return {"actions": np.asarray(data["actions"][..., :8])}
