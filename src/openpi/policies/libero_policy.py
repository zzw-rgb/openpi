"""LIBERO 仿真平台的输入输出变换（transform）：仿真观测 ⇄ 模型统一格式。

LIBERO 是一个机器人操作的仿真基准（benchmark）。本文件提供该平台专属的两个 transform，
被 policy_config 作为“平台 data_transforms”环节插入推理数据流；训练与推理共用同一套输入变换。
  - LiberoInputs：把仿真观测对齐到模型统一输入。直接取 8 维 state；把第三人称主视角与手腕视角
    图像统一成 uint8 的 HWC 排布，填入三路图像槽位。LIBERO 无右手腕相机，该槽位用零图占位——
    对 π0/π0.5 把其 image_mask 置 False（标记为补零无效），π0-FAST 不做 mask。actions 仅训练时
    存在，prompt（语言指令）透传给后续 tokenize。类内注释还说明了接自有数据集时如何改键名。
  - LiberoOutputs：仅推理用，把模型输出裁回平台真实维度。模型内部动作维为 32（其余是 pad），
    LIBERO 只取前 7 维，即 [ah, 32] → [ah, 7]。

另有 make_libero_example 造随机观测样例、_parse_image 统一图像 dtype 与排布，供测试/调试。
输入：LIBERO 观测字典（observation/* 键）。输出：含 state、image、image_mask 的字典。
归一化/反归一化由 policy_config 拼入的 Normalize/Unnormalize 在本变换前后完成；结构与
droid_policy.py 对应，区别在相机配置与动作维度。
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# 造一份随机的 LIBERO 观测样例（供测试/调试用）：8 维 state、两路 224x224 图像、一条 prompt。
def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


# 同 droid 里的 _parse_image：统一成 uint8、HWC。浮点图乘 255；CHW 转 HWC。
def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# LiberoInputs：把 LIBERO 仿真平台的观测对齐到模型统一输入格式（训练与推理共用）。
# 想接自己的数据集时，复制这个类、按下方注释改键名即可。
@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        # 第三人称主视角 + 手腕视角，统一成 uint8 HWC。image: [H, W, C]
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        # 统一输入格式：state + 三路图像 + 三路掩码。LIBERO 没有右手腕相机，用零图占位。
        inputs = {
            "state": data["observation/state"],  # state: [s=8]
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                # 不存在的相机用同形状零图补齐，保证模型三路输入槽位都在。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                # π0 把占位图 mask 掉（False）；π0-FAST 不 mask（True）。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        # 训练时才有 actions；透传给后续变换（真正 pad 到模型维度在 PadStatesAndActions 里做）。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # 语言指令：透传给后续 tokenize。输出字典必须带 "prompt" 键。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# LiberoOutputs：把模型输出的动作还原成 LIBERO 数据集格式（仅推理用），主要是裁掉 pad 维。
@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        # 模型内部动作维为 32（含 pad），LIBERO 只取前 7 维为真实动作。actions: [ah, 32] -> [ah, 7]
        return {"actions": np.asarray(data["actions"][..., :7])}
