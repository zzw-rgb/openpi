"""从训练好的 checkpoint（检查点）装配出可直接推理的 Policy（策略）对象。

在推理/部署链路里，本文件是“上电装配”环节：把磁盘上的一份 checkpoint 变成一个随时能
调用 infer 的 Policy。核心函数 create_trained_policy 依次完成四件事：
  1. 定位并加载权重：checkpoint 可为本地路径或 gs:// 远端（必要时自动下载缓存）；通过目录里
     是否存在 model.safetensors 判断是 PyTorch 还是 JAX 格式，分别用对应方式载入模型。
  2. 加载归一化统计量（norm stats，mean/std 或 q01/q99）：优先取 checkpoint 自带 assets 里的
     统计量，确保推理时的归一化与训练时严格一致，否则动作会系统性偏移。
  3. 拼装 transform（变换）链：这是整条推理数据流的骨架。输入链顺序为 repack（重排键名）→
     InjectDefaultPrompt（注入默认语言指令）→ 平台 data_transforms（droid/libero 各自实现）→
     Normalize（归一化）→ 模型 model_transforms（tokenize、图像 resize、pad 到定长）；输出链顺序
     大致相反：模型输出后处理 → Unnormalize（反归一化）→ 平台输出裁维 → repack 反向。
  4. 构造并返回 Policy，附带 sample_kwargs（如去噪步数）、policy_metadata 与后端/设备信息。

输入：TrainConfig（训练配置）+ checkpoint 目录。输出：policy.py 中的 Policy 实例。
被 scripts/serve_policy.py 调用来生成待起服务的策略；与 policies/droid_policy.py、
libero_policy.py（提供平台变换）和 policy.py（最终执行 infer）配合。
"""

import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


# 从一个训练好的 checkpoint 装配出可直接用于推理的 Policy：
# 加载模型权重 → 加载归一化统计量 → 按平台把输入/输出 transform 链拼好 → 返回 Policy。
def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    # checkpoint 可能是本地路径，也可能是 gs:// 远端；必要时先下载/缓存到本地再返回路径。
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    # 通过目录里是否存在 model.safetensors 判断权重是 PyTorch 还是 JAX 格式。
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        # PyTorch：加载 safetensors 权重，并把选定参数转成 bfloat16 以省显存/提速。
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # JAX：从 params 目录恢复参数（以 bfloat16 载入）装配出模型。
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    # data_config 描述该平台/数据集的字段与变换（相机命名、是否用分位数归一化等）。
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        # 归一化统计量（mean/std 或 q01/q99）优先从 checkpoint 自带的 assets 加载，
        # 确保推理用的归一化与训练时完全一致，否则动作会系统性偏移。
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # 组装最终 Policy。核心是把“输入链”和“输出链”按正确顺序拼好——这是整条推理
    # 数据流的骨架：
    return _policy.Policy(
        model,
        # 输入链（对进来的观测依次施加，顺序敏感）：
        transforms=[
            # 1) repack：把机器人端原始键名重排成本仓统一命名（可为空）。
            *repack_transforms.inputs,
            # 2) 若观测里没有 prompt，注入默认语言指令。
            transforms.InjectDefaultPrompt(default_prompt),
            # 3) 平台数据变换：拼 state、整理相机图像、映射动作维度（droid/libero 各自实现）。
            *data_config.data_transforms.inputs,
            # 4) 归一化：用训练时的统计量把 state（和训练时的 action）归一化到统一尺度。
            #    必须在 tokenize/pad 之前，因为 π0.5 会把归一化后的 state 离散成文本 token。
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 5) 模型变换：tokenize（prompt(+state)→token）、图像 resize_with_pad、pad 到定长等。
            *data_config.model_transforms.inputs,
        ],
        # 输出链（对模型产出的动作依次施加，顺序与输入链大致相反）：
        output_transforms=[
            # 1) 模型输出后处理（如 FAST 模型把 token 解码回动作）。
            *data_config.model_transforms.outputs,
            # 2) 反归一化：把归一化空间的动作还原成真实物理量（与第 4 步互逆）。
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 3) 平台输出变换：裁掉 pad 维、还原成机器人可执行的动作维度。
            *data_config.data_transforms.outputs,
            # 4) repack 反向：还原键名（可为空）。
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
