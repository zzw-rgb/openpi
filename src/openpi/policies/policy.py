"""推理主流程封装：把训练好的模型包成对外统一的 Policy（策略）对象。

这是整条推理/部署链路面向机器人端的核心执行者。机器人本体端（真机或仿真）采集到的单帧
观测——多路图像 image、低维本体状态 state、语言指令 prompt——经 websocket 送到这里，由
Policy.infer() 串起三段完整流程：
  1. 输入 transform（变换）：归一化 state、把 prompt(+state) tokenize（分词）、图像
     resize_with_pad、pad（补零）到模型定长维度，并加上 batch 维。
  2. model.sample_actions：flow matching（流匹配）从纯噪声出发做多步欧拉积分去噪，
     采样出一段归一化空间的动作序列（action chunk，动作块）。π0.5 走的就是这一路。
  3. 输出 transform：去掉 batch 维、Unnormalize（反归一化）把动作还原成真实物理量，
     并按平台裁掉 pad 维，最终返回形如 [ah, ad]（动作步数 × 动作维）的动作块。

关键类：Policy 负责上述 infer 主流程，同时兼容 JAX（JIT 编译 sample_actions）与 PyTorch
（搬设备、eval 模式）两种后端；PolicyRecorder 是可选包装器，透传 infer 的同时把每步输入
观测与输出动作落盘为 .npy，供离线调试复现。

输入：obs（单帧观测字典，未 batch）。输出：含 actions（[ah, ad] 动作块）与 state、
policy_timing 的字典。上游由 policy_config.create_trained_policy 装配 transform 链并构造
本对象，再由 serving/websocket_policy_server.py 在每个连接里反复调用 infer。
"""

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


# Policy（策略）是推理链路对外的统一入口：机器人端把观测（多路图像 + 低维 state +
# 语言 prompt）发进来，infer() 依次跑“输入 transform（含归一化、tokenize、pad）
# → model.sample_actions（flow matching 去噪采样出动作序列）→ 输出 transform（把
# 归一化动作反归一化回真实物理量）”，最后返回一个 [ah, ad] 的动作块给机器人执行。
class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        # 把一串输入 transform 组合成单个可调用对象，infer 时按顺序作用在观测上。
        # 顺序在 policy_config.create_trained_policy 里装配：平台变换 → 注入默认 prompt
        # → 数据变换 → Normalize（归一化）→ 模型变换（tokenize、pad 到定长）。
        self._input_transform = _transforms.compose(transforms)
        # 输出 transform 顺序相反：模型输出后处理 → Unnormalize（反归一化）→ 平台输出裁剪。
        self._output_transform = _transforms.compose(output_transforms)
        # 传给 sample_actions 的额外参数（如去噪步数 num_steps）。
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            # PyTorch 分支：把模型搬到目标设备并切到 eval 模式（关闭 dropout 等），
            # 直接用其 sample_actions 方法。
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            # JAX 分支：用 module_jit 对 sample_actions 做 JIT 编译以加速重复推理；
            # 采样需要随机数 key（flow matching 的初始噪声），无则用固定种子 0。
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # obs：机器人端发来的单帧观测字典（未 batch），如 image/state/prompt 等。
        # noise：可选的外部指定初始噪声，便于复现同一次去噪结果；不给则内部随机采样。

        # Make a copy since transformations may modify the inputs in place.
        # transform 可能就地改写字典，先做一次浅拷贝，避免污染调用方传入的 obs。
        inputs = jax.tree.map(lambda x: x, obs)
        # 【第一步】跑输入 transform：归一化 state、把 prompt(+state) tokenize、
        # 图像 resize_with_pad、动作/状态 pad 到模型维度等。产物是模型期望的字段格式。
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            # 每个叶子加一个 batch 维（[...] -> [1, ...]）并转成 jax.Array，模型按 batch 处理。
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            # 从 RNG 里劈出一个子 key 供本次采样使用，另一半留给下次，保证随机流不重复。
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            # PyTorch 分支：numpy -> tensor -> 目标设备，并同样加 batch 维。
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            # PyTorch 侧 sample_actions 第一个参数用设备名占位（不需要 JAX 的 RNG key）。
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            # 外部传入噪声时转成对应框架张量；noise: [ah, ad]
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
                # 补上 batch 维：[ah, ad] -> [1, ah, ad]，与模型期望对齐。
            sample_kwargs["noise"] = noise

        # 把扁平字典组装成结构化的 Observation 对象（图像/掩码/state/token 等分槽位）。
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        # 【第二步】采样动作：flow matching 从纯噪声出发做欧拉积分去噪，得到动作序列。
        # actions: [B=1, ah, ad]（此时仍是归一化空间的值）。
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        # 去掉 batch 维并搬回 numpy（PyTorch 还需 detach + 移回 CPU）：[1, ah, ad] -> [ah, ad]。
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # 【第三步】跑输出 transform：Unnormalize 把动作从归一化空间还原为真实物理量，
        # 并按平台裁掉 pad 出来的多余维度（如 DROID 取前 8 维、LIBERO 取前 7 维）。
        # actions: [ah, ad] -> 反归一化 -> 机器人可执行值。
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


# PolicyRecorder 是一个包装器：透传 infer 调用，同时把每步的输入观测和输出动作
# 落盘成 .npy，用于离线调试/复现（把内层真实 Policy 当被代理对象）。
class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
