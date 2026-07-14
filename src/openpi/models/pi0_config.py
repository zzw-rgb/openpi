"""π0 / π0.5 的模型配置文件，属于「模型定义」链路的入口配置。

本文件定义冻结 dataclass ``Pi0Config``（继承 ``model.py`` 的 ``BaseModelConfig``），集中声明
构建 Pi0 模型所需的全部超参数：数据类型、PaliGemma 主干与动作专家（action expert）各自的
Gemma 规格、动作维度 ``action_dim``、一次预测的动作步数 ``action_horizon``、语言 token 上限
``max_token_len`` 等。

核心是 ``pi05`` 布尔开关：它是区分两代模型的「总闸」。``__post_init__`` 据此联动默认值——
π0.5 因把离散化的 state 塞进语言 prompt 而将 token 上限设为 200、并默认开启 state 离散化
（discrete_state_input），π0 则为 48 且不离散化。``model_type`` 据此报告为 PI05 或 PI0
（影响权重加载与数据 transform）。

关键方法：``create`` 延迟 import 并实例化 ``pi0.Pi0``（避免循环依赖）；``inputs_spec`` 声明
模型输入的形状/类型规格（不含真实数据），供 ``fake_obs``/``fake_act``、lazy_init 造占位张量。
本文件不含前向计算，仅作配置；上游由训练/推理脚本读取，下游产出交给 ``pi0.py``。
"""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


# π0 / π0.5 的配置。pi05 布尔开关是区分两代模型的总闸，联动 max_token_len、discrete_state_input 等默认值。
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"  # 视觉-语言主干用 2B Gemma
    action_expert_variant: _gemma.Variant = "gemma_300m"  # 动作专家用 300m 小 Gemma

    # Set the model specific defaults.
    action_dim: int = 32  # 动作维度（不同机器人会被 pad/截到该维）
    action_horizon: int = 50  # 一次预测的动作步数（action chunk 长度 ah）
    max_token_len: int = None  # type: ignore  # 语言 token 上限，下面按 pi05 自动取 200 / 48
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    # π0.5 与 π0 的差异全靠这个开关：(a) state 走离散语言 token 而非连续 suffix token；(b) 用 adaRMSNorm 注入时间
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    # 模型本身不读这个字段，由数据处理侧（ModelTransformFactory）决定是否把 state 离散化进 prompt
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        # π0.5 的语言 prompt 里塞了离散化的 state，占更多 token，故上限 200；π0 只有 prompt，48 够用
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        # 默认让“是否离散化 state”跟随 pi05：π0.5 离散、π0 不离散
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # 同一份 Pi0 代码，按 pi05 报告为 PI05 或 PI0（影响权重加载、数据 transform 等）
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        # 真正实例化模型；延迟 import 避免循环依赖
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    # 声明模型输入的形状/类型规格（不含真实数据），供 lazy_init、fake_obs 等按此造占位张量
    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 三路相机（底座 + 左右腕）的 224x224x3 图像
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    # 按 LoRA 配置返回“冻结哪些参数”的过滤器：LoRA 微调时冻结原始大权重、只训练 LoRA 低秩增量
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")  # 匹配全部 llm 参数（含主干与专家）
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")  # 后缀 _1 的是第二套专家=action expert
        if "lora" in self.paligemma_variant:
            # 主干用 LoRA：冻结 llm 参数
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                # 若动作专家不用 LoRA（要全量训练），就把它从冻结集合里排除
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
