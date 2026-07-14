"""数据变换（data transforms）流水线：把原始数据集样本与模型输出，在「数据集空间」与
「模型空间」之间来回搬运和整形。属于 openpi（π0 / π0.5）的数据链路，训练与推理都用它。

核心约定：每个变换是一个可调用对象（`DataTransformFn` 协议），吃一个可能嵌套的字典、返回一个字典，
处理的是「未 batch 的单帧」，叶子约定为 numpy 数组。多个变换用 `Group` 归组、`CompositeTransform`
按序串联，形成输入链（喂给模型前）与输出链（模型采样后还原）。

关键类：
  - `DataTransformFn` / `Group` / `CompositeTransform`：统一接口、输入/输出分组、顺序组合。
  - `RepackTransform`：按键名映射重排字典结构（把数据集字段名对齐到模型期望的字段名）。
  - `InjectDefaultPrompt` / `PromptFromLeRobotTask`：注入或从 LeRobot 任务名生成语言指令 prompt。
  - `Normalize` / `Unnormalize`：归一化与其逆运算。`use_quantiles=True` 走分位数归一化，把 [q01, q99]
    线性映射到 [-1, 1]（π0.5 用，对离群值更鲁棒）；否则走 z-score（减均值除标准差）。统计量来自
    `openpi/shared/normalize.py` 的 `NormStats`。
  - `ResizeImages`：图像缩放到模型输入尺寸。
  - `SubsampleActions` / `DeltaActions` / `AbsoluteActions`：动作序列的抽帧、以及绝对量↔增量（delta）互转。
  - `TokenizePrompt` / `TokenizeFASTInputs` / `ExtractFASTActions`：把语言 prompt 用 tokenizer 编码成
    token（装配进模型输入），以及 FAST（π0 的动作离散化方案）动作 token 的编码与解码。
  - `PadStatesAndActions`：把状态/动作维度用零填充（pad）到模型固定维度。

主要输入 / 输出：输入为数据集读到的单帧观测/动作字典；输出为整形、归一化、tokenize 后可直接喂模型的字典
（输入链），或把模型输出反归一化、反 tokenize 回真实物理量的字典（输出链）。

配合关系：依赖 `openpi/models/tokenizer.py`（文本/动作 tokenize）、`openpi/shared/normalize.py`
（归一化统计量），并被训练数据管线与推理 policy 组织成具体的输入/输出变换链使用。
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


# DataTransformFn：所有数据变换的统一接口协议。约定为“吃一个（可能嵌套的）字典，
# 返回一个字典”。infer 链路就是把一串这样的可调用对象按序作用在观测/动作上。
# 每个叶子约定是 numpy 数组，且处理的是“未 batch 的单帧”。
@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


# 把一串 transform 串成一个：__call__ 时按顺序把上一个的输出喂给下一个（管道）。
@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        # 把输入扁平化成 "a/b/c" 形式的键，再按 structure 里声明的旧路径取值、重排成新结构。
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


# 若观测里没有 prompt，就注入一个默认语言指令；已有则不覆盖。
@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


# Normalize：用训练时算好的统计量把 state（训练时还有 action）归一化到统一尺度。
# 为什么归一化：各维物理量量纲差异大（弧度 vs 米 vs 归一夹爪），不归一化模型难学/难推；
# 归一后各维分布相近，也便于 π0.5 把 state 离散成固定 bin 的文本 token。
@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    # True 用分位数归一化（映射到 [-1,1]），False 用 z-score（减均值除标准差）。
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        # 用分位数归一化时，校验统计量确实带有 q01/q99。
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        # 没有统计量则原样返回（例如某些不需要归一化的场景）。
        if self.norm_stats is None:
            return data

        # 只对 norm_stats 里出现的键（如 "state"/"actions"）做归一化，其余字段透传。
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    # z-score 归一化：(x - mean) / std。加 1e-6 防止除零。
    # 统计量可能比 x 维度长，按 x 的最后一维截齐。
    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    # 分位数归一化：把 [q01, q99] 线性映射到 [-1, 1]。
    # 为什么归到 [-1,1]：对离群值更鲁棒（不受极端 min/max 拉伸），也契合 flow matching
    # 从标准噪声出发的动作空间尺度。公式：(x-q01)/(q99-q01)∈[0,1] -> *2-1 ∈[-1,1]。
    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


# Unnormalize：Normalize 的逆运算，用在输出链。为什么必须反归一化：模型采样出的动作
# 处在归一化空间（约 [-1,1] 或 z-score 尺度），机器人要的是真实物理量（弧度/米等），
# 必须用同一套统计量还原回去，否则下发的动作幅度完全错误。
@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        # strict=True：反归一化时统计量声明的键必须都在（缺了就是链路装配错了）。
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    # z-score 反归一化：x*std + mean。这里把统计量 pad 到 x 的维度——pad 部分 std=1、mean=0，
    # 即对 pad 出来的动作维不做尺度改变（后续会被平台 Outputs 裁掉）。
    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    # 分位数反归一化：[-1,1] -> [q01,q99]。公式：(x+1)/2 ∈[0,1] -> *(q99-q01)+q01。
    # 若统计量维度 dim 小于动作维（模型 pad 到了更宽），只对前 dim 维还原，pad 尾部原样拼回。
    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


# ResizeImages：把各路相机图像统一到模型输入尺寸（如 224x224）。
# 为什么用 resize_with_pad 而不是直接 resize：先按比例缩放再补边（letterbox），保持原始宽高比、
# 不拉伸变形——几何形变会破坏视觉-动作对齐，尤其抓取任务对物体形状敏感。
@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


# 按步长对动作序列下采样（如把高频动作抽稀），推理链一般不用，训练/数据处理场景可能用到。
@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


# DeltaActions：把“绝对动作”转成“相对当前 state 的增量动作”（delta）。
# 为什么用 delta：很多平台学“相对位移”比学“绝对目标位”更好泛化。mask 指定哪些维走 delta
# （通常关节位置维走 delta、夹爪等维保持绝对）。做法：actions[d] -= state[d]（仅 mask 为 True 的维）。
@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        # 没有动作或没给 mask 时不做任何处理。
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        # 对前 dims 维：mask 为 True 的减去对应 state 分量，False 的减 0（保持绝对）。
        # expand_dims 在 ah 维广播：同一帧 state 作用到动作序列的每一步。
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


# AbsoluteActions：DeltaActions 的逆运算，把 delta 动作加回当前 state 还原成绝对动作。
# 用在输出链：模型若输出 delta，需加上 state 才能得到机器人可执行的绝对目标。
@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        # 与 DeltaActions 对称：mask 为 True 的维加回 state 分量。
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


# TokenizePrompt：把语言 prompt（π0.5 还含离散化的 state）编码成定长 token 序列走语言通道。
# discrete_state_input=True 是 π0.5 的关键：把归一化后的 state 离散成 256-bin，拼进
# 形如 "Task: ..., State: ...; Action:" 的文本，让 state 也以 token 形式喂给语言模型。
@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        # 取出并从字典移除 prompt（后面用 token 表示，原文本不再需要）。缺 prompt 直接报错。
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        # π0.5：需要把 state 一并离散进 prompt，因此这里要求 state 存在。
        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            # π0：state 走单独的数值通道，不进 prompt，这里置空。
            state = None

        # prompt 可能是 0 维 numpy 字符串标量，取出 Python str。
        if not isinstance(prompt, str):
            prompt = prompt.item()

        # tokenize：返回定长 token 序列及其有效掩码（pad 到固定 L，mask 标出真实 token）。
        # tokens: [L]，token_masks: [L]
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


# TokenizeFASTInputs：π0-FAST 变体的 tokenize。FAST 把动作本身也编码成离散 token，
# 用自回归方式生成，因此除了 prompt token，还产出：
#  - ar_mask：标哪些位置是自回归生成的（动作 token）；
#  - loss_mask：训练时只在动作 token 上算 loss。
@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        # 训练时 actions 存在会被一并编码进 token；推理时为 None，只编码 prompt+state。
        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


# ExtractFASTActions：π0-FAST 输出链专用。FAST 模型输出的是离散 token，需解码回连续动作。
@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        # 此处 "actions" 实际是模型吐出的 token，解码成 [action_horizon, action_dim] 的连续动作。
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        # 从 LeRobot 数据集的 task_index 查表得到该条轨迹对应的语言任务描述，作为 prompt。
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


# PadStatesAndActions：把 state 和 action 的维度零填充到模型统一的 action_dim（π0 系列为 32）。
# 为什么要 pad：模型对所有平台用同一套宽度为 32 的动作/状态张量；真实平台维度不足（如 DROID=8、
# LIBERO=7）时补零对齐，推理完再由平台 Outputs 裁掉多余维。
@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        # state: [s] -> [model_action_dim]（尾部补零）。
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        # 训练时的动作也 pad：actions: [ah, ad] -> [ah, model_action_dim]。
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


# apply_tree：对 tree 中“键出现在 selector 里”的叶子施加 fn(叶子, selector对应值)，其余透传。
# Normalize/Unnormalize 借它只对 state/actions 这类有统计量的字段做归一化，图像等不动。
def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    # strict：要求 selector 声明的每个键都能在数据里找到，缺了报错（防链路装配错位）。
    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


# pad_to_dim：沿指定轴把数组末尾补到 target_dim（默认补 0）。已够长则原样返回（不裁剪）。
def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
