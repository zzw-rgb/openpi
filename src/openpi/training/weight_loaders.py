"""weight_loaders.py：训练开始前，把预训练权重（pretrained weights）灌入随机初始化的模型参数。

角色：训练链路的"权重灌入器"。config.py 的 TrainConfig 指定用哪个 WeightLoader，train.py 在 init_train_state 中
经 _load_weights_and_validate 调用它，把随机初始化的参数树替换为预训练值，随后校验结构/形状/dtype。产出：与输入结构一致、
已填好预训练值的参数树。

统一接口 WeightLoader.load(params)：入参 params 是模型的"参数占位树"（只有形状/结构、无有效数值），
返回结构完全相同、但用预训练值填好的参数树；若只加载了一部分（如仅主干），必须与原 params 合并补齐缺失部分。
关键类：
  - NoOpWeightLoader：什么都不加载、原样返回，用于从随机初始化从零训练。
  - CheckpointWeightLoader：从一个已有 checkpoint 全量加载参数（如 π0/π0.5 base 权重续训或微调）。
  - PaliGemmaWeightLoader：只加载官方 PaliGemma 的视觉语言主干（VLM backbone），动作专家等其余部分保持随机初始化。
  - _merge_params(loaded_params, params, missing_regex)：把加载到的子集与占位树合并，按正则决定哪些键保留随机初始化。
说明：本文件只管"把值填进去"，不涉及 LoRA freeze / EMA 等训练期逻辑（那些在 config.py 与 train.py）。
"""

import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    # 空加载器：不改任何权重，原样返回。用于"从随机初始化开始训练"的场景。
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        # 从 checkpoint 目录加载全量参数为 np.ndarray（dtype 转换与分片留给训练代码去做）。
        # maybe_download 支持本地路径或 gs:// 远程（自动下载缓存）。
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        # checkpoint 一般不含 LoRA 参数。若当前模型是 LoRA 变体，名字含 "lora" 的参数在 loaded 里缺失，
        # 用 missing_regex 把这些缺项从 params（新初始化值）补进来，其余用 checkpoint 覆盖。
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        # 下载官方 PaliGemma 预训练权重（.npz，224 分辨率视觉语言主干）。
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        # 扁平权重按 "/" 反嵌套成树，挂到 "PaliGemma" 子树下（对应模型里的视觉语言主干命名）。
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        # missing_regex=".*"：PaliGemma 只覆盖主干；π0/π0.5 特有的 action expert 等参数不在其中，
        # 这些"缺失"的全部从 params 的新初始化值补齐，从而支持带 action expert 的模型。
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    # 把两棵参数树都拍扁成 "a/b/c" -> array 的字典，便于按路径逐项比对。
    # flat_ref = 模型期望的参数（占位/新初始化值）；flat_loaded = 从磁盘加载的预训练值。
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    # 第一步：凡是加载值中"模型也有"的键，就用加载值覆盖（dtype 对齐到模型期望）。
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()  # 尽早释放加载权重的内存

    # Then, merge any missing weights as defined by the missing regex.
    # 第二步：模型需要、但加载值里没有、且键名匹配 missing_regex 的参数，用模型自带的初始化值补上。
    # 这样最终结果结构与 params 完全一致（预训练能覆盖的覆盖，覆盖不到的保持新初始化）。
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
