"""归一化统计量（normalization statistics）的表示、在线计算与读写。属于 openpi（π0 / π0.5）
数据链路的「统计量基础设施」：训练前扫一遍数据集算出每个字段（如 state、actions）的逐维统计量，
随 checkpoint 存档；训练与推理时加载，交给 `openpi/transforms.py` 的 Normalize/Unnormalize 使用。

支持两种归一化所需的量：z-score 用 mean/std（减均值除标准差）；分位数归一化（π0.5 常用，对离群值更鲁棒）
用 q01/q99（1% 与 99% 分位数），把 [q01, q99] 线性映射到 [-1, 1]。

关键类 / 函数：
  - `NormStats`：一个字段的统计量容器，逐维保存 mean、std、以及可选的 q01、q99。
  - `RunningStats`：流式（online）统计。数据集通常无法一次性载入内存，这里边读 batch 边更新
    均值/平方均值（推出方差）、min/max，并用直方图（histogram，默认 5000 桶）近似分位数；
    `get_statistics()` 汇总成 `NormStats`。
  - `_NormStatsDict` / `serialize_json` / `deserialize_json`：把「字段名 → NormStats」的字典与 JSON
    互相转换（基于 pydantic 校验）。
  - `save` / `load`：把统计量字典写入 / 读出磁盘目录（norm_stats.json）。

主要输入 / 输出：输入为逐 batch 的向量（形如 [..., D]，最后一维是特征维）；输出为每个字段的 `NormStats`，
以及其 JSON 存档文件。

配合关系：由训练前的统计量计算脚本产出，被 `openpi/transforms.py` 的 `Normalize`/`Unnormalize` 消费，
`NormStats` 也作为 `transforms.py` 里 `NormStats` 类型别名的来源。
"""

import json
import pathlib

import numpy as np
import numpydantic
import pydantic


# NormStats：一个字段（如 state 或 actions）的归一化统计量容器，逐维保存。
# mean/std 供 z-score 归一化；q01/q99（1%/99% 分位数）供分位数归一化（π0.5 用）。
# 训练前扫一遍数据算好、随 checkpoint 存档，推理时加载来做归一化/反归一化。
@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


# RunningStats：流式统计。数据集通常无法一次性载入内存，这里边读 batch 边在线更新
# 均值/平方均值（推出方差）、min/max、以及用直方图近似的分位数。
class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        # 把除最后一维外都拉平成 batch 维：[..., D] -> [N, D]，逐维（D）统计。
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            # 首个 batch：直接初始化各统计量，并按当前 min/max 建立分位数用的直方图分桶。
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            # 后续 batch：若出现更大/更小值，扩展 min/max 并按新范围重排直方图分桶。
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        # 按样本数加权，把本 batch 均值增量融进全局均值（在线均值更新，等价于全量平均）。
        # 同时维护平方均值，后面用 E[x^2]-E[x]^2 得到方差。
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        # 把本 batch 计入各维直方图，供事后估分位数。
        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        # 方差 = E[x^2] - E[x]^2，clip 到非负防浮点误差出现微小负值，再开方得标准差。
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        # 从直方图估出 1%/99% 分位数，打包成 NormStats 返回（供两种归一化方式）。
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        # 直方图法估分位数：目标累计计数 = q * 总数；对每一维沿累计直方图找到跨过该计数的桶，
        # 取该桶左边界作为分位点。桶越多（_num_quantile_bins）近似越精细。
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


# 从目录读取 norm_stats.json，反序列化成 {字段名: NormStats}。推理起服务时加载它做归一化。
def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
