# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")

文件级总述（中文）
================
本文件是 openpi（π0 / π0.5）里 Gemma 大语言模型（Large Language Model, LLM）骨干的
JAX/Flax 实现，属于「模型主干」链路，被 `src/openpi/models/pi0.py` 等模型装配文件调用。
它的特点是一份代码同时承载两套参数（两个「专家 / expert」）：expert 0 是 PaliGemma 视觉-语言
主干，处理由图像 patch token（来自 SigLIP，见 siglip.py）与语言 token 拼成的 prefix；
expert 1 是动作专家（action expert），处理机器人动作相关的 suffix token。两个专家共享同一次
自注意力（attention）计算，因此能跨专家互相看到对方，实现「视觉-语言条件 → 动作生成」的融合。

关键类 / 函数：
  - `Config` / `get_config`：单个专家的结构超参数（层数、宽度、head 数等）与预设变体（variant）。
  - `RMSNorm`：均方根归一化（Root-Mean-Square Normalization），同时实现普通 RMSNorm 与自适应
    RMSNorm（adaRMS / adaRMSNorm）。π0.5 的动作专家把扩散/流匹配（flow matching）的时间 t 编码成
    条件 `cond`，由它动态生成缩放（scale）、偏移（shift）、门控（gate），把时间条件注入每一层（类似 DiT 的 AdaLN）。
  - `Embedder`：词嵌入（token embedding），token id 查表成向量，也能投影回词表 logits。
  - `Attention`：多头注意力，含旋转位置编码（Rotary Position Embedding, RoPE）与 GQA 分组、KV 缓存。
  - `FeedForward` / `Block`：前馈网络与单个 Transformer 层；`Block` 里用 adaRMS 的 gate 做门控残差。
  - `Module`：把若干 `Block` 堆叠成完整骨干，暴露 `embed`（查嵌入）与 `__call__`（前向）。
  - 辅助：`_apply_rope`（施加 RoPE）、`_gated_residual`（门控残差）。

输入 / 输出：输入为各专家的 token 嵌入序列、位置索引、注意力掩码（attention mask）、可选 KV 缓存；
输出为各专家最后一层的隐藏表示（供上层做动作预测或解 logits）与更新后的 KV 缓存。

配合关系：上游接 siglip.py（提供图像 patch token）与 tokenizer（提供语言 token）；由 pi0 系列模型
文件组织 prefix/suffix 与掩码后调用本文件；lora.py 提供可选的 LoRA 低秩适配层。
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


# 单个 expert（专家）的结构超参数。一个 Gemma Module 会持有一个 Config 列表，
# 列表里每个元素描述一个 expert 的形状。
@dataclasses.dataclass
class Config:
    width: int  # 特征维度 D / d_model（token 向量的宽度）
    depth: int  # Transformer 层数（所有 expert 必须一致，因为要逐层一起跑）
    mlp_dim: int  # FeedForward 中间隐藏层维度
    num_heads: int  # query 注意力头数 N
    num_kv_heads: int  # key/value 头数 K；小于 num_heads 时为分组查询注意力（GQA），多个 query 头共享一组 KV
    head_dim: int  # 每个注意力头的维度 H
    # LoRA（低秩适配）配置，键如 "attn"/"ffn"；为空则该模块用全量权重训练/推理。
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


# 按名字返回预定义的 Gemma 结构配置。注意这里所有变体都是 num_kv_heads=1（多查询注意力 MQA），
# 8 个 query 头共享 1 组 KV，能显著减小 KV cache 显存并加速推理。
# gemma_300m 常用作 action expert，gemma_2b 常用作 PaliGemma 语言主干；带 _lora 后缀的变体
# 在 attn 和 ffn 上挂 LoRA（用于参数高效微调）。
def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


# RMSNorm（均方根归一化）：Transformer 里替代 LayerNorm 的归一化方式。
# 为什么用 RMSNorm 而不是 LayerNorm：它只按"均方根"缩放、不减均值、也没有 bias，
# 计算更省、数值更稳，是 Gemma/LLaMA 等现代 LLM 的标准选择。
# 这里做了两件事合一：
#   - cond=None 时：普通 RMSNorm（PaliGemma 主干用）；
#   - cond 非 None 时：自适应 RMSNorm（adaRMS / adaRMSNorm，action expert 用），
#     把外部条件（flow-matching 的时间 t 经 MLP 得到的向量）注入到缩放/偏移里。
@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        # x: [B, L, D]，cond（若有）: [B, D_cond]
        dtype = x.dtype  # original dtype, could be half-precision
        # 计算每个 token 特征维上的均方（方差近似），在 float32 下算以保证数值稳定。
        # var: [B, L, 1]
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        # 按均方根归一化：x / sqrt(mean(x^2) + eps)，eps=1e-6 防止除零。normed_inputs: [B, L, D]
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            # 普通 RMSNorm：一个可学习的逐通道缩放参数 scale（[D]），初始化为 0，
            # 用 (1 + scale) 作用，等价于初始时恒等（Gemma 的约定，便于从预训练权重加载）。
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        # 自适应 RMSNorm（adaRMS）：不再用固定的可学习 scale，而是由条件 cond 动态生成
        # 缩放（scale）、偏移（shift）、门控（gate）。为什么这样做：π0.5 的 action expert 是
        # 一个 flow-matching 去噪网络，其行为需要随"扩散时间 t"变化——把 t 编码成 cond 后，
        # 让归一化的缩放/偏移随 t 自适应，就把时间条件平滑地注入到每一层（类似 DiT 的 AdaLN）。
        # 一个 Dense 把 cond 映射到 3*D 维，再切成三份。kernel 初始化为 0：训练初期 scale/shift/gate
        # 都为 0，等价于"归一化后不缩放不偏移、残差门控为 0"，让网络从恒等映射稳定起步。
        # modulation: [B, 3*D]
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        # 在 token 维（L）上插一个广播维：modulation[:, None, :] -> [B, 1, 3*D]，
        # 切成 scale/shift/gate 各 [B, 1, D]，对同一序列内所有 token 共享同一组调制。
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        # 归一化后做仿射：(1 + scale) 缩放再加 shift。normed_inputs: [B, L, D]
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        # gate 不在这里用，而是返回给外层做"门控残差"（见 _gated_residual）：残差支路乘以 gate，
        # 让条件也能控制每层对主干的贡献强度。
        return normed_inputs.astype(dtype), gate


# 词嵌入（token embedding）模块：把整数 token id 查表成向量，也能把向量投影回词表 logits。
@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int  # 词表大小（PaliGemma 约 25.7 万）
    embed_dim: int  # 嵌入维度 = width D

    def setup(self):
        # 嵌入表：[vocab_size, embed_dim]，每一行是一个 token 的向量。
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        # x: [B, L]（token id）-> 查表 -> [B, L, D]
        x = self.input_embedding_table[(x,)]
        # 乘以 sqrt(D) 缩放：Transformer 的经典约定，让嵌入量级与后续 RoPE/attention 匹配。
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        # 把隐藏向量 [B, L, D] 用嵌入表转置投影回词表 logits [B, L, vocab_size]（权重共享/tied）。
        return jnp.dot(x, self.input_embedding_table.T)


# 多专家自注意力（attention）模块。
# 关键点：xs 是一个列表，每个元素是一个 expert 的 token（如 [PaliGemma_tokens, action_tokens]），
# 某个 expert 不参与本次前向时其位置为 None。各 expert 用各自的 QKV 投影权重把 token 投到
# query/key/value，然后把所有 expert 的 q、k、v 沿 token 维拼接成一条长序列，做一次统一的
# attention——于是不同 expert 的 token 能相互看到（跨专家注意力），这正是"视觉语言条件 + 动作"融合之处。
@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        # 拼接后要做同一次 attention，所以各 expert 的头维/头数/KV 头数必须一致（否则无法在头维上对齐）。
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        # 逐 expert 把输入 token 投影成 q/k/v。
        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue  # 该 expert 本次不参与
            if config.num_kv_heads == config.num_heads:
                # 头数=KV 头数（普通多头注意力）：用一个融合权重一次算出 q、k、v。
                # 权重形状 [3, N, D, H]，einsum 把输入 x[B,S,D] 投成 [3,B,S,K,H]（这里 K=N）。
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # x: [B, S, D] -> 3BSKH（3 表示 q/k/v 三份）
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                # 头数 > KV 头数（分组查询注意力 GQA / 多查询注意力 MQA）：q 和 kv 分开投影，
                # 因为它们头数不同。这样 KV 只算 num_kv_heads 组，省显存、加速推理。
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # x: [B, T, D] -> q: [B, T, N, H]
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # x: [B, S, D] -> k,v: 各 [B, S, K, H]（K=num_kv_heads）
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        # 把所有参与 expert 的 q/k/v 沿 token 维（axis=1）拼成一条长序列。
        # 拼接后：q [B, T_total, N, H]，k/v [B, S_total, K, H]（T_total/S_total 是各 expert token 数之和）。
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 给 query 加旋转位置编码（RoPE）。为什么用 RoPE：它把绝对位置编成对 q/k 的旋转，
        # 注意力打分时只依赖相对位置差，天然具备相对位置感知且易外推到更长序列。
        q = _apply_rope(q, positions=positions)
        # 注意力打分前对 q 缩放 1/sqrt(H)，防止点积过大导致 softmax 饱和（标准做法）。
        q *= self.configs[0].head_dim ** -0.5

        # key 同样加 RoPE（q、k 一起旋转，点积才反映相对位置）。
        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        # KV cache（键值缓存）：自回归/分步推理时，把过去 token 的 k、v 缓存下来，
        # 本步只算新 token 的 k、v，然后拼到缓存前面，避免重复计算历史。这里把当前 k/v
        # 拼在缓存之后（axis=1 是 token 维）。训练时 kv_cache 为 None，不走这条路。
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        # 为 GQA/MQA 把 query 头拆成 [K 组, 每组 G 个头]：N = K * G。
        # q: [B, T, N, H] -> [B, T, K, G, H]
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        # 注意力打分 logits = q · k（在头维 H 上求内积）。同一 KV 组 K 内的 G 个 query 头共用该组的 k。
        # 输出 logits: [B, K, G, T, S]（每对 query-key 位置一个分数）。用 float32 累加保证精度。
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # 校验 mask 形状为 [B, 1, T, S]：1 会广播到所有头，T 是 query 长度、S 是 key 长度。
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # 用一个很大的负数把被 mask 掉的位置的 logits 压到近似 -inf，softmax 后其概率≈0。
        # mask 来自 make_attn_mask 的分块 mask：prefix（图像+语言）内部双向可见，suffix（动作）
        # 因果可见——这样动作 token 能看全部 prefix，但生成保持自回归/合理的可见性约束。
        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        # attn_mask[:, :, None, :, :]: [B,1,T,S] -> [B,1,1,T,S]，广播到 K、G 头维。
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        # 在 key 维（S）上做 softmax 得注意力权重 probs: [B, K, G, T, S]，再降回原 dtype。
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # 用注意力权重对 value 加权求和：probs[...,T,S] · v[...,S,H] -> encoded[B,T,K,G,H]。
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        # 合并 KV 组与组内头，恢复成 N=K*G 个头：encoded [B, T, K, G, H] -> [B, T, N, H]
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # 输出投影：把 attention 结果从 [多头, 头维] 投回特征维 D，并按各 expert 切回自己的 token 段，
        # 各自用自己的输出权重（out 投影也是逐 expert 独立的）。
        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # 该 expert 的 token 在拼接序列里占 [start:end) 这一段。
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                # encoded[:, start:end]: [B, T_i, N, H] -> out: [B, T_i, D]
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)  # 保持与输入 xs 一一对应的占位

        # 返回每个 expert 的注意力输出，以及更新后的 (k, v) 供 KV cache 使用。
        return out, (k, v)


# 前馈网络（FeedForward / FFN）：Transformer 每层 attention 之后的逐 token 非线性变换。
# 这里用的是 GeGLU（Gated GELU）门控结构：两条并行线性投影，一条过 GELU 当"门"，
# 逐元素相乘后再投影回原维度。相比普通 MLP，门控结构表达力更强，是 Gemma 的标准 FFN。
@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int  # 输入/输出维度 D
    hidden_dim: int  # 中间隐藏维度（mlp_dim）

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        # 两个升维权重打包在一起：w_gating[0] 算门，w_gating[1] 算主值。形状 [2, D, hidden]。
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        # 门支路：x[B,L,D] · w[0][D,hidden] -> [B,L,hidden]，再过 GELU 激活当门控值。
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        # 主支路：x -> [B, L, hidden]
        ff1 = jnp.dot(x, w_gating[1])
        # 门控相乘（GeGLU 的核心）：gate_value ⊙ ff1 -> [B, L, hidden]
        activations = gate_value * ff1

        # 降维权重：[hidden, D]，把隐藏表示投回特征维。
        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        # activations[B,L,hidden] · [hidden,D] -> outputs[B,L,D]
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


# 一个 Transformer 层（Block）。标准的 Pre-Norm 结构，两段残差：
#   x = x + gate * Attention(RMSNorm(x))
#   x = x + gate * FFN(RMSNorm(x))
# 每段先归一化再进子层，输出经门控（adaRMS 情形）后加回残差。列表 xs 里每个元素对应一个 expert，
# 各 expert 用各自的归一化/FFN 权重，但共享同一个 Attention（跨专家）。
# sharding.activation_sharding_constraint 是多卡切分约束（对激活的分片提示），不改变数值。
@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        # adarms_cond: 每个 expert 一个条件向量（或 None）；有条件的走 adaRMS，无条件的走普通 RMSNorm。
        xs = sharding.activation_sharding_constraint(xs)
        # 训练时可选 dropout；dropout=0 时用恒等函数占位（推理路径）。
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        # ---- 第一段：注意力子层 ----
        # 逐 expert 先做 attention 前的 RMSNorm（Pre-Norm）。若该 expert 有 adarms_cond，
        # 则返回门控 gate 用于门控残差；否则 gate 为 None（普通残差相加）。
        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        # 一次跨专家 attention，返回各 expert 的输出与更新后的 KV cache。
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        # 门控残差：x = x + gate * post_attn（gate 为 None 时就是普通残差 x + post_attn）。
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # ---- 第二段：前馈子层 ----
        # 逐 expert 做 FFN 前的 RMSNorm，然后过各自的 FeedForward（这里用带 LoRA 的版本）。
        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        # 第二段门控残差。
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


# KV cache 类型：一对 (k, v)，各带层维 l（每层各存一份）、batch b、token 数、头数、头维。
KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


# 整个 Transformer 模型：把若干 expert 的 token 一起过 depth 层 Block，最后各自做 final RMSNorm。
# "mixture of different weights for different tokens"：不同 token（视觉语言 vs 动作）走不同权重的 expert，
# 但在每层 attention 里彼此可见。
@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str  # 嵌入/计算用的 dtype（如 bfloat16）

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False  # 是否启用自适应 RMSNorm（action expert 用时间条件时为 True）

    def setup(self):
        # all experts must have the same depth
        # 所有 expert 必须同层数，才能在同一个 scan 里逐层并肩前向。
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # 词嵌入表只给第一个 expert（PaliGemma）建；action expert 的输入不是查表得来，
        # 而是由外部（动作/状态编码）直接给出 embedding。
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        # nn.remat：梯度检查点（重计算），前向不保存中间激活、反向时重算，用时间换显存。
        # nothing_saveable = 什么都不缓存（最省显存）。static_argnums=(5,) 把 deterministic 标为静态参数。
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        # nn.scan：把 depth 个相同结构的 Block 沿"层"维堆叠成一次扫描，参数在 axis=0 上按层堆放，
        # 编译更快、代码更简洁（等价于 for 循环叠 depth 层，但对 XLA 更友好）。
        # in_axes 指定各输入是"按层切分（0）"还是"每层广播复用（nn.broadcast）"：
        #   kv_cache 按层切分（每层各一份），positions/mask/adarms_cond/deterministic 每层共用。
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        # 每个 expert 一个最终归一化（输出前的 RMSNorm）。
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    # 把语言 token id 查表成向量（仅第一个 expert 用）。tokens: [B, T] -> [B, T, D]
    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    # 主前向。输入是"已嵌入"的各 expert token 列表（图像+语言 token 已由 SigLIP/embedder 准备好，
    # 动作 token 由外部编码好），本函数只负责过 Transformer。
    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],  # 每个 token 的位置索引（供 RoPE 用）
        mask: at.Bool[at.Array, "b t s"],  # 分块注意力 mask（prefix 双向 + suffix 因果）
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,  # 各 expert 的 adaRMS 条件
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        # mask: [B, T, S] -> [B, 1, T, S]，多出的 1 会在 attention 里广播到所有头。
        mask = jnp.asarray(mask)[:, None, :, :]
        # 没给条件就全填 None，退化为普通 RMSNorm。
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # 过 depth 层 Block（scan）。
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # 逐 expert 做最终 RMSNorm（RMSNorm 返回 (out, gate)，这里只取 out=[0]）。None 的 expert 原样返回。
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    # 参数初始化辅助：用一批全零假输入跑一遍 embed 和前向，触发 Flax 惰性建参。
    # use_adarms[i] 决定第 i 个 expert 是否用 adaRMS（决定要不要建那份 modulation Dense 权重）。
    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


# 旋转位置编码（RoPE）：把每个 token 的位置编成一次二维平面旋转，作用到 q/k 上。
# 直觉：把头维两两配对成 (x1, x2) 复平面坐标，按角度 position/timescale 旋转；
# 两个 token 的 q·k 内积只依赖它们的角度差（即相对位置），所以模型获得相对位置感知、且易外推。
# 注意这里注释里的 D 指"头维 H"（函数内 x.shape[-1] 是每头维度）。
def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    # 不同维度对使用不同频率（timescale），低维转得快、高维转得慢，覆盖多尺度的位置信息。
    # freq_exponents: [H/2]，timescale = 10000^(2i/H)。
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    # 每个位置每个频率对应的旋转角度（弧度）：positions[...,L,1] / timescale -> [..., L, H/2]
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    # 把头维前后对半切成 x1、x2 作为旋转的两个分量，应用二维旋转矩阵。
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # 命名规则：第 0 个 expert（PaliGemma）不加后缀，以便无缝加载官方 PaliGemma 权重；
    # 后续 expert（action expert）加 "_1" 等后缀、从头初始化。实际只用两个 expert。
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


# 门控残差：gate 为 None 时是普通残差 x + y；有 gate（adaRMS 情形）时残差支路乘 gate，
# 让条件（时间 t）也能调节每层子层对主干的贡献强度。
def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
