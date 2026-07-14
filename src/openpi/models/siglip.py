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

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision.

文件级总述（中文）
================
本文件是 openpi（π0 / π0.5）里 SigLIP 视觉编码器（一个视觉 Transformer, Vision Transformer / ViT）
的 JAX/Flax 实现，属于「感知前端」链路：把机器人相机图像编码成一串 patch token，供 Gemma 骨干
（见 gemma.py）当作 prefix 的一部分来读。整体流程为：图像切成小块（patch）→ 每块线性投影成一个
token → 加位置编码（position embedding）→ 过 Transformer encoder → 输出每个 patch 的特征 token。
openpi 采用 So400m/14 变体，且池化类型（pool_type）设为 "none"，即不做全局池化、保留全部 patch token；
这些图像 token 会与语言 token 拼接成 prefix 一起送进 Gemma（π0.5 的关键设计之一）。

关键类 / 函数：
  - `posemb_sincos_2d` / `get_posemb`：位置编码，分别给出二维正弦-余弦（固定不学习）与可学习两种；
    SigLIP 默认用可学习位置编码，为 Transformer 注入「每个 patch 在图像哪个位置」的信息。
  - `MlpBlock`：Transformer 层内的前馈网络（MLP）。
  - `Encoder1DBlock`：单个 Transformer encoder 层（自注意力 + MLP + 残差 + 归一化）。
  - `Encoder`：把多个 `Encoder1DBlock` 堆叠成完整编码器。
  - `MAPHead`：多头注意力池化头（Multihead Attention Pooling），在需要单一图像向量时使用。
  - `_Module` / `Module` / `decode_variant`：整体模型（含 patch 嵌入卷积、位置编码、Encoder）、
    对外构造函数、以及把变体名（如 So400m/14）解码成具体超参数。

输入 / 输出：输入为图像张量（形如 [B, H, W, C]）；输出为 patch token 序列（形如 [B, P, D]，
P 为 patch 数，D 为特征维度），pool_type="none" 时不聚合为单向量。

配合关系：输出的 patch token 由 pi0 系列模型文件拼进 prefix 后交给 gemma.py 的骨干处理。
"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


# 二维正弦-余弦位置编码：为每个 patch 网格坐标 (y, x) 生成固定（不学习）的位置向量。
# 为什么需要位置编码：Transformer 本身对 token 顺序无感，必须显式注入"这个 patch 在图像哪个位置"。
# 二维 sincos 把高、宽两个方向各编一半维度，用不同频率的 sin/cos 表示坐标。
def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    # y, x: [h, w] 的网格坐标。
    y, x = jnp.mgrid[:h, :w]

    # width 必须能被 4 整除：sin(x)/cos(x)/sin(y)/cos(y) 四段各占 width/4。
    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    # omega: 一组从低到高的频率（与 RoPE 类似的多尺度思想）。
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    # 把每个位置坐标乘上每个频率：flatten 后 [P] 与 omega[width/4] 外积 -> [P, width/4]。
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    # 拼成 [P, width]，再加 batch 维 -> [1, P, width]（可广播到任意 batch）。
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


# 按类型返回位置编码：可学习（learn）或固定二维 sincos（sincos2d）。SigLIP 默认用可学习位置编码。
def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        # 可学习位置编码：一张 [1, P, width] 的参数表，随训练更新。
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


# ViT 的前馈块（MLP / FFN）：升维 -> GELU -> 降维。与 Gemma 的 GeGLU 不同，这里是标准两层 MLP。
class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim（不指定则为 4 倍输入维）
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d（B, L, D）
        # 升维到 mlp_dim：[B, L, D] -> [B, L, mlp_dim]
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        # 降回原维度：[B, L, mlp_dim] -> [B, L, D]
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


# 单层 Transformer encoder：多头自注意力（MHSA）+ MLP，两段都是 Pre-Norm 残差。
# 与 Gemma 的 Block 相比：这里用 LayerNorm（而非 RMSNorm）、标准双向自注意力（无 mask、无 RoPE、
# 无 KV cache），因为视觉 patch 之间是全连接双向可见的，且只做一次性编码。
class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        # out 字典收集各中间激活，便于调试/特征提取，不影响主输出。
        out = {}
        x = sharding.activation_sharding_constraint(x)
        # ---- 自注意力子层（Pre-Norm）----
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        # 自注意力：query 和 key/value 都来自同一 y（自己看自己），patch 间双向交换信息。
        # y: [B, L, D] -> [B, L, D]
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        # 残差相加。
        x = out["+sa"] = x + y

        # ---- MLP 子层（Pre-Norm）----
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        # 残差相加。
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


# 把 depth 层 Encoder1DBlock 叠起来，末尾再接一个 LayerNorm。
# 支持两种堆叠方式：scan（用 nn.scan + 梯度检查点，省显存/编译快）或普通 for 循环。
class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int  # 层数
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            # scan 路径：把单层 Block 包上 remat（梯度检查点，反向重算省显存），
            # 再用 nn.scan 沿层维堆 depth 次；参数在 axis=0 按层堆放。
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            # 把 scan 输出的各层中间激活拆回按层的字典。
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            # 普通路径：显式 for 循环叠 depth 层，逐层前向。
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        # 末尾 LayerNorm（encoder_norm），输出最终的 patch token 表示 [B, L, D]。
        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


# 多头注意力池化（MAP Head）：用一个可学习的 query 向量 probe 去"询问"所有 patch token，
# 把整张图聚合成一个向量。当 pool_type="map" 时用它做全局池化。openpi 用 pool_type="none"，不走这里。
class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        # probe：一个可学习的查询向量 [1,1,D]，复制到 batch -> [B,1,D]。
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        # 以 probe 为 query、所有 patch 为 key/value 做注意力，得到 [B, 1, D]（一个聚合 token）。
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        # 再接一个 LayerNorm + MLP 残差，最后去掉长度维返回 [B, D]。
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


# ViT 主体：图像 -> patch token -> 位置编码 -> Encoder -> 输出。
# 注意实际实例化时通过下方 Module() 工厂 + decode_variant 填入 So400m/14 的具体超参。
class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None  # 分类头类别数；None 表示不建分类头（当特征提取器用）
    patch_size: Sequence[int] = (16, 16)  # 每个 patch 的像素尺寸（So400m/14 用 14x14）
    width: int = 768  # 特征维度 D
    depth: int = 12  # Encoder 层数
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"（位置编码类型）
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"（openpi 用 "none"：不池化，保留每个 patch token）
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # patch 提取和位置编码在 float32 下做，数值更稳。
        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)  # image: [B, H, W, C]

        # Patch extraction
        # 用一个卷积核大小=步长=patch_size 的卷积做"切 patch + 投影"：不重叠地扫过图像，
        # 每个 patch 变成一个 width 维向量。这等价于把图像切成小块再各自线性投影。
        # image [B, H, W, C] -> x [B, h, w, width]，其中 h=H/patch, w=W/patch。
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        # 把二维 patch 网格摊平成 token 序列：[B, h, w, width] -> [B, h*w, width]，即 [B, P, D]。
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        # 加位置编码（在可能拼接 cls token 之前）。x: [B, P, D] + posemb[1, P, D]
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        # pool_type=="tok" 时在序列前拼一个可学习的 [CLS] token 做全局汇聚；openpi 不用这条。
        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # 现在转回计算 dtype（可能是半精度 bfloat16），Encoder 在该精度下算以省显存/加速。
        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        # 过 Transformer encoder，得到每个 patch 的编码 [B, L, D]。
        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        # 池化：把 patch token 序列聚合成一个全局向量 x（用于分类/表征）。
        # openpi 走 "none" 分支：不池化，x 保持为全部 patch token，直接当图像 token 交给 Gemma。
        if self.pool_type == "map":
            # 注意力池化。
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            # 全局平均池化（对所有 token 取均值）。
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            # 取第 0 个 token 作为全局表示。
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            # 取 [CLS] token 作为全局表示，其余为 patch token。
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            # 不池化：保留每个 patch 的 token（openpi 用这条）。
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        # 把 patch token 还原成二维空间排布 [B, h, w, D]（便于需要空间结构的下游使用）。
        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        # 可选的表征投影层（pre_logits）：再过一层 Dense + tanh。默认关闭。
        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        # 可选分类头：把特征投到 num_classes 维 logits。openpi 当特征提取器用，num_classes=None，不建。
        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        # 返回 (x, out)：pool_type="none" 时 x 就是全部图像 patch token [B, P, D]，
        # out 里还带各中间激活。这些 patch token 会被拼进 Gemma 的 prefix。
        return x, out


# 工厂函数：按变体名（如 "So400m/14"）解析出结构超参，再构造 _Module。
def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


# 把 "So400m/14" 这样的字符串解析成超参字典：字母部分选宽度/深度/头数等，"/14" 指定 patch 大小 14x14。
def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    # 各规模变体的宽度/深度/mlp_dim/头数查表（源自 ViT/SigLIP 论文的标准配置表）。
    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
