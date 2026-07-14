"""π0 / π0.5 主模型的定义文件，属于「模型定义」链路的核心。

本文件实现基于流匹配（flow matching）的连续动作生成模型 Pi0：一个视觉-语言主干
（PaliGemma，图像走 SigLIP、文本走 Gemma-2B）负责理解「看到什么、要做什么」，再接一个
小的「动作专家」（action expert，Gemma-300m）把带噪动作去噪成真实动作序列（action chunk）。

关键构件：
- ``make_attn_mask``：由输入掩码与 ``mask_ar`` 构造分块注意力掩码，让图像+语言前缀内部双向
  （prefix-lm）、动作/状态后缀相对前缀因果。
- ``posemb_sincos``：把标量（此处为流匹配时间 t）编码成 sin/cos 位置向量。
- ``Pi0`` 类：``embed_prefix`` 组装前缀 token（图像+语言），``embed_suffix`` 组装后缀 token
  （state 与带噪动作，并按 pi05 决定是否用 adaRMS 注入时间 t）；``compute_loss`` 走训练路径
  （回归速度场的 MSE），``sample_actions`` 走推理路径（欧拉法沿速度场把 t 从 1 积分到 0，
  借 KV cache 复用不变的前缀）。

主要输入：``Observation``（图像、state、tokenized prompt）与真实 ``Actions``；
输出：训练时为逐步损失 ``[*b, ah]``，推理时为动作序列 ``[b, ah, ad]``。
上游由训练/推理脚本经 ``pi0_config.Pi0Config.create`` 实例化调用，下游依赖 ``model.py``
的基类与 ``Observation``、以及 ``gemma``/``siglip`` 主干实现。pi05 开关联动三处 π0.5 特有改动：
state 走离散语言 token、adaRMSNorm 注入时间、更长的 token 上限。
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


# 根据两个一维掩码构造二维注意力掩码 attn_mask[b, q, k]：query 位置 q 能否看到 key 位置 k。
# 核心技巧：对 mask_ar 沿序列做累加和（cumsum），把序列切成若干“块”。
#   - mask_ar[i]=0 表示位置 i 与前一个位置属于同一块（累加和不变）；
#   - mask_ar[i]=1 表示从位置 i 起开启一个新块（累加和 +1）。
# 规则“key 的累加和 <= query 的累加和才可见”于是实现分块注意力：
#   同块内部双向互看，靠后的块能看到靠前的块，但靠前的块看不到靠后的块（因果分块）。
# 在 pi0 里用它同时表达：图像+语言前缀内部双向（prefix-lm），动作/状态相对前缀是因果。
def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)  # 每个位置所属“块”的编号 [B, N]
    # cumsum[:, None, :] 是 key 维、cumsum[:, :, None] 是 query 维；key 块号<=query 块号才可见
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N(query), N(key)]
    # padding 位置（input_mask=False）无论作为 query 还是 key 都不参与注意力
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # [B, N, N]
    return jnp.logical_and(attn_mask, valid_mask)  # [B, N, N]


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    # 把标量位置（这里是 flow matching 的时间 t）编码成 sin/cos 向量：pos:[b] -> [b, embedding_dim]
    # 用一组几何递增的周期覆盖不同尺度，让网络能分辨 t 的细微变化
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    # π0 / π0.5 主模型：一个视觉-语言主干（PaliGemma）+ 一个“动作专家”（action expert）小 Transformer，
    # 用 flow matching 直接生成一段连续动作序列（action chunk）。pi05 开关控制三处 π0.5 特有改动。
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        # PaliGemma（2B）作为视觉-语言主干；action expert（300m）是并行的第二套 Gemma 权重，专门处理动作 token
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        # 一个 Gemma Module 同时持有两套专家的参数（configs 里两个 config），前向时按 token 归属分派
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,  # π0.5 才启用 adaRMSNorm 来注入时间条件
            )
        )
        # use_adarms=[False, True]：只有第二套专家（action expert）用 adaRMS，主干 VLM 不用
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # SigLIP 图像编码器 img + Gemma 语言/动作主干 llm，一起放进 PaliGemma 容器
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 把连续动作向量投影到 action expert 的隐藏维度：[B, ah, ad] -> [B, ah, width]
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            # π0.5：时间 t 经 time_mlp 变成 adaRMS 的条件向量（见 embed_suffix），不再和动作拼接
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # π0：state 作为一个连续 token 进 suffix，需要单独的投影层
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # π0：把 [动作 emb; 时间 emb] 拼接（2*width）后过 MLP 融合时间信息
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # 把 action expert 的输出隐藏向量投影回动作维度，得到预测的速度场 v_t：[B, ah, width] -> [B, ah, ad]
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    # 构造 prefix（前缀）token：多路相机图像 + 语言 prompt。这部分是“条件”，去噪时可缓存进 KV cache。
    # 返回 (tokens 特征, input_mask 有效位, ar_mask 分块标记)。ar_mask 全 False => 前缀内部全双向可见（prefix-lm）。
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        # 逐个相机视角过 SigLIP，得到一组图像 patch token：image_tokens: [B, l_img, emb]
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            # 该视角是否有效（缺失相机时整片 mask 掉），把标量 mask 广播到每个 patch：[B] -> [B, l_img]
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            # 图像 token 之间双向注意（同属前缀块），故 ar_mask 记 False
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        # π0.5 里语言 prompt 已经把离散化的 state 拼进去了（Task/State/Action 模板，见 tokenizer）
        if obs.tokenized_prompt is not None:
            # method="embed"：只查 embedding 表把 token id 变向量，不做前向：[B, L] -> [B, L, emb]
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            # 语言 token 与图像 token 同属前缀，全双向，故也记 False
            ar_mask += [False] * tokenized_inputs.shape[1]
        # 沿序列维把各路 token 拼起来：tokens: [B, S_prefix, emb]，S_prefix = 各视角 patch 数 + 语言 token 数
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        # 构造 suffix（后缀）token：这部分随每个去噪步变化（带噪动作 x_t + 时间 t），走 action expert。
        # 返回还多一个 adarms_cond：π0.5 用它做 adaRMSNorm 的时间条件；π0 为 None。
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            # π0：连续 state 经 state_proj 变成 1 个 token 放进 suffix：[B, s] -> [B, 1, width]
            # （π0.5 不走这里，state 已在语言 prompt 里离散成 bin token）
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            # state 是 suffix 的第一个 token，相对前缀开启新块（ar_mask=True），使前缀看不到它
            ar_mask += [True]

        # 带噪动作投影到隐藏维：noisy_actions:[B, ah, ad] -> action_tokens:[B, ah, width]
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 时间 t 做 sin-cos 编码：timestep:[B] -> time_emb:[B, width]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            # π0.5：把 time_emb 过一个 swish-MLP 得到条件向量 adarms_cond，之后在 Gemma 内部用它
            # 自适应地缩放/调制 RMSNorm（adaRMSNorm），从而把时间信息注入每一层，而不占用 token 位。
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens  # 动作 token 本身不拼时间
            adarms_cond = time_emb  # [B, width]，作为时间条件传给 llm
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # π0：把时间 emb 复制到每个动作步，和动作 token 沿特征维拼接后过 MLP 融合
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)  # [B, ah, width]
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # [B, ah, 2*width]
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)  # -> [B, ah, width]
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None  # π0 不用 adaRMS
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        # 动作块第一个 token 开启新块（True），其余动作 token 与之同块（False）=> 动作 token 之间相当于能互看，
        # 且整体只能看前缀/状态、前缀看不到动作。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        # 拼成完整 suffix：tokens:[B, S_suffix, width]，S_suffix = (π0 多 1 个 state) + ah
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # 训练目标 = flow matching：学习一个速度场，把纯噪声沿直线搬运到真实动作。
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 图像增强/缩放/补默认 mask（train=True 时才做数据增强）
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]  # 去掉 (ah, ad) 两维，剩批维
        noise = jax.random.normal(noise_rng, actions.shape)  # 采样高斯噪声，形状同真实动作 [B, ah, ad]
        # 采时间 t~Beta(1.5,1)，再压到 [0.001, 1]。Beta(1.5,1) 偏大 t，让训练更多落在“接近纯噪声”的区段
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # [B]
        time_expanded = time[..., None, None]  # [B, 1, 1] 便于和 [B, ah, ad] 广播
        # 线性插值构造带噪动作：约定 t=1 是纯噪声、t=0 是目标动作（扩散文献惯例，与 π0 论文相反）
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # [B, ah, ad]
        u_t = noise - actions  # 该直线路径的真实速度场（常向量，即回归目标）：[B, ah, ad]

        # one big forward pass of prefix + suffix at once
        # 训练时前缀和后缀一次性拼起来跑一遍前向（无需 KV cache）
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # [B, S_prefix+S_suffix]
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # [S_prefix+S_suffix]
        attn_mask = make_attn_mask(input_mask, ar_mask)  # [B, S, S] 分块注意力
        positions = jnp.cumsum(input_mask, axis=1) - 1  # 每个有效 token 的位置编号（padding 不计数）
        # llm 接收 [前缀 token, 后缀 token] 两段，分别由 VLM 主干和 action expert 处理；只有后缀走 adarms_cond
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        # 取后缀末尾 ah 个 token（即动作 token）的输出，投影回动作维得到预测速度场 v_t：[B, ah, ad]
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 逐样本、逐动作步的 MSE(v_t, u_t)，对动作维求平均：[B, ah, ad] -> [B, ah]
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # 推理 = 从纯噪声出发，用欧拉法沿学到的速度场把 t 从 1 积分到 0，得到动作。
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps  # 步长为负：时间从 1 往 0 走
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))  # 起点 x_1 = 纯噪声

        # first fill KV cache with a forward pass of the prefix
        # 前缀（图像+语言）在整个去噪过程中不变，故先跑一次前向把它的 K/V 存进 kv_cache，
        # 之后每个去噪步只算 suffix（动作 token）对前缀的注意力，省去重复计算图像/语言。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # 第二段传 None 表示这次只处理前缀、不算动作专家；返回填好的 kv_cache
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # 单个欧拉步：给定当前带噪动作 x_t 和时间 time，预测速度 v_t，再走一步 x <- x + dt*v_t
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            # suffix token 相互之间的可见性：[B, S_suffix, S_suffix]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            # suffix（query）看 prefix（key）：前缀全部有效即可见，广播成 [B, S_suffix, S_prefix]
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            # 拼成 suffix 对 (prefix+suffix) 全序列的注意力掩码：[B, S_suffix, S_prefix+S_suffix]
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # suffix 的位置编号要接在 prefix 之后：前缀有效长度 + suffix 内部累加 - 1
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 只喂 suffix，复用 kv_cache 里的前缀 K/V；第一段 None 表示不重算前缀
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            # 取动作 token 输出 -> 预测速度场 v_t：[B, ah, ad]
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 欧拉一步：x 沿 v_t 前进 dt（dt<0，t 递减），返回新的 (x_{t+dt}, time+dt)
            return x_t + dt * v_t, time + dt

        # 循环终止条件：time 减到 ~0 就停（-dt/2 是防浮点误差的阈值）
        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        # 从 (噪声, t=1) 迭代到 t≈0，x_0 即最终动作：[B, ah, ad]
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
