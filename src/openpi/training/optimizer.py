"""optimizer.py：学习率调度（LR schedule）与优化器（optimizer）的配置与组装。

角色：训练链路里的"优化器工厂"。config.py 的 TrainConfig 里挑定一个 LRScheduleConfig + 一个 OptimizerConfig，
train.py 在 init_train_state 时调用本文件的 create_optimizer 把二者组装成最终的 optax.GradientTransformation，
交给 train_step 做梯度更新。产出：一个 step -> 参数更新 的 optax 变换。

设计模式：统一的"配置类（frozen dataclass）+ create()"——每个 dataclass 存一份不可变超参，create() 现场造出对应的 optax 对象。
关键类：
  - LRScheduleConfig（协议）及实现 CosineDecaySchedule（warmup + cosine 衰减，最常用）、RsqrtDecaySchedule
    （warmup 后按 1/sqrt(step) 衰减，适合总步数不定的长训练）：create() 返回 optax.Schedule（step -> lr）。
  - OptimizerConfig（协议）及实现 AdamW（权重衰减 Adam，主力）、SGD：create() 返回优化器变换，可接入 weight_decay_mask
    与梯度裁剪（clip_gradient_norm）。
  - create_optimizer(optimizer_cfg, lr_schedule_cfg, weight_decay_mask)：把 schedule 注入优化器并叠加梯度处理，得到最终变换。
配合：被 config.py（声明各 config 用哪套 LR/优化器）与 train.py（实例化并用于 train_step）引用。
"""

import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    # 学习率调度配置的统一接口：create() 返回一个 optax.Schedule（step -> lr 的函数）。
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    # warmup + cosine 衰减：训练最常用的组合。
    warmup_steps: int = 1_000  # 前若干步线性升温，避免一上来大 lr 冲垮预训练权重
    peak_lr: float = 2.5e-5  # warmup 结束时达到的峰值 lr
    decay_steps: int = 30_000  # 从峰值按 cosine 衰减到 decay_lr 的总步数
    decay_lr: float = 2.5e-6  # 衰减终点 lr

    def create(self) -> optax.Schedule:
        # init_value 设成 peak/(warmup+1)：从接近 0 的很小值线性升到 peak，再 cosine 降到 end_value。
        # warmup 让早期梯度统计稳定；cosine 尾部平滑降低 lr，利于收敛与泛化。
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    # warmup 后按 1/sqrt(step) 衰减（Transformer 训练常见）。相比 cosine 不需要预先定"总步数"，
    # 适合训练步数不确定/很长的场景。
    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        # 拼接两段：前 warmup_steps 步线性升到 peak_lr；之后按 peak/sqrt((timescale+step)/timescale) 衰减。
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    # 优化器配置的统一接口：create() 接收学习率（标量或 schedule）与可选的 weight decay 掩码，
    # 返回 optax.GradientTransformation。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9  # 一阶矩（动量）衰减率
    b2: float = 0.95  # 二阶矩衰减率；比常见 0.999 略小，对大模型/长序列训练更稳
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    # weight decay 取极小值（近似关闭）：置 0 反而触发某些环境的 OOM，故用可忽略的正数替代。
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0  # 梯度全局范数裁剪阈值，防止个别大梯度导致发散

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        # chain：先按全局范数裁剪梯度，再走 AdamW。顺序很关键——裁剪要在 Adam 更新之前作用于原始梯度。
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        # SGD 分支不支持 weight decay 掩码（断言其为 None）。默认训练用的是 AdamW，这里作为备选。
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    # 组装入口：先由 schedule 生成"step->lr"函数，再交给 optimizer 造出最终的梯度变换。
    # train.py::init_train_state 调用它得到 tx，供 tx.update / tx.init 使用。
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
