"""checkpoints.py：基于 orbax 的 checkpoint（检查点）保存与恢复。

角色：训练链路的"存档/读档"层。train.py 在主循环里周期性调 save_state 写盘，断点续训时（--resume）用 restore_state 读回；
初始化目录由 initialize_checkpoint_dir 负责。产出：磁盘上按 step 组织的 checkpoint 目录，其中的 params 可直接用于推理/部署。

一个 step 的 checkpoint 存三类"item"：
  - train_state：完整训练状态（步数、参数、optimizer 状态、EMA 参数等），用于断点续训；
  - params：可直接用于推理的参数（优先存 EMA 权重），单独拆出便于部署，无需带上 optimizer 状态；
  - assets：归一化统计量 norm stats，推理时需用训练时同一套 norm 来（反）归一化 state/action。
关键函数：initialize_checkpoint_dir（按 overwrite/resume 处理已存在目录，避免误覆盖）、save_state / restore_state
（写入/读回三类 item）、load_norm_stats（单独读 norm stats）、_split_params / _merge_params
（保存前把"推理参数"从 train_state 剥离避免重复存两份，恢复时再拼回）。CallbackHandler 等封装 orbax 的异步保存回调。
配合：由 train.py 调用；依赖 shared/normalize.py（norm stats 结构）、training/utils.py（TrainState 定义）、data_loader.py（取 norm stats）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    # 目录已存在时按标志决定处理方式：overwrite 清空重来；resume 标记续训；都没有则报错，避免误覆盖。
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 建 orbax CheckpointManager，为三类 item 各配一个 handler：
    # assets 用自定义回调 handler（写 norm stats）；train_state 与 params 用 PyTree handler。
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,  # 默认只保留最新 1 个 checkpoint，省磁盘
            keep_period=keep_period,  # 但每隔 keep_period 步的 checkpoint 永久保留（里程碑存档）
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步写盘超时上限
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    # 边界情况：目录在但还没写过任何 checkpoint（或只有 step 0）。此时无从恢复，取消 resume 改为从头训。
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    # assets 的写入回调：把当前数据管线用的 norm stats 存到 assets/<asset_id> 下，
    # 保证这份 checkpoint 自带反归一化所需的统计量，推理时可直接配对使用。
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    # 拆分：把"可用于推理的参数"（优先 EMA）单独作为 params item，train_state 里则清掉这份以免重复。
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    # 交给 orbax 异步落盘（三类 item 各自序列化）。
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader  # 恢复时不需要 data_loader，仅为与 save_state 保持一致的签名

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        # 用与保存时相同的拆分方式构造"目标形状"，orbax 据此把磁盘数据填回对应结构。
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,  # step=None 表示恢复最新一个
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    # 把拆开的 train_state 与 params 合并回完整 TrainState（含 EMA 判定）。
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    # 让 orbax 在保存 assets item 时调用任意函数（这里是写 norm stats）。只支持保存，不支持恢复。
    def save(self, directory: epath.Path, args: CallbackSave):
        # 多进程训练下只在主进程（rank 0）写一次，避免重复写盘。
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    # 把"推理用参数"从 state 中剥离成独立的 params：
    #  - 有 EMA：推理参数取 EMA，train_state 里把 ema_params 清空（普通 params 仍保留供续训）。
    #  - 无 EMA：推理参数就是 params，train_state 里把 params 清空（用空 dict 作为"已剥离"的标记）。
    # 目的：params item 单独可加载做部署；同时避免同一份权重在两个 item 里存两遍。
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    # _split_params 的逆操作，靠 train_state.params 是否为空来区分当初走的是哪条分支：
    #  - train_state.params 非空 => 当初有 EMA（params 未被清空），故把恢复的 params 填回 ema_params。
    #  - train_state.params 为空 => 当初无 EMA，把恢复的 params 填回 params。
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
