"""训练入口（train.py）：openpi（π0 / π0.5）训练链路的最外层主程序。

角色：本文件不实现 π0/π0.5 的 flow matching（流匹配）损失本身——那在 models/pi0.py::compute_loss 里；
它负责把整套训练流程"串起来"并驱动主循环。上游由命令行 `python scripts/train.py <config_name> --exp_name=...`
调用（config 名字见 training/config.py 底部 _CONFIGS）；产出：周期性写出的 checkpoint（含可推理参数与 norm stats）
和 wandb 训练曲线。

主要流程（都在 main() 里）：
  1. 解析 TrainConfig（training/config.py），拿到 model / data / optimizer / lr / batch / 步数等全部超参；
  2. 建立设备网格 mesh 与 sharding（FSDP 全分片 + 数据并行，见 training/sharding.py）；
  3. 由 data_loader.create_data_loader（training/data_loader.py）造 DataLoader，取一个 batch 观察数据形状；
  4. init_train_state：初始化 TrainState（模型参数 + optimizer 状态 + 可选 EMA 参数），
     并经 weight_loaders（training/weight_loaders.py）灌入预训练权重；LoRA/部分冻结时用 freeze_filter 决定哪些参数可训练；
  5. 主循环反复调用 train_step：前向 compute_loss -> 反传 -> optimizer 更新 -> （若启用）EMA 滑动平均 ->
     周期性用 checkpoints.save_state（training/checkpoints.py）存档、向 wandb 记日志。

关键函数：init_logging/init_wandb（日志与实验追踪）、_load_weights_and_validate（加载并校验预训练权重的结构/形状/dtype）、
init_train_state（构建训练状态）、train_step（被 jax.jit 编译、反复调用的单步训练函数）、main（编排全流程）。
记号约定：B=batch，ah=action_horizon（动作时间步），ad=动作维度，s=状态维度，L=token 数，C/H/W=图像通道/高/宽。
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    # 用 weight_loader（见 weight_loaders.py）按 params 的形状/结构去加载预训练权重。
    # params_shape 只是"形状占位"（ShapeDtypeStruct 的 pytree），不含真实数值。
    loaded_params = loader.load(params_shape)
    # 校验加载回来的 pytree 结构、形状、dtype 与期望完全一致，防止权重对不上模型。
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    # loader 可能只加载了部分权重（其余仍是占位 ShapeDtypeStruct）。这里把占位项剔除，
    # 只留真正加载到数值的参数返回，后续在 init 里与新初始化的参数合并。
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    # 构造 optimizer（optax GradientTransformation）：包含梯度裁剪 + AdamW + 学习率 schedule。
    # 具体见 optimizer.py::create_optimizer。这里 weight_decay_mask=None（不做逐参数的 weight decay 掩码）。
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        # 按 config.model（Pi0Config 等）随机初始化模型及其全部参数。
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        # 若提供了预训练权重（partial_params），把它们覆盖进随机初始化的参数里。
        # nnx.split 把模型拆成 graphdef（结构）+ state（参数）；replace_by_pure_dict 用加载的权重
        # 替换对应项（若不是 state 的子集会报错）；再 merge 回完整模型。
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        # 被 freeze 的参数（LoRA 场景下的主干）不参与训练，转成 bfloat16 省显存、加速前向；
        # 可训练参数保持 float32 以保证优化数值稳定。
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # 打包成 TrainState：步数、全部参数、模型结构 graphdef、optimizer、optimizer 状态、EMA 相关。
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            # optimizer 状态只针对"可训练"参数（trainable_filter = Param 且非 freeze），
            # freeze 的参数不进 optimizer，节省 Adam 一阶/二阶矩的显存。
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            # 若启用 EMA，则额外保存一份 EMA 参数（初值=当前参数）。EMA 权重通常泛化更好，用于推理/存档。
            ema_params=None if config.ema_decay is None else params,
        )

    # 先用 jax.eval_shape 只跑"形状推断"（不真正分配显存/算数值），得到 TrainState 的形状树，
    # 据此计算 FSDP sharding 方案（参数如何切分到多设备）。
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # resume（断点续训）时不在这里初始化真实数值，只返回形状与 sharding；
    # 真正的参数会由 checkpoints.restore_state 从磁盘恢复。
    if resume:
        return train_state_shape, state_sharding

    # 非续训：先把预训练权重加载好（如 PaliGemma / pi0_base checkpoint）。
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    # 用 jit 真正执行 init：输入（预训练权重）在各设备复制，输出按 FSDP 方案切分。
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.  # 复用 partial_params 缓冲区省显存
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # 把结构 graphdef 与参数 state 合并回可调用的模型对象；train() 切到训练模式（如启用 dropout）。
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # 调用模型的 flow matching 损失（实现在 models/pi0.py::compute_loss）：
        # 内部对每个动作维度/时间步采样噪声与时间 t，回归 flow 的速度场，得到逐元素损失。
        # chunked_loss: [B, ah, ad]（对动作 chunk 的逐元素损失），取均值得到标量 loss。
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # 用 step 号 fold 进随机种子，保证每步噪声/时间采样不同且可复现。
    train_rng = jax.random.fold_in(rng, state.step)
    # batch = (Observation, Actions)。observation 含图像/state/tokenized prompt 等；actions: [B, ah, ad]。
    observation, actions = batch

    # Filter out frozen params.
    # 只对"可训练参数"求梯度：DiffState 指定第 0 个入参（model）中 trainable_filter 选中的部分参与微分，
    # freeze 的参数不产生梯度（LoRA/部分冻结时省算力省显存）。
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 一次前向 + 反向，同时拿到标量 loss 和梯度树 grads（结构与可训练参数一致）。
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 取出当前可训练参数，喂给 optimizer 计算更新量。
    params = state.params.filter(config.trainable_filter)
    # optimizer 一步：内部先按 global norm 裁剪梯度，再走 AdamW（含 lr schedule）。返回参数增量与新 opt 状态。
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    # 把增量应用到参数上：new_params = params + updates。
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    # 把更新后的可训练参数写回模型，再取出"完整"参数树（含未训练的 freeze 参数），得到新一步的全量 params。
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # 组装新的 TrainState：步数 +1，参数与 opt 状态替换为新值。
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # 更新 EMA（指数滑动平均）参数：ema = decay*ema + (1-decay)*new。
        # EMA 平滑了训练后期的参数抖动，通常得到泛化更好的权重，存档/推理时优先用它。
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    # 只挑"权重矩阵"（kernel）来统计 param_norm：排除 bias/scale/位置嵌入/输入嵌入，且要求维度>1。
    # 这样 param_norm 更能反映主体权重规模，便于监控训练是否发散。
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # 训练监控指标：损失、梯度全局范数（发散/爆炸预警）、权重全局范数。
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 全局 batch 必须能被设备数整除，否则无法均匀切到各设备做数据并行。
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 开启 JAX 编译缓存，避免每次启动都重新编译，缩短冷启动。
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 主随机种子拆成两路：train_rng 供训练步用，init_rng 供参数初始化用。
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 建设备网格 mesh。fsdp_devices>1 时启用 FSDP（模型分片）；其余维度做数据并行。
    # data_sharding：batch 沿 DATA_AXIS 切分（数据并行）；replicated_sharding：整份复制（如 rng）。
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 准备 checkpoint 目录并判断是否续训（resume）。见 checkpoints.py。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 造 DataLoader（见 data_loader.py）：内部装配好 repack/data/normalize/model 一系列 transform，
    # 迭代产出 (Observation, actions) 且已按 data_sharding 切好片。
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    # 先取一个 batch：一是打印各张量形状做健全性检查，二是给后续 jit 编译提供具体形状。
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # 把首个 batch 的多路相机图像横向拼接后记到 wandb，肉眼确认数据/归一化没接错。
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化 TrainState（含预训练权重灌入或续训占位）；block_until_ready 等待异步计算落地。
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 续训时：从磁盘 checkpoint 恢复真实参数/optimizer 状态/步数。
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 把 train_step 用 jit 编译成 ptrain_step：固定 config，声明输入/输出的 sharding。
    # donate_argnums=(1,) 表示复用 train_state 的缓冲区（旧 state 用完即弃），显著省显存。
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # 起始步：新训为 0；续训则接着 checkpoint 的步数。
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # ===== 主训练循环 =====
    infos = []
    for step in pbar:
        # 在设备网格上下文里执行一步：前向 compute_loss -> 反传 -> 更新 -> EMA，返回新 state 与监控指标。
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        # 每隔 log_interval 步：把这段区间的指标堆叠、取均值，打印并上报 wandb，然后清空缓冲。
        # 攒着一起 reduce 是为了少做几次 device->host 同步，不拖慢训练。
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        # 取下一个 batch（放在 step 计算之后，形成"算当前 / 预取下一"的节奏）。
        batch = next(data_iter)

        # 到达存档间隔、或最后一步时保存 checkpoint（含 params/EMA/optimizer 状态/归一化 assets）。
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # 存档是异步的，退出前等所有写盘任务真正完成。
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
