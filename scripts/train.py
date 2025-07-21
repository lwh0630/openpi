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
import rich

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
    # 定义日志级别到单字符的映射，提高可读性
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        """自定义日志格式器，用于修改日志级别名称"""
        def format(self, record):
            # 将日志级别名称替换为简短的单字符
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # 创建自定义格式器实例
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S", # 时间格式：时:分:秒
    )

    # 获取根日志器并设置其级别和格式器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # 设置日志级别为INFO
    logger.handlers[0].setFormatter(formatter) # 设置第一个处理器（通常是控制台处理器）的格式器


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """
    初始化 Weights & Biases (wandb) 实验跟踪。

    Args:
        config: 训练配置对象。
        resuming: 是否从之前的运行恢复。
        log_code: 是否记录代码。
        enabled: 是否启用 wandb。
    """
    if not enabled:
        wandb.init(mode="disabled") # 如果未启用，则禁用 wandb
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.") # 检查检查点目录是否存在

    if resuming:
        # 如果是恢复模式，从文件中读取 wandb 运行 ID 并恢复
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 否则，初始化新的 wandb 运行并记录配置
        wandb.init(
            name=config.exp_name, # 实验名称
            config=dataclasses.asdict(config), # 将配置对象转换为字典并记录
            project=config.project_name, # 项目名称
        )
        # 将当前运行的 ID 写入文件，以便将来恢复
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        # 如果需要，记录当前目录及其父目录的代码
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """
    加载并验证权重。返回已加载的权重子集。

    Args:
        loader: 权重加载器。
        params_shape: 期望的参数结构（包含形状和 dtype 信息）。

    Returns:
        已加载的参数（仅包含实际加载的值，不包含 ShapeDtypeStruct）。
    """
    loaded_params = loader.load(params_shape) # 使用加载器加载权重
    # 检查加载的参数是否与期望的形状和 dtype 匹配
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 从加载的参数中移除 jax.ShapeDtypeStruct。
    # 这确保只返回实际加载的参数。
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """
    初始化训练状态。

    Args:
        config: 训练配置对象。
        init_rng: 用于初始化的 JAX 随机数生成器键。
        mesh: JAX 分片网格。
        resume: 是否恢复训练。

    Returns:
        一个元组，包含初始化的训练状态和训练状态的分片信息。
    """
    # 创建优化器
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """
        内部初始化函数，用于创建训练状态。

        Args:
            rng: 随机数生成器键。
            partial_params: 部分预加载的参数。

        Returns:
            初始化的训练状态。
        """
        rng, model_rng = jax.random.split(rng) # 分割随机数生成器键
        # 初始化模型（及其参数）
        model = config.model.create(model_rng)

        # 如果提供了部分参数，则将其合并到模型中
        if partial_params is not None:
            graphdef, state = nnx.split(model) # 分割模型的图定义和状态
            # 这将在 partial_params 不是状态子集时产生错误。
            state.replace_by_pure_dict(partial_params) # 用部分参数替换模型状态
            model = nnx.merge(graphdef, state) # 合并回模型

        params = nnx.state(model) # 获取模型的参数
        # 将冻结的参数转换为 bfloat16 类型。
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0, # 初始步数为 0
            params=params, # 模型参数
            model_def=nnx.graphdef(model), # 模型的图定义
            tx=tx, # 优化器
            opt_state=tx.init(params.filter(config.trainable_filter)), # 优化器状态，只针对可训练参数
            ema_decay=config.ema_decay, # EMA 衰减率
            ema_params=None if config.ema_decay is None else params, # EMA 参数（如果启用 EMA）
        )

    # 评估训练状态的形状以确定分片
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True) # 计算 FSDP 分片

    if resume:
        # 如果是恢复模式，只返回形状和分片信息
        return train_state_shape, state_sharding

    # 加载并验证权重，获取部分参数
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    # 定义复制分片，用于输入参数
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化训练状态并混入部分参数
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 释放 partial_params 缓冲区，优化内存
        in_shardings=replicated_sharding, # 输入分片
        out_shardings=state_sharding, # 输出分片
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """
    执行一个训练步骤。

    Args:
        config: 训练配置对象。
        rng: JAX 随机数生成器键。
        state: 当前的训练状态。
        batch: 包含观察和动作的批次数据。

    Returns:
        一个元组，包含更新后的训练状态和包含损失、梯度范数等信息的字典。
    """
    # 将模型定义和参数合并回完整的模型对象
    model = nnx.merge(state.model_def, state.params)
    model.train() # 将模型设置为训练模式

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        """
        损失函数。

        Args:
            model: 模型实例。
            rng: 随机数生成器键。
            observation: 观察数据。
            actions: 动作数据。

        Returns:
            批次损失的均值。
        """
        # 计算分块损失
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss) # 返回损失的均值

    # 为当前训练步骤生成新的随机数键
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch # 解包批次数据

    # 过滤掉冻结的参数，只计算可训练参数的梯度
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 计算损失和梯度
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 获取可训练参数
    params = state.params.filter(config.trainable_filter)
    # 应用梯度更新
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates) # 更新参数

    # 就地更新模型并返回新的完整状态
    nnx.update(model, new_params)
    new_params = nnx.state(model) # 获取更新后的模型参数

    # 更新训练状态
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    # 如果启用了 EMA，则更新 EMA 参数
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 过滤出非偏置、非缩放、非嵌入的核参数
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")), # 排除偏置、缩放、位置/输入嵌入
            lambda _, x: x.value.ndim > 1, # 只选择维度大于 1 的参数（通常是权重矩阵）
        ),
    )
    # 收集并返回训练信息
    info = {
        "loss": loss, # 损失
        "grad_norm": optax.global_norm(grads), # 梯度全局范数
        "param_norm": optax.global_norm(kernel_params), # 核参数的全局范数
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """
    主训练函数。

    Args:
        config: 训练配置对象。
    """
    init_logging() # 初始化日志
    logging.info(f"Running on: {platform.node()}") # 记录当前运行的机器名

    # 检查批次大小是否能被设备数量整除
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 设置 JAX 编译缓存目录
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed) # 创建初始随机数键
    train_rng, init_rng = jax.random.split(rng) # 分割训练和初始化随机数键

    mesh = sharding.make_mesh(config.fsdp_devices) # 创建 JAX 分片网格
    # 定义数据分片策略
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 定义模型复制分片策略
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化检查点管理器并检查是否恢复训练
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    # 初始化 wandb
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建数据加载器
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding, # 数据分片
        shuffle=True, # 启用数据混洗
    )
    data_iter = iter(data_loader) # 获取数据迭代器
    batch = next(data_iter) # 获取第一个批次数据
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}") # 记录数据加载器信息

    # 记录第一个批次中的图像到 wandb 进行 sanity check
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0) # 记录相机视图图像

    # 初始化训练状态 其中 train_state_sharding 指的是神经网络分片
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state) # 等待 JAX 操作完成
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}") # 记录训练状态信息

    if resuming:
        # 如果是恢复模式，从检查点恢复训练状态
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # JIT 编译训练步骤函数
    ptrain_step = jax.jit(
        functools.partial(train_step, config), # 函数应用配置
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding), # 输入分片
        out_shardings=(train_state_sharding, replicated_sharding), # 输出分片
        donate_argnums=(1,), # 捐赠参数，优化内存
    )

    start_step = int(train_state.step) # 获取起始训练步数
    # 初始化进度条
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = [] # 用于存储每个步骤的信息
    for step in pbar:
        with sharding.set_mesh(mesh):
            # 执行训练步骤
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info) # 收集信息
        if step % config.log_interval == 0:
            # 每隔 log_interval 步记录日志
            stacked_infos = common_utils.stack_forest(infos) # 堆叠信息
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos)) # 计算均值并获取到 CPU
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items()) # 格式化信息字符串
            pbar.write(f"Step {step}: {info_str}") # 写入进度条
            wandb.log(reduced_info, step=step) # 记录到 wandb
            infos = [] # 清空信息列表
        batch = next(data_iter) # 获取下一个批次数据

        # 保存检查点
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished() # 等待检查点管理器完成所有保存操作
    rich.print("[green]Done!")

if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用 main 函数并传入通过命令行解析的配置
    main(_config.cli())
